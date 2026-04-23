# PII Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-hosted PII detection/masking service exposed as a Docker-Compose REST API plus a stdio MCP sidecar for Claude Desktop integration.

**Architecture:** FastAPI app with a single-instance HuggingFace model (`openai/privacy-filter`) loaded at startup. Inference runs in a thread pool gated by an `asyncio.Semaphore` to keep memory bounded and the event loop unblocked. Long inputs are auto-chunked using token-aligned `offset_mapping` (no decode round-trip). A separate `mcp_server/` package wraps the REST endpoints as MCP tools.

**Tech Stack:** Python 3.11, FastAPI, Uvicorn, Transformers, PyTorch, Pydantic v2, slowapi (rate limiting), pytest + httpx (testing), Docker Compose.

**Reference docs:**
- Spec: [`docs/superpowers/specs/2026-04-23-pii-filter-design.md`](../specs/2026-04-23-pii-filter-design.md)
- Baseline PRD: [`docs/PRD-pii-filter-service.md`](../../PRD-pii-filter-service.md)

---

## File Layout

Each file has one clear responsibility. Tests sit beside the modules they exercise.

```
pii-filter/
├── .gitignore
├── .env.example
├── README.md
├── Dockerfile
├── docker-compose.yml
├── docker-compose.gpu.yml
├── requirements.txt
├── pytest.ini
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI app + route registration
│   ├── config.py          # Settings (env vars)
│   ├── labels.py          # KNOWN_LABELS set + validate_labels()
│   ├── schemas.py         # Pydantic request/response models
│   ├── model.py           # Model singleton + semaphore-gated inference
│   ├── chunker.py         # Long-text chunking + dedup
│   ├── ratelimit.py       # slowapi limiter setup
│   └── routes/
│       ├── __init__.py
│       ├── health.py      # /health, /ready
│       ├── detect.py      # /detect, /mask
│       └── batch.py       # /detect/batch
└── tests/
    ├── __init__.py
    ├── conftest.py        # Fake model + tokenizer fixtures, TestClient
    ├── test_config.py
    ├── test_labels.py
    ├── test_schemas.py
    ├── test_chunker.py
    ├── test_model.py
    ├── test_health.py
    ├── test_detect.py
    ├── test_mask.py
    ├── test_batch.py
    └── test_ratelimit.py

mcp_server/
├── pyproject.toml
├── README.md
├── pii_filter_mcp/
│   ├── __init__.py
│   ├── __main__.py        # python -m pii_filter_mcp entrypoint
│   └── server.py          # MCP tool definitions
└── tests/
    └── test_tools.py
```

**File-responsibility rules followed:**
- One concern per file (e.g., `labels.py` is only the known-set + validator).
- Routes split by resource family (`health`, `detect`/`mask`, `batch`) so each file stays small.
- `model.py` owns the singleton and the semaphore — nothing else touches the global state.
- The MCP package is fully separate so it can be `pip install`-ed without the FastAPI dep tree.

---

## Working Convention

- Commands assume CWD is the repo root: `/Users/voiceflow/Documents/workspace/2026/privacy-filter`.
- Use a venv: `python3.11 -m venv .venv && source .venv/bin/activate`.
- Each task ends with a commit. Conventional-commits style: `feat(scope): summary` / `chore(scope): summary` / `test(scope): summary` / `docs(scope): summary`.
- Tests use `pytest` and the model is **always mocked** (a real model load is a 3GB download — out of scope for unit tests).
- Tokenizer in tests is the real `gpt2` fast tokenizer (small, fast, gives genuine `offset_mapping` semantics). Tests cache it locally on first run.

---

## Phase 1 — Foundation

### Task 1: Repo scaffolding

**Files:**
- Create: `.gitignore`
- Create: `pytest.ini`
- Create: `requirements.txt`
- Create: `app/__init__.py` (empty)
- Create: `app/routes/__init__.py` (empty)
- Create: `tests/__init__.py` (empty)

- [ ] **Step 1: Initialize the git repo**

```bash
cd /Users/voiceflow/Documents/workspace/2026/privacy-filter
git init
git add docs/
git commit -m "chore: initial PRD and design spec"
```

- [ ] **Step 2: Write `.gitignore`**

```gitignore
.venv/
__pycache__/
*.pyc
.pytest_cache/
.coverage
htmlcov/
.env
model_cache/
*.egg-info/
dist/
build/
.DS_Store
.idea/
.vscode/
```

- [ ] **Step 3: Write `requirements.txt`**

```
fastapi==0.115.0
uvicorn[standard]==0.30.6
transformers==4.45.2
torch==2.4.1
pydantic==2.9.2
pydantic-settings==2.5.2
slowapi==0.1.9
pytest==8.3.3
pytest-asyncio==0.24.0
httpx==0.27.2
```

- [ ] **Step 4: Write `pytest.ini`**

```ini
[pytest]
asyncio_mode = auto
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
filterwarnings =
    ignore::DeprecationWarning
```

- [ ] **Step 5: Create the empty package init files**

```bash
mkdir -p app/routes tests
touch app/__init__.py app/routes/__init__.py tests/__init__.py
```

- [ ] **Step 6: Set up venv and install**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Expected: all packages install successfully.

- [ ] **Step 7: Commit**

```bash
git add .gitignore pytest.ini requirements.txt app/ tests/
git commit -m "chore: scaffold repo (venv config, deps, package layout)"
```

---

### Task 2: Config module

**Files:**
- Create: `app/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

`tests/test_config.py`:
```python
import os
from app.config import Settings


def test_defaults_match_spec(monkeypatch):
    # Clear any env that might bleed in
    for key in ("DEVICE", "DEFAULT_MODE", "MAX_TEXT_LENGTH",
                "CHUNK_SIZE_TOKENS", "CHUNK_OVERLAP_TOKENS", "SMART_SPLIT",
                "RATE_LIMIT_ENABLED", "RATE_LIMIT_PER_IP",
                "MAX_BATCH_SIZE", "MAX_BATCH_TOTAL_TOKENS",
                "MAX_CONCURRENT_INFERENCES", "MODEL_NAME", "LOG_LEVEL"):
        monkeypatch.delenv(key, raising=False)

    s = Settings()
    assert s.device == "cpu"
    assert s.default_mode == "balanced"
    assert s.max_text_length == 524_288
    assert s.chunk_size_tokens == 120_000
    assert s.chunk_overlap_tokens == 512
    assert s.smart_split is True
    assert s.rate_limit_enabled is True
    assert s.rate_limit_per_ip == "60/10minutes"
    assert s.max_batch_size == 32
    assert s.max_batch_total_tokens == 200_000
    assert s.max_concurrent_inferences == 2
    assert s.model_name == "openai/privacy-filter"
    assert s.log_level == "info"


def test_overrides_from_env(monkeypatch):
    monkeypatch.setenv("DEVICE", "cuda")
    monkeypatch.setenv("MAX_BATCH_SIZE", "8")
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
    s = Settings()
    assert s.device == "cuda"
    assert s.max_batch_size == 8
    assert s.rate_limit_enabled is False
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_config.py -v
```

Expected: ImportError — `app.config` doesn't exist yet.

- [ ] **Step 3: Implement `app/config.py`**

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Model
    model_name: str = "openai/privacy-filter"
    device: str = "cpu"
    default_mode: str = "balanced"

    # Limits
    max_text_length: int = 524_288
    chunk_size_tokens: int = 120_000
    chunk_overlap_tokens: int = 512
    smart_split: bool = True

    # Concurrency
    max_concurrent_inferences: int = 2

    # Batch
    max_batch_size: int = 32
    max_batch_total_tokens: int = 200_000

    # Rate limit
    rate_limit_enabled: bool = True
    rate_limit_per_ip: str = "60/10minutes"

    # Logging
    log_level: str = "info"


def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_config.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add app/config.py tests/test_config.py
git commit -m "feat(config): env-driven settings with spec defaults"
```

---

### Task 3: Labels module

**Files:**
- Create: `app/labels.py`
- Create: `tests/test_labels.py`

- [ ] **Step 1: Write the failing test**

`tests/test_labels.py`:
```python
import pytest
from app.labels import KNOWN_LABELS, validate_labels, UnknownLabelError


def test_known_labels_are_the_eight_from_spec():
    assert KNOWN_LABELS == frozenset({
        "private_person", "private_email", "private_phone",
        "private_address", "account_number", "private_url",
        "private_date", "secret",
    })


def test_validate_labels_accepts_none():
    assert validate_labels(None) is None


def test_validate_labels_accepts_empty_list_as_none():
    assert validate_labels([]) is None


def test_validate_labels_returns_frozen_subset():
    out = validate_labels(["private_email", "private_phone"])
    assert out == frozenset({"private_email", "private_phone"})


def test_validate_labels_rejects_unknown():
    with pytest.raises(UnknownLabelError) as exc:
        validate_labels(["private_email", "bogus_label"])
    assert "bogus_label" in str(exc.value)
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
pytest tests/test_labels.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `app/labels.py`**

```python
from typing import Optional, Sequence


KNOWN_LABELS: frozenset[str] = frozenset({
    "private_person",
    "private_email",
    "private_phone",
    "private_address",
    "account_number",
    "private_url",
    "private_date",
    "secret",
})


class UnknownLabelError(ValueError):
    def __init__(self, unknown: list[str]) -> None:
        super().__init__(
            f"Unknown label(s): {sorted(unknown)}. "
            f"Allowed: {sorted(KNOWN_LABELS)}"
        )
        self.unknown = unknown


def validate_labels(labels: Optional[Sequence[str]]) -> Optional[frozenset[str]]:
    if not labels:
        return None
    requested = set(labels)
    unknown = sorted(requested - KNOWN_LABELS)
    if unknown:
        raise UnknownLabelError(unknown)
    return frozenset(requested)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_labels.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add app/labels.py tests/test_labels.py
git commit -m "feat(labels): known label set with validator"
```

---

## Phase 2 — Schemas, Model, Inference

### Task 4: Pydantic schemas

**Files:**
- Create: `app/schemas.py`
- Create: `tests/test_schemas.py`

- [ ] **Step 1: Write the failing test**

`tests/test_schemas.py`:
```python
import pytest
from pydantic import ValidationError
from app.schemas import (
    DetectRequest, DetectResponse, MaskRequest, MaskResponse,
    Entity, Mode, BatchRequest, BatchItem,
)


def test_detect_request_minimal():
    r = DetectRequest(text="hello")
    assert r.text == "hello"
    assert r.mode == "balanced"
    assert r.mask is False
    assert r.mask_char == "[REDACTED]"
    assert r.labels is None


def test_detect_request_rejects_invalid_mode():
    with pytest.raises(ValidationError):
        DetectRequest(text="x", mode="aggressive")


def test_entity_shape():
    e = Entity(label="private_email", start=0, end=5, text="hello", score=0.9)
    assert e.label == "private_email"
    assert e.score == 0.9


def test_batch_request_requires_items():
    with pytest.raises(ValidationError):
        BatchRequest(items=[])  # min length 1


def test_batch_item_inherits_detect_fields():
    item = BatchItem(text="hi", labels=["private_email"])
    assert item.labels == ["private_email"]
    assert item.mode == "balanced"
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
pytest tests/test_schemas.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `app/schemas.py`**

```python
from typing import Literal, Optional
from pydantic import BaseModel, Field, ConfigDict


Mode = Literal["precise", "balanced", "recall"]


class Entity(BaseModel):
    label: str
    start: int = Field(..., ge=0)
    end: int = Field(..., ge=0)
    text: str
    score: float = Field(..., ge=0.0, le=1.0)


class DetectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1)
    mode: Mode = "balanced"
    mask: bool = False
    mask_char: str = "[REDACTED]"
    labels: Optional[list[str]] = None


class DetectMeta(BaseModel):
    model: str
    mode: Mode
    entity_count: int
    processing_ms: int
    chunks_processed: Optional[int] = None
    input_tokens: Optional[int] = None


class DetectResponse(BaseModel):
    text: str
    entities: list[Entity]
    masked_text: Optional[str] = None
    meta: DetectMeta


class MaskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1)
    mode: Mode = "balanced"
    mask_char: str = "[REDACTED]"
    labels: Optional[list[str]] = None


class MaskResponse(BaseModel):
    masked_text: str
    entity_count: int
    processing_ms: int


class BatchItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1)
    mode: Mode = "balanced"
    mask_char: str = "[REDACTED]"
    labels: Optional[list[str]] = None


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[BatchItem] = Field(..., min_length=1)
    mask: bool = False


class BatchItemError(BaseModel):
    code: str
    message: str


class BatchItemResult(BaseModel):
    status: Literal["ok", "error"]
    entities: Optional[list[Entity]] = None
    masked_text: Optional[str] = None
    meta: Optional[dict] = None
    error: Optional[BatchItemError] = None


class BatchMeta(BaseModel):
    model: str
    batch_size: int
    processing_ms: int


class BatchResponse(BaseModel):
    results: list[BatchItemResult]
    meta: BatchMeta
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_schemas.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add app/schemas.py tests/test_schemas.py
git commit -m "feat(schemas): pydantic request/response models for detect, mask, batch"
```

---

### Task 5: Test fixtures (fake model + tokenizer)

**Files:**
- Create: `tests/conftest.py`

- [ ] **Step 1: Write the conftest**

`tests/conftest.py`:
```python
"""
Shared test fixtures.

Strategy:
- Use real `gpt2` fast tokenizer for offset_mapping correctness in chunker tests.
  It is small, downloads once, then uses HF cache.
- Mock the inference function so tests don't need the real 3GB privacy-filter model.
"""
from __future__ import annotations

import asyncio
from typing import Callable, Optional

import pytest
from transformers import AutoTokenizer


@pytest.fixture(scope="session")
def fast_tokenizer():
    tok = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    assert tok.is_fast
    return tok


@pytest.fixture
def fake_inference():
    """
    Returns a callable matching the real `run_inference(text, mode) -> list[dict]`
    signature. Default behavior: detect every 'alice@example.com' as private_email
    and every 'Alice Smith' as private_person. Tests can replace .impl for custom logic.
    """
    class _Fake:
        impl: Callable[[str, str], list[dict]] = staticmethod(
            lambda text, mode: _default_detect(text)
        )

        def __call__(self, text: str, mode: str = "balanced") -> list[dict]:
            return self.impl(text, mode)

    return _Fake()


def _default_detect(text: str) -> list[dict]:
    out = []
    needles = [
        ("Alice Smith", "private_person", 0.97),
        ("alice@example.com", "private_email", 0.99),
        ("+33 6 12 34 56 78", "private_phone", 0.95),
    ]
    for needle, label, score in needles:
        start = 0
        while True:
            idx = text.find(needle, start)
            if idx == -1:
                break
            out.append({
                "label": label,
                "start": idx,
                "end": idx + len(needle),
                "text": needle,
                "score": score,
            })
            start = idx + len(needle)
    return out


@pytest.fixture
def patch_model(monkeypatch, fast_tokenizer, fake_inference):
    """
    Patches app.model so the API layer thinks the model is loaded
    and routes inference calls through fake_inference.
    """
    from app import model as model_module

    state = model_module.ModelState(
        tokenizer=fast_tokenizer,
        run_inference=fake_inference,
        device="cpu",
        model_name="test/fake",
        loaded=True,
    )
    monkeypatch.setattr(model_module, "_state", state)
    # Reset semaphore for each test
    monkeypatch.setattr(model_module, "_semaphore", asyncio.Semaphore(2))
    return state
```

- [ ] **Step 2: Commit**

(No tests run here — fixtures are only useful via consumers in later tasks.)

```bash
git add tests/conftest.py
git commit -m "test: add fake tokenizer and inference fixtures"
```

---

### Task 6: Model singleton + inference semaphore

**Files:**
- Create: `app/model.py`
- Create: `tests/test_model.py`

- [ ] **Step 1: Write the failing test**

`tests/test_model.py`:
```python
import asyncio
import pytest
from app import model as model_module


@pytest.fixture
def reset_state(monkeypatch):
    monkeypatch.setattr(model_module, "_state", None)
    monkeypatch.setattr(model_module, "_semaphore", asyncio.Semaphore(2))


def test_get_state_raises_when_not_loaded(reset_state):
    with pytest.raises(model_module.ModelNotLoadedError):
        model_module.get_state()


def test_state_is_set_and_returned(monkeypatch, fast_tokenizer):
    state = model_module.ModelState(
        tokenizer=fast_tokenizer,
        run_inference=lambda t, m: [],
        device="cpu",
        model_name="x/y",
        loaded=True,
    )
    monkeypatch.setattr(model_module, "_state", state)
    assert model_module.get_state() is state


async def test_run_inference_async_uses_semaphore(monkeypatch, fast_tokenizer):
    """Verify the semaphore caps in-flight calls."""
    sem = asyncio.Semaphore(1)
    monkeypatch.setattr(model_module, "_semaphore", sem)

    in_flight = 0
    max_in_flight = 0

    def slow(text, mode):
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        # Simulate work via blocking sleep (we're inside to_thread)
        import time; time.sleep(0.05)
        in_flight -= 1
        return []

    state = model_module.ModelState(
        tokenizer=fast_tokenizer,
        run_inference=slow,
        device="cpu",
        model_name="x/y",
        loaded=True,
    )
    monkeypatch.setattr(model_module, "_state", state)

    await asyncio.gather(*[
        model_module.run_inference_async("foo", "balanced")
        for _ in range(5)
    ])
    assert max_in_flight == 1
```

- [ ] **Step 2: Run test to confirm failure**

```bash
pytest tests/test_model.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `app/model.py`**

```python
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable, Optional

from app.config import get_settings


class ModelNotLoadedError(RuntimeError):
    pass


@dataclass
class ModelState:
    tokenizer: object
    run_inference: Callable[[str, str], list[dict]]
    device: str
    model_name: str
    loaded: bool


_state: Optional[ModelState] = None
_semaphore: asyncio.Semaphore = asyncio.Semaphore(
    get_settings().max_concurrent_inferences
)


def get_state() -> ModelState:
    if _state is None or not _state.loaded:
        raise ModelNotLoadedError("Model not yet loaded")
    return _state


def is_loaded() -> bool:
    return _state is not None and _state.loaded


async def run_inference_async(text: str, mode: str) -> list[dict]:
    state = get_state()
    async with _semaphore:
        return await asyncio.to_thread(state.run_inference, text, mode)


def load_model() -> None:
    """
    Load the real privacy-filter model. Called from FastAPI startup.
    Kept thin so tests can swap the global state directly.
    """
    global _state, _semaphore
    settings = get_settings()

    from transformers import AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(settings.model_name, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError(
            f"Tokenizer for {settings.model_name} is not a fast tokenizer; "
            "offset_mapping is required for chunker correctness."
        )

    pipe = pipeline(
        "token-classification",
        model=settings.model_name,
        tokenizer=tokenizer,
        device=0 if settings.device == "cuda" else -1,
        aggregation_strategy="simple",
    )

    def _run(text: str, mode: str) -> list[dict]:
        # `mode` may be folded into pipeline params in a future model version.
        # For v1 it is recorded but does not change inference parameters.
        raw = pipe(text)
        return [
            {
                "label": r["entity_group"],
                "start": int(r["start"]),
                "end": int(r["end"]),
                "text": r["word"],
                "score": float(r["score"]),
            }
            for r in raw
        ]

    _state = ModelState(
        tokenizer=tokenizer,
        run_inference=_run,
        device=settings.device,
        model_name=settings.model_name,
        loaded=True,
    )
    _semaphore = asyncio.Semaphore(settings.max_concurrent_inferences)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_model.py -v
```

Expected: 3 passed (the third is `async`, pytest-asyncio handles it via `asyncio_mode = auto`).

- [ ] **Step 5: Commit**

```bash
git add app/model.py tests/test_model.py
git commit -m "feat(model): singleton state + asyncio semaphore + threadpool inference"
```

---

## Phase 3 — Chunker

### Task 7: Chunker — single-pass fast path

**Files:**
- Create: `app/chunker.py`
- Create: `tests/test_chunker.py`

- [ ] **Step 1: Write the failing test**

`tests/test_chunker.py`:
```python
from app.chunker import detect_with_chunking


def test_short_text_takes_fast_path(fast_tokenizer, fake_inference):
    text = "Hello, my name is Alice Smith."
    result = detect_with_chunking(
        text=text,
        tokenizer=fast_tokenizer,
        run_inference=fake_inference,
        mode="balanced",
        chunk_size_tokens=120_000,
        overlap_tokens=512,
        smart_split=False,
    )
    assert result.chunks_processed == 1
    assert result.input_tokens == len(fast_tokenizer.encode(text, add_special_tokens=False))
    assert any(e["label"] == "private_person" and e["text"] == "Alice Smith"
               for e in result.entities)


def test_short_text_offsets_match_source(fast_tokenizer, fake_inference):
    text = "Email me at alice@example.com please."
    result = detect_with_chunking(
        text=text, tokenizer=fast_tokenizer, run_inference=fake_inference,
        mode="balanced", chunk_size_tokens=120_000, overlap_tokens=512,
        smart_split=False,
    )
    e = next(x for x in result.entities if x["label"] == "private_email")
    assert text[e["start"]:e["end"]] == "alice@example.com"
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
pytest tests/test_chunker.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `app/chunker.py` (fast-path only for now)**

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ChunkResult:
    entities: list[dict]
    chunks_processed: int
    input_tokens: int


def detect_with_chunking(
    text: str,
    tokenizer,
    run_inference: Callable[[str, str], list[dict]],
    mode: str,
    chunk_size_tokens: int,
    overlap_tokens: int,
    smart_split: bool,
) -> ChunkResult:
    """
    Detect PII with automatic chunking for long inputs.

    Uses tokenizer.offset_mapping so chunk slicing happens against the
    original text — no decode round-trip, no offset drift.
    """
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    tokens = encoding["input_ids"]
    offsets = encoding["offset_mapping"]
    n_tokens = len(tokens)

    if n_tokens <= chunk_size_tokens:
        entities = run_inference(text, mode)
        return ChunkResult(entities=entities, chunks_processed=1, input_tokens=n_tokens)

    raise NotImplementedError("Multi-chunk path lands in Task 8")
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_chunker.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add app/chunker.py tests/test_chunker.py
git commit -m "feat(chunker): fast-path single-pass detection with offset_mapping"
```

---

### Task 8: Chunker — multi-chunk slicing + dedup

**Files:**
- Modify: `app/chunker.py`
- Modify: `tests/test_chunker.py`

- [ ] **Step 1: Add failing tests for multi-chunk**

Append to `tests/test_chunker.py`:
```python
def _make_long_text(filler_tokens: int, fast_tokenizer, sentinel: str) -> str:
    """Build text with > N tokens by repeating a phrase, with `sentinel` at a known location."""
    base = "The quick brown fox jumps over the lazy dog. "
    chunk_text = base * (filler_tokens // 10)  # ~10 tokens per repeat
    inserted = chunk_text + sentinel + " " + chunk_text
    return inserted


def test_multi_chunk_processes_more_than_one_chunk(fast_tokenizer, fake_inference):
    text = _make_long_text(2_000, fast_tokenizer, "Email me at alice@example.com.")
    # Force chunking with a small chunk size
    result = detect_with_chunking(
        text=text, tokenizer=fast_tokenizer, run_inference=fake_inference,
        mode="balanced", chunk_size_tokens=1_000, overlap_tokens=64,
        smart_split=False,
    )
    assert result.chunks_processed >= 2
    # The email should still be found and offsets should map back to source
    emails = [e for e in result.entities if e["label"] == "private_email"]
    assert emails, "Expected at least one private_email entity"
    for e in emails:
        assert text[e["start"]:e["end"]] == "alice@example.com"


def test_overlap_dedup_collapses_duplicate_spans(fast_tokenizer, fake_inference):
    """A PII span sitting in the overlap window between two chunks must
    appear exactly once in the merged result."""
    # Build text where the email lives near a chunk boundary
    head = "x " * 500
    tail = " y" * 500
    text = head + "alice@example.com" + tail
    result = detect_with_chunking(
        text=text, tokenizer=fast_tokenizer, run_inference=fake_inference,
        mode="balanced", chunk_size_tokens=600, overlap_tokens=200,
        smart_split=False,
    )
    emails = [e for e in result.entities if e["label"] == "private_email"]
    assert len(emails) == 1
    e = emails[0]
    assert text[e["start"]:e["end"]] == "alice@example.com"


def test_dedup_keeps_higher_score():
    from app.chunker import deduplicate_spans
    spans = [
        {"label": "private_email", "start": 0, "end": 5, "text": "x", "score": 0.7},
        {"label": "private_email", "start": 0, "end": 5, "text": "x", "score": 0.95},
    ]
    out = deduplicate_spans(spans)
    assert len(out) == 1
    assert out[0]["score"] == 0.95


def test_dedup_keeps_different_labels_overlapping():
    from app.chunker import deduplicate_spans
    spans = [
        {"label": "private_email", "start": 0, "end": 10, "text": "x", "score": 0.9},
        {"label": "private_phone", "start": 5, "end": 12, "text": "y", "score": 0.9},
    ]
    out = deduplicate_spans(spans)
    assert len(out) == 2
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_chunker.py -v
```

Expected: 4 new tests fail (NotImplementedError, ImportError on `deduplicate_spans`).

- [ ] **Step 3: Implement multi-chunk path + dedup**

Replace `app/chunker.py`:
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable


@dataclass
class ChunkResult:
    entities: list[dict]
    chunks_processed: int
    input_tokens: int


def detect_with_chunking(
    text: str,
    tokenizer,
    run_inference: Callable[[str, str], list[dict]],
    mode: str,
    chunk_size_tokens: int,
    overlap_tokens: int,
    smart_split: bool,
) -> ChunkResult:
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    tokens = encoding["input_ids"]
    offsets = encoding["offset_mapping"]
    n_tokens = len(tokens)

    if n_tokens <= chunk_size_tokens:
        entities = run_inference(text, mode)
        return ChunkResult(entities=entities, chunks_processed=1, input_tokens=n_tokens)

    step = chunk_size_tokens - overlap_tokens
    if step <= 0:
        raise ValueError("chunk_size_tokens must be greater than overlap_tokens")

    all_entities: list[dict] = []
    chunks_processed = 0

    start_tok = 0
    while start_tok < n_tokens:
        end_tok = min(start_tok + chunk_size_tokens, n_tokens)
        char_start = offsets[start_tok][0]
        char_end = offsets[end_tok - 1][1]

        # smart_split nudges char boundaries to nearest paragraph break (best-effort)
        if smart_split:
            char_start, char_end = _nudge_to_paragraph(text, offsets,
                                                      start_tok, end_tok,
                                                      char_start, char_end)

        chunk_text = text[char_start:char_end]
        spans = run_inference(chunk_text, mode)
        for s in spans:
            all_entities.append({
                **s,
                "start": s["start"] + char_start,
                "end": s["end"] + char_start,
            })
        chunks_processed += 1
        if end_tok >= n_tokens:
            break
        start_tok += step

    deduped = deduplicate_spans(all_entities)
    return ChunkResult(entities=deduped, chunks_processed=chunks_processed,
                       input_tokens=n_tokens)


def deduplicate_spans(entities: Iterable[dict]) -> list[dict]:
    """
    Collapse spans that share a label and overlap in character range,
    keeping the one with the higher score.
    """
    sorted_ents = sorted(entities, key=lambda e: (e["start"], -e["score"]))
    kept: list[dict] = []
    for e in sorted_ents:
        absorbed = False
        for k in kept:
            if k["label"] == e["label"] and e["start"] < k["end"] and e["end"] > k["start"]:
                # Overlap with same label: keep higher score
                if e["score"] > k["score"]:
                    k.update(e)
                absorbed = True
                break
        if not absorbed:
            kept.append(dict(e))
    kept.sort(key=lambda e: e["start"])
    return kept


def _nudge_to_paragraph(text: str, offsets, start_tok: int, end_tok: int,
                        char_start: int, char_end: int,
                        token_window: int = 200) -> tuple[int, int]:
    """
    Try to align the chunk's end (and next-chunk's start) to the nearest "\n\n"
    within ±token_window tokens of the calculated split. Best-effort:
    if no paragraph break exists in range, falls back to the calculated boundaries.
    """
    # Search a character window ~ token_window tokens around char_end
    win_start_tok = max(0, end_tok - token_window)
    win_end_tok = min(len(offsets), end_tok + token_window)
    if win_end_tok > win_start_tok:
        win_char_start = offsets[win_start_tok][0]
        win_char_end = offsets[win_end_tok - 1][1]
        # Look for \n\n inside the window, prefer the one closest to char_end
        best = -1
        idx = text.find("\n\n", win_char_start, win_char_end)
        while idx != -1:
            if best == -1 or abs(idx - char_end) < abs(best - char_end):
                best = idx
            idx = text.find("\n\n", idx + 2, win_char_end)
        if best != -1:
            char_end = best
    return char_start, char_end
```

- [ ] **Step 4: Run all chunker tests**

```bash
pytest tests/test_chunker.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add app/chunker.py tests/test_chunker.py
git commit -m "feat(chunker): multi-chunk slicing, smart-split nudge, span dedup"
```

---

### Task 9: Chunker — multibyte text correctness

**Files:**
- Modify: `tests/test_chunker.py`

- [ ] **Step 1: Add multibyte stress test**

Append to `tests/test_chunker.py`:
```python
def test_offsets_are_correct_with_cjk_and_emoji(fast_tokenizer, fake_inference):
    """Spans returned for chunked input must slice the original text correctly
    even when chunks straddle multibyte characters."""
    head = "你好世界 " * 200            # CJK
    middle = "🎉🚀✨ " * 100             # emoji
    text = head + "alice@example.com" + middle + head
    result = detect_with_chunking(
        text=text, tokenizer=fast_tokenizer, run_inference=fake_inference,
        mode="balanced", chunk_size_tokens=400, overlap_tokens=64,
        smart_split=False,
    )
    emails = [e for e in result.entities if e["label"] == "private_email"]
    assert emails, "Expected the email to be detected"
    for e in emails:
        assert text[e["start"]:e["end"]] == "alice@example.com", (
            f"Offset drift: text[{e['start']}:{e['end']}]="
            f"{text[e['start']:e['end']]!r}"
        )
```

- [ ] **Step 2: Run**

```bash
pytest tests/test_chunker.py::test_offsets_are_correct_with_cjk_and_emoji -v
```

Expected: pass on first run (offset_mapping is the whole point — there's no fix needed if Task 8 was implemented correctly).

If it fails: investigate offset usage in the multi-chunk path before proceeding. The fix should not introduce a `decode()` round-trip.

- [ ] **Step 3: Commit**

```bash
git add tests/test_chunker.py
git commit -m "test(chunker): multibyte (CJK+emoji) offset correctness"
```

---

## Phase 4 — API Surface

### Task 10: FastAPI app skeleton + /health + /ready

**Files:**
- Create: `app/routes/health.py`
- Create: `app/main.py`
- Create: `tests/test_health.py`

- [ ] **Step 1: Write the failing test**

`tests/test_health.py`:
```python
from fastapi.testclient import TestClient
import asyncio
import pytest

from app.main import create_app
from app import model as model_module


@pytest.fixture
def client_unloaded(monkeypatch):
    monkeypatch.setattr(model_module, "_state", None)
    monkeypatch.setattr(model_module, "_semaphore", asyncio.Semaphore(2))
    app = create_app(load_at_startup=False)
    return TestClient(app)


@pytest.fixture
def client_loaded(patch_model):
    app = create_app(load_at_startup=False)
    return TestClient(app)


def test_health_always_200_when_unloaded(client_unloaded):
    r = client_unloaded.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is False


def test_health_reports_loaded_when_ready(client_loaded):
    r = client_loaded.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["model_loaded"] is True
    assert body["device"] == "cpu"


def test_ready_503_when_unloaded(client_unloaded):
    r = client_unloaded.get("/ready")
    assert r.status_code == 503


def test_ready_200_when_loaded(client_loaded):
    r = client_loaded.get("/ready")
    assert r.status_code == 200
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_health.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `app/routes/health.py`**

```python
from fastapi import APIRouter, Response

from app import model as model_module


router = APIRouter()


@router.get("/health")
def health():
    if model_module.is_loaded():
        state = model_module.get_state()
        return {"status": "ok", "model_loaded": True, "device": state.device}
    return {"status": "ok", "model_loaded": False, "device": None}


@router.get("/ready")
def ready(response: Response):
    if model_module.is_loaded():
        return {"status": "ready"}
    response.status_code = 503
    return {"status": "loading"}
```

- [ ] **Step 4: Implement `app/main.py`**

```python
from __future__ import annotations

from contextlib import asynccontextmanager
from fastapi import FastAPI

from app import model as model_module
from app.routes import health as health_routes


def create_app(load_at_startup: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if load_at_startup:
            model_module.load_model()
        yield

    app = FastAPI(
        title="pii-filter",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.include_router(health_routes.router)
    return app


app = create_app(load_at_startup=True)
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_health.py -v
```

Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add app/main.py app/routes/health.py tests/test_health.py
git commit -m "feat(api): app skeleton with /health and /ready"
```

---

### Task 11: /detect endpoint

**Files:**
- Create: `app/routes/detect.py`
- Modify: `app/main.py`
- Create: `tests/test_detect.py`

- [ ] **Step 1: Write the failing test**

`tests/test_detect.py`:
```python
from fastapi.testclient import TestClient
import pytest

from app.main import create_app


@pytest.fixture
def client(patch_model):
    return TestClient(create_app(load_at_startup=False))


def test_detect_returns_entities(client):
    r = client.post("/detect", json={"text": "My name is Alice Smith."})
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "My name is Alice Smith."
    labels = [e["label"] for e in body["entities"]]
    assert "private_person" in labels
    assert body["meta"]["entity_count"] == len(body["entities"])
    assert body["meta"]["model"] == "test/fake"
    assert body["masked_text"] is None


def test_detect_with_mask_returns_masked_text(client):
    r = client.post("/detect", json={
        "text": "My name is Alice Smith.",
        "mask": True,
    })
    body = r.json()
    assert body["masked_text"] == "My name is [REDACTED]."


def test_detect_with_custom_mask_char(client):
    r = client.post("/detect", json={
        "text": "Email me at alice@example.com please.",
        "mask": True,
        "mask_char": "***",
    })
    assert r.json()["masked_text"] == "Email me at *** please."


def test_detect_with_label_filter_returns_only_requested(client):
    r = client.post("/detect", json={
        "text": "Alice Smith wrote alice@example.com",
        "labels": ["private_email"],
    })
    body = r.json()
    labels = {e["label"] for e in body["entities"]}
    assert labels == {"private_email"}


def test_detect_unknown_label_is_422(client):
    r = client.post("/detect", json={"text": "x", "labels": ["bogus"]})
    assert r.status_code == 422
    assert "bogus" in r.text


def test_detect_returns_503_when_model_not_loaded(monkeypatch):
    from app import model as model_module
    monkeypatch.setattr(model_module, "_state", None)
    client = TestClient(create_app(load_at_startup=False))
    r = client.post("/detect", json={"text": "x"})
    assert r.status_code == 503


def test_detect_text_too_long_is_422(client, monkeypatch):
    from app.config import Settings
    # Force a tiny limit for this test
    monkeypatch.setattr("app.routes.detect.get_settings",
                        lambda: Settings(max_text_length=10))
    r = client.post("/detect", json={"text": "x" * 50})
    assert r.status_code == 422
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_detect.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `app/routes/detect.py`**

```python
from __future__ import annotations

import time
from fastapi import APIRouter, HTTPException, status

from app.chunker import detect_with_chunking
from app.config import get_settings
from app.labels import UnknownLabelError, validate_labels
from app.model import (
    ModelNotLoadedError, get_state, run_inference_async,
)
from app.schemas import (
    DetectMeta, DetectRequest, DetectResponse, Entity, MaskRequest, MaskResponse,
)


router = APIRouter()


def _apply_mask(text: str, entities: list[Entity], mask_char: str) -> str:
    # Apply right-to-left so earlier offsets stay valid
    parts = sorted(entities, key=lambda e: e.start, reverse=True)
    out = text
    for e in parts:
        out = out[: e.start] + mask_char + out[e.end :]
    return out


def _filter_by_labels(spans: list[dict], allowed: frozenset[str] | None) -> list[dict]:
    if allowed is None:
        return spans
    return [s for s in spans if s["label"] in allowed]


@router.post("/detect", response_model=DetectResponse)
async def detect(req: DetectRequest) -> DetectResponse:
    settings = get_settings()
    if len(req.text) > settings.max_text_length:
        raise HTTPException(
            status_code=422,
            detail=f"text exceeds MAX_TEXT_LENGTH ({settings.max_text_length})",
        )
    try:
        allowed = validate_labels(req.labels)
    except UnknownLabelError as e:
        raise HTTPException(status_code=422, detail=str(e))

    try:
        state = get_state()
    except ModelNotLoadedError:
        raise HTTPException(status_code=503, detail="model not loaded")

    started = time.perf_counter()

    # Run inference (with chunking) inside the thread pool
    import asyncio

    def _do() -> "_InferenceResult":
        result = detect_with_chunking(
            text=req.text,
            tokenizer=state.tokenizer,
            run_inference=lambda t, m: state.run_inference(t, m),
            mode=req.mode,
            chunk_size_tokens=settings.chunk_size_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
            smart_split=settings.smart_split,
        )
        return result

    # Use the same semaphore-gated pathway
    from app import model as model_module
    async with model_module._semaphore:
        result = await asyncio.to_thread(_do)

    spans = _filter_by_labels(result.entities, allowed)
    entities = [Entity(**s) for s in spans]
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    meta = DetectMeta(
        model=state.model_name,
        mode=req.mode,
        entity_count=len(entities),
        processing_ms=elapsed_ms,
        chunks_processed=result.chunks_processed if result.chunks_processed > 1 else None,
        input_tokens=result.input_tokens if result.chunks_processed > 1 else None,
    )

    masked_text = _apply_mask(req.text, entities, req.mask_char) if req.mask else None

    return DetectResponse(
        text=req.text,
        entities=entities,
        masked_text=masked_text,
        meta=meta,
    )
```

(The `/mask` endpoint goes in Task 12 in the same file.)

- [ ] **Step 4: Wire router in `app/main.py`**

Modify `app/main.py`. Replace:
```python
from app.routes import health as health_routes
```
with:
```python
from app.routes import detect as detect_routes
from app.routes import health as health_routes
```

And inside `create_app`:
```python
    app.include_router(health_routes.router)
    app.include_router(detect_routes.router)
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_detect.py -v
```

Expected: 7 passed.

- [ ] **Step 6: Commit**

```bash
git add app/routes/detect.py app/main.py tests/test_detect.py
git commit -m "feat(api): /detect endpoint with chunking, label filter, masking"
```

---

### Task 12: /mask endpoint

**Files:**
- Modify: `app/routes/detect.py`
- Create: `tests/test_mask.py`

- [ ] **Step 1: Write the failing test**

`tests/test_mask.py`:
```python
from fastapi.testclient import TestClient
import pytest
from app.main import create_app


@pytest.fixture
def client(patch_model):
    return TestClient(create_app(load_at_startup=False))


def test_mask_returns_only_masked_text(client):
    r = client.post("/mask", json={"text": "Call alice@example.com tonight."})
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"masked_text", "entity_count", "processing_ms"}
    assert body["masked_text"] == "Call [REDACTED] tonight."
    assert body["entity_count"] == 1


def test_mask_respects_label_filter(client):
    r = client.post("/mask", json={
        "text": "Alice Smith wrote alice@example.com",
        "labels": ["private_email"],
    })
    body = r.json()
    # Person not in label filter, so left alone; email masked.
    assert body["masked_text"] == "Alice Smith wrote [REDACTED]"
    assert body["entity_count"] == 1


def test_mask_unknown_label_is_422(client):
    r = client.post("/mask", json={"text": "x", "labels": ["bogus"]})
    assert r.status_code == 422
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_mask.py -v
```

Expected: 404 (route not registered yet).

- [ ] **Step 3: Add the /mask handler to `app/routes/detect.py`**

Append to `app/routes/detect.py`:
```python
@router.post("/mask", response_model=MaskResponse)
async def mask(req: MaskRequest) -> MaskResponse:
    # Reuse /detect logic via a synthetic DetectRequest with mask=True
    detect_req = DetectRequest(
        text=req.text,
        mode=req.mode,
        mask=True,
        mask_char=req.mask_char,
        labels=req.labels,
    )
    detect_resp = await detect(detect_req)
    return MaskResponse(
        masked_text=detect_resp.masked_text or req.text,
        entity_count=detect_resp.meta.entity_count,
        processing_ms=detect_resp.meta.processing_ms,
    )
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_mask.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add app/routes/detect.py tests/test_mask.py
git commit -m "feat(api): /mask convenience endpoint"
```

---

### Task 13: /detect/batch endpoint

**Files:**
- Create: `app/routes/batch.py`
- Modify: `app/main.py`
- Create: `tests/test_batch.py`

- [ ] **Step 1: Write the failing test**

`tests/test_batch.py`:
```python
from fastapi.testclient import TestClient
import pytest
from app.main import create_app


@pytest.fixture
def client(patch_model):
    return TestClient(create_app(load_at_startup=False))


def test_batch_happy_path(client):
    r = client.post("/detect/batch", json={
        "items": [
            {"text": "Hello Alice Smith"},
            {"text": "alice@example.com"},
        ],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["meta"]["batch_size"] == 2
    assert len(body["results"]) == 2
    assert body["results"][0]["status"] == "ok"
    assert body["results"][1]["status"] == "ok"
    assert {e["label"] for e in body["results"][0]["entities"]} == {"private_person"}


def test_batch_with_mask_top_level(client):
    r = client.post("/detect/batch", json={
        "items": [{"text": "alice@example.com"}],
        "mask": True,
    })
    body = r.json()
    assert body["results"][0]["masked_text"] == "[REDACTED]"


def test_batch_per_item_label_filter(client):
    r = client.post("/detect/batch", json={
        "items": [
            {"text": "Alice Smith and alice@example.com",
             "labels": ["private_email"]},
        ],
    })
    body = r.json()
    labels = {e["label"] for e in body["results"][0]["entities"]}
    assert labels == {"private_email"}


def test_batch_size_exceeds_limit(client, monkeypatch):
    from app.config import Settings
    monkeypatch.setattr("app.routes.batch.get_settings",
                        lambda: Settings(max_batch_size=2))
    r = client.post("/detect/batch", json={
        "items": [{"text": "x"}, {"text": "y"}, {"text": "z"}],
    })
    assert r.status_code == 422
    assert "batch" in r.text.lower()


def test_batch_total_token_budget_exceeded(client, monkeypatch):
    from app.config import Settings
    monkeypatch.setattr("app.routes.batch.get_settings",
                        lambda: Settings(max_batch_total_tokens=10))
    r = client.post("/detect/batch", json={
        "items": [{"text": "the quick brown fox " * 50}],
    })
    assert r.status_code == 422
    assert "token" in r.text.lower()


def test_batch_per_item_too_long_returns_partial_error(client, monkeypatch):
    """An item exceeding chunk size yields a per-item 'item_too_long' error;
    other items still succeed."""
    from app.config import Settings
    monkeypatch.setattr(
        "app.routes.batch.get_settings",
        lambda: Settings(chunk_size_tokens=20,
                         max_batch_total_tokens=10_000),
    )
    r = client.post("/detect/batch", json={
        "items": [
            {"text": "alice@example.com"},                # short
            {"text": "the quick brown fox " * 100},       # too long
        ],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["results"][0]["status"] == "ok"
    assert body["results"][1]["status"] == "error"
    assert body["results"][1]["error"]["code"] == "item_too_long"


def test_batch_503_when_model_not_loaded(monkeypatch):
    from app import model as model_module
    monkeypatch.setattr(model_module, "_state", None)
    client = TestClient(create_app(load_at_startup=False))
    r = client.post("/detect/batch", json={"items": [{"text": "x"}]})
    assert r.status_code == 503


def test_batch_unknown_label_is_per_item_error_not_request_error(client):
    """Per spec: per-item label issues return as item-level errors, not 422 for the whole batch."""
    r = client.post("/detect/batch", json={
        "items": [{"text": "x", "labels": ["bogus"]}],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["results"][0]["status"] == "error"
    assert "bogus" in body["results"][0]["error"]["message"]
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_batch.py -v
```

Expected: 404 / ImportError.

- [ ] **Step 3: Implement `app/routes/batch.py`**

```python
from __future__ import annotations

import asyncio
import time
from fastapi import APIRouter, HTTPException

from app import model as model_module
from app.chunker import detect_with_chunking
from app.config import get_settings
from app.labels import UnknownLabelError, validate_labels
from app.model import ModelNotLoadedError, get_state
from app.schemas import (
    BatchItem, BatchItemError, BatchItemResult, BatchMeta, BatchRequest,
    BatchResponse, Entity,
)


router = APIRouter()


def _apply_mask(text: str, entities: list[Entity], mask_char: str) -> str:
    parts = sorted(entities, key=lambda e: e.start, reverse=True)
    out = text
    for e in parts:
        out = out[: e.start] + mask_char + out[e.end :]
    return out


def _count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


@router.post("/detect/batch", response_model=BatchResponse)
async def detect_batch(req: BatchRequest) -> BatchResponse:
    settings = get_settings()

    if len(req.items) > settings.max_batch_size:
        raise HTTPException(
            status_code=422,
            detail=f"batch size {len(req.items)} exceeds MAX_BATCH_SIZE "
                   f"({settings.max_batch_size})",
        )

    try:
        state = get_state()
    except ModelNotLoadedError:
        raise HTTPException(status_code=503, detail="model not loaded")

    # Pre-tokenize for budget check + per-item length validation
    token_counts = [_count_tokens(state.tokenizer, item.text) for item in req.items]
    total_tokens = sum(token_counts)
    if total_tokens > settings.max_batch_total_tokens:
        raise HTTPException(
            status_code=422,
            detail=f"batch total tokens {total_tokens} exceeds "
                   f"MAX_BATCH_TOTAL_TOKENS ({settings.max_batch_total_tokens})",
        )

    started = time.perf_counter()
    results: list[BatchItemResult] = []

    for item, n_tok in zip(req.items, token_counts):
        results.append(await _process_item(item, n_tok, req.mask, settings, state))

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return BatchResponse(
        results=results,
        meta=BatchMeta(
            model=state.model_name,
            batch_size=len(req.items),
            processing_ms=elapsed_ms,
        ),
    )


async def _process_item(item: BatchItem, n_tok: int, mask_top: bool,
                        settings, state) -> BatchItemResult:
    if n_tok > settings.chunk_size_tokens:
        return BatchItemResult(
            status="error",
            error=BatchItemError(
                code="item_too_long",
                message=(f"item has {n_tok} tokens which exceeds "
                         f"CHUNK_SIZE_TOKENS ({settings.chunk_size_tokens}); "
                         "chunking is not allowed in batch — use POST /detect"),
            ),
        )

    try:
        allowed = validate_labels(item.labels)
    except UnknownLabelError as e:
        return BatchItemResult(
            status="error",
            error=BatchItemError(code="unknown_label", message=str(e)),
        )

    try:
        async with model_module._semaphore:
            spans = await asyncio.to_thread(state.run_inference, item.text, item.mode)
    except Exception as exc:  # noqa: BLE001 — surface inference failure per item
        return BatchItemResult(
            status="error",
            error=BatchItemError(code="inference_failed", message=str(exc)),
        )

    if allowed is not None:
        spans = [s for s in spans if s["label"] in allowed]
    entities = [Entity(**s) for s in spans]
    masked_text = _apply_mask(item.text, entities, item.mask_char) if mask_top else None

    return BatchItemResult(
        status="ok",
        entities=entities,
        masked_text=masked_text,
        meta={"entity_count": len(entities)},
    )
```

- [ ] **Step 4: Wire router in `app/main.py`**

Modify the imports and registration to include batch routes:
```python
from app.routes import batch as batch_routes
from app.routes import detect as detect_routes
from app.routes import health as health_routes
```

```python
    app.include_router(health_routes.router)
    app.include_router(detect_routes.router)
    app.include_router(batch_routes.router)
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_batch.py -v
```

Expected: 8 passed.

- [ ] **Step 6: Commit**

```bash
git add app/routes/batch.py app/main.py tests/test_batch.py
git commit -m "feat(api): /detect/batch with hybrid error semantics and per-item validation"
```

---

### Task 14: Rate limiting

**Files:**
- Create: `app/ratelimit.py`
- Modify: `app/main.py`
- Modify: `app/routes/detect.py` (decorate)
- Modify: `app/routes/batch.py` (decorate)
- Create: `tests/test_ratelimit.py`

- [ ] **Step 1: Write the failing test**

`tests/test_ratelimit.py`:
```python
from fastapi.testclient import TestClient
import pytest

from app.main import create_app


@pytest.fixture
def client_limited(patch_model, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
    monkeypatch.setenv("RATE_LIMIT_PER_IP", "3/10minutes")
    return TestClient(create_app(load_at_startup=False))


@pytest.fixture
def client_unlimited(patch_model, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
    return TestClient(create_app(load_at_startup=False))


def test_rate_limit_returns_429_after_limit(client_limited):
    for i in range(3):
        r = client_limited.post("/detect", json={"text": "x"})
        assert r.status_code == 200, f"expected 200 on call {i+1}, got {r.status_code}"
    r = client_limited.post("/detect", json={"text": "x"})
    assert r.status_code == 429


def test_health_exempt_from_rate_limit(client_limited):
    for _ in range(20):
        r = client_limited.get("/health")
        assert r.status_code == 200


def test_rate_limit_disabled_means_unlimited(client_unlimited):
    for _ in range(20):
        r = client_unlimited.post("/detect", json={"text": "x"})
        assert r.status_code == 200
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_ratelimit.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `app/ratelimit.py`**

```python
"""
Per-IP rate limiting via slowapi.

Design notes:
- A single module-level Limiter singleton is shared across all route modules,
  so per-IP counters are unified. Two limiter instances would mean each route
  has its own counter, defeating the per-IP cap.
- The limit string is provided as a CALLABLE so it is re-evaluated per request.
  This lets test fixtures override env vars after the module is imported.
- `RATE_LIMIT_ENABLED=false` is honored by returning a sky-high limit
  ("1000000/second") rather than disabling the limiter — keeps slowapi's
  decorator wiring uniform.
"""
from __future__ import annotations

from fastapi import FastAPI, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.config import get_settings


def _key_func(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return get_remote_address(request)


# Single shared limiter instance for all routes.
limiter: Limiter = Limiter(key_func=_key_func, enabled=True)


def current_limit() -> str:
    """Used as a callable in @limiter.limit so env overrides are honored per request."""
    s = get_settings()
    if not s.rate_limit_enabled:
        return "1000000/second"
    return s.rate_limit_per_ip


def install(app: FastAPI) -> None:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


def reset_for_tests() -> None:
    """Clear in-memory counters between tests."""
    limiter.reset()
```

- [ ] **Step 4: Wire into `app/main.py`**

Replace `app/main.py`:
```python
from __future__ import annotations

from contextlib import asynccontextmanager
from fastapi import FastAPI

from app import model as model_module
from app.ratelimit import install, reset_for_tests
from app.routes import batch as batch_routes
from app.routes import detect as detect_routes
from app.routes import health as health_routes


def create_app(load_at_startup: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if load_at_startup:
            model_module.load_model()
        yield

    app = FastAPI(title="pii-filter", version="1.0.0", lifespan=lifespan)

    install(app)
    reset_for_tests()  # Clean counters when called in test contexts that build a new app

    app.include_router(health_routes.router)
    app.include_router(detect_routes.router)
    app.include_router(batch_routes.router)
    return app


app = create_app(load_at_startup=True)
```

- [ ] **Step 5: Decorate the rate-limited routes (full final code)**

Replace `app/routes/detect.py` with the complete final version (rate limit added, all imports finalized):

```python
from __future__ import annotations

import asyncio
import time
from fastapi import APIRouter, HTTPException, Request

from app import model as model_module
from app.chunker import detect_with_chunking
from app.config import get_settings
from app.labels import UnknownLabelError, validate_labels
from app.model import ModelNotLoadedError, get_state
from app.ratelimit import current_limit, limiter
from app.schemas import (
    DetectMeta, DetectRequest, DetectResponse, Entity, MaskRequest, MaskResponse,
)


router = APIRouter()


def _apply_mask(text: str, entities: list[Entity], mask_char: str) -> str:
    parts = sorted(entities, key=lambda e: e.start, reverse=True)
    out = text
    for e in parts:
        out = out[: e.start] + mask_char + out[e.end :]
    return out


def _filter_by_labels(spans: list[dict], allowed: frozenset[str] | None) -> list[dict]:
    if allowed is None:
        return spans
    return [s for s in spans if s["label"] in allowed]


@router.post("/detect", response_model=DetectResponse)
@limiter.limit(current_limit)
async def detect(request: Request, req: DetectRequest) -> DetectResponse:
    settings = get_settings()
    if len(req.text) > settings.max_text_length:
        raise HTTPException(
            status_code=422,
            detail=f"text exceeds MAX_TEXT_LENGTH ({settings.max_text_length})",
        )
    try:
        allowed = validate_labels(req.labels)
    except UnknownLabelError as e:
        raise HTTPException(status_code=422, detail=str(e))

    try:
        state = get_state()
    except ModelNotLoadedError:
        raise HTTPException(status_code=503, detail="model not loaded")

    started = time.perf_counter()

    def _do():
        return detect_with_chunking(
            text=req.text,
            tokenizer=state.tokenizer,
            run_inference=lambda t, m: state.run_inference(t, m),
            mode=req.mode,
            chunk_size_tokens=settings.chunk_size_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
            smart_split=settings.smart_split,
        )

    async with model_module._semaphore:
        result = await asyncio.to_thread(_do)

    spans = _filter_by_labels(result.entities, allowed)
    entities = [Entity(**s) for s in spans]
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    meta = DetectMeta(
        model=state.model_name,
        mode=req.mode,
        entity_count=len(entities),
        processing_ms=elapsed_ms,
        chunks_processed=result.chunks_processed if result.chunks_processed > 1 else None,
        input_tokens=result.input_tokens if result.chunks_processed > 1 else None,
    )
    masked_text = _apply_mask(req.text, entities, req.mask_char) if req.mask else None

    return DetectResponse(text=req.text, entities=entities,
                          masked_text=masked_text, meta=meta)


@router.post("/mask", response_model=MaskResponse)
@limiter.limit(current_limit)
async def mask(request: Request, req: MaskRequest) -> MaskResponse:
    detect_req = DetectRequest(
        text=req.text, mode=req.mode, mask=True,
        mask_char=req.mask_char, labels=req.labels,
    )
    detect_resp = await detect(request, detect_req)
    return MaskResponse(
        masked_text=detect_resp.masked_text or req.text,
        entity_count=detect_resp.meta.entity_count,
        processing_ms=detect_resp.meta.processing_ms,
    )
```

Replace `app/routes/batch.py` with the complete final version (rate limit added, `Request` parameter wired):

```python
from __future__ import annotations

import asyncio
import time
from fastapi import APIRouter, HTTPException, Request

from app import model as model_module
from app.config import get_settings
from app.labels import UnknownLabelError, validate_labels
from app.model import ModelNotLoadedError, get_state
from app.ratelimit import current_limit, limiter
from app.schemas import (
    BatchItem, BatchItemError, BatchItemResult, BatchMeta, BatchRequest,
    BatchResponse, Entity,
)


router = APIRouter()


def _apply_mask(text: str, entities: list[Entity], mask_char: str) -> str:
    parts = sorted(entities, key=lambda e: e.start, reverse=True)
    out = text
    for e in parts:
        out = out[: e.start] + mask_char + out[e.end :]
    return out


def _count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


@router.post("/detect/batch", response_model=BatchResponse)
@limiter.limit(current_limit)
async def detect_batch(request: Request, req: BatchRequest) -> BatchResponse:
    settings = get_settings()

    if len(req.items) > settings.max_batch_size:
        raise HTTPException(
            status_code=422,
            detail=f"batch size {len(req.items)} exceeds MAX_BATCH_SIZE "
                   f"({settings.max_batch_size})",
        )

    try:
        state = get_state()
    except ModelNotLoadedError:
        raise HTTPException(status_code=503, detail="model not loaded")

    token_counts = [_count_tokens(state.tokenizer, item.text) for item in req.items]
    total_tokens = sum(token_counts)
    if total_tokens > settings.max_batch_total_tokens:
        raise HTTPException(
            status_code=422,
            detail=f"batch total tokens {total_tokens} exceeds "
                   f"MAX_BATCH_TOTAL_TOKENS ({settings.max_batch_total_tokens})",
        )

    started = time.perf_counter()
    results: list[BatchItemResult] = []
    for item, n_tok in zip(req.items, token_counts):
        results.append(await _process_item(item, n_tok, req.mask, settings, state))

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return BatchResponse(
        results=results,
        meta=BatchMeta(model=state.model_name, batch_size=len(req.items),
                       processing_ms=elapsed_ms),
    )


async def _process_item(item: BatchItem, n_tok: int, mask_top: bool,
                        settings, state) -> BatchItemResult:
    if n_tok > settings.chunk_size_tokens:
        return BatchItemResult(
            status="error",
            error=BatchItemError(
                code="item_too_long",
                message=(f"item has {n_tok} tokens which exceeds "
                         f"CHUNK_SIZE_TOKENS ({settings.chunk_size_tokens}); "
                         "chunking is not allowed in batch — use POST /detect"),
            ),
        )

    try:
        allowed = validate_labels(item.labels)
    except UnknownLabelError as e:
        return BatchItemResult(
            status="error",
            error=BatchItemError(code="unknown_label", message=str(e)),
        )

    try:
        async with model_module._semaphore:
            spans = await asyncio.to_thread(state.run_inference, item.text, item.mode)
    except Exception as exc:  # noqa: BLE001 — surface inference failure per item
        return BatchItemResult(
            status="error",
            error=BatchItemError(code="inference_failed", message=str(exc)),
        )

    if allowed is not None:
        spans = [s for s in spans if s["label"] in allowed]
    entities = [Entity(**s) for s in spans]
    masked_text = _apply_mask(item.text, entities, item.mask_char) if mask_top else None

    return BatchItemResult(
        status="ok", entities=entities, masked_text=masked_text,
        meta={"entity_count": len(entities)},
    )
```

> **slowapi requirements:**
> 1. Every rate-limited handler MUST have `request: Request` as the first parameter — slowapi inspects the signature.
> 2. The `current_limit` callable is re-evaluated per request, so test fixtures that change env vars take effect immediately.
> 3. Internal `await detect(...)` call sites (only in `mask`) must pass the request through.

- [ ] **Step 6: Run all tests**

```bash
pytest tests/ -v
```

Expected: every prior test still passes (200s on /detect/mask/batch are unaffected by the limit when below threshold), plus the 3 new rate-limit tests pass.

- [ ] **Step 7: Commit**

```bash
git add app/ratelimit.py app/main.py app/routes/ tests/test_ratelimit.py
git commit -m "feat(api): per-IP rate limiting via slowapi (env-tunable, default 60/10min)"
```

---

## Phase 5 — Containerization

### Task 15: Dockerfile

**Files:**
- Create: `Dockerfile`

- [ ] **Step 1: Write `Dockerfile`**

```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Non-root user
RUN useradd --system --create-home --uid 10001 piifilter
USER piifilter

COPY --chown=piifilter:piifilter app/ ./app/

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

- [ ] **Step 2: Build to confirm it works**

```bash
docker build -t pii-filter:dev .
```

Expected: build succeeds.

- [ ] **Step 3: Commit**

```bash
git add Dockerfile
git commit -m "chore(docker): non-root Python 3.11-slim image"
```

---

### Task 16: docker-compose.yml + .env.example

**Files:**
- Create: `docker-compose.yml`
- Create: `.env.example`

- [ ] **Step 1: Write `docker-compose.yml`**

```yaml
services:
  pii-filter:
    build: .
    image: pii-filter:latest
    container_name: pii-filter
    ports:
      - "${PORT:-8080}:8080"
    environment:
      - HF_HOME=/app/model_cache
      - DEVICE=${DEVICE:-cpu}
      - DEFAULT_MODE=${DEFAULT_MODE:-balanced}
      - LOG_LEVEL=${LOG_LEVEL:-info}
      - MAX_TEXT_LENGTH=${MAX_TEXT_LENGTH:-524288}
      - CHUNK_SIZE_TOKENS=${CHUNK_SIZE_TOKENS:-120000}
      - CHUNK_OVERLAP_TOKENS=${CHUNK_OVERLAP_TOKENS:-512}
      - SMART_SPLIT=${SMART_SPLIT:-true}
      - MAX_CONCURRENT_INFERENCES=${MAX_CONCURRENT_INFERENCES:-2}
      - MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-32}
      - MAX_BATCH_TOTAL_TOKENS=${MAX_BATCH_TOTAL_TOKENS:-200000}
      - RATE_LIMIT_ENABLED=${RATE_LIMIT_ENABLED:-true}
      - RATE_LIMIT_PER_IP=${RATE_LIMIT_PER_IP:-60/10minutes}
    volumes:
      - pii_model_cache:/app/model_cache
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 600s
    restart: unless-stopped

volumes:
  pii_model_cache:
    driver: local
```

- [ ] **Step 2: Write `.env.example`**

```env
PORT=8080
DEVICE=cpu
DEFAULT_MODE=balanced
LOG_LEVEL=info
MAX_TEXT_LENGTH=524288
CHUNK_SIZE_TOKENS=120000
CHUNK_OVERLAP_TOKENS=512
SMART_SPLIT=true
MAX_CONCURRENT_INFERENCES=2
MAX_BATCH_SIZE=32
MAX_BATCH_TOTAL_TOKENS=200000
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PER_IP=60/10minutes
```

- [ ] **Step 3: Validate the compose file**

```bash
docker compose config
```

Expected: valid YAML, no warnings about unknown keys.

- [ ] **Step 4: Commit**

```bash
git add docker-compose.yml .env.example
git commit -m "chore(docker): compose stack with healthcheck (start_period=600s)"
```

---

### Task 17: docker-compose.gpu.yml

**Files:**
- Create: `docker-compose.gpu.yml`

- [ ] **Step 1: Write the override**

```yaml
# Use with: docker compose -f docker-compose.yml -f docker-compose.gpu.yml up
services:
  pii-filter:
    image: nvidia/cuda:12.4.1-runtime-ubuntu22.04
    environment:
      - DEVICE=cuda
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

- [ ] **Step 2: Validate the override**

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml config
```

Expected: merges cleanly. (Won't actually launch unless host has NVIDIA Container Toolkit; that's expected.)

- [ ] **Step 3: Commit**

```bash
git add docker-compose.gpu.yml
git commit -m "chore(docker): GPU compose override (nvidia/cuda runtime)"
```

---

## Phase 6 — MCP Sidecar

### Task 18: MCP package skeleton

**Files:**
- Create: `mcp_server/pyproject.toml`
- Create: `mcp_server/pii_filter_mcp/__init__.py` (empty)
- Create: `mcp_server/pii_filter_mcp/__main__.py`
- Create: `mcp_server/pii_filter_mcp/server.py`
- Create: `mcp_server/tests/__init__.py` (empty)

- [ ] **Step 1: Write `mcp_server/pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "pii-filter-mcp"
version = "1.0.0"
description = "MCP server exposing pii-filter as Claude Desktop tools"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "mcp>=1.0.0",
    "httpx>=0.27.0",
]

[project.scripts]
pii-filter-mcp = "pii_filter_mcp.__main__:main"

[tool.setuptools.packages.find]
include = ["pii_filter_mcp*"]
```

- [ ] **Step 2: Write `mcp_server/pii_filter_mcp/server.py`**

```python
"""MCP server exposing pii-filter REST endpoints as tools."""
from __future__ import annotations

import os
from typing import Optional

import httpx
from mcp.server import Server
from mcp.types import TextContent, Tool


PII_FILTER_URL = os.environ.get("PII_FILTER_URL", "http://localhost:8080")
TIMEOUT_SEC = float(os.environ.get("PII_FILTER_TIMEOUT_SEC", "60"))


def build_server() -> Server:
    server = Server("pii-filter")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="detect_pii",
                description=(
                    "Detect PII spans in the given text. Returns structured "
                    "entities with label, character offsets, surface text, and score."
                ),
                inputSchema={
                    "type": "object",
                    "required": ["text"],
                    "properties": {
                        "text": {"type": "string"},
                        "mode": {
                            "type": "string",
                            "enum": ["precise", "balanced", "recall"],
                        },
                        "labels": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
            ),
            Tool(
                name="mask_pii",
                description=(
                    "Mask PII in the given text and return the redacted string."
                ),
                inputSchema={
                    "type": "object",
                    "required": ["text"],
                    "properties": {
                        "text": {"type": "string"},
                        "mode": {
                            "type": "string",
                            "enum": ["precise", "balanced", "recall"],
                        },
                        "labels": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "mask_char": {"type": "string"},
                    },
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        if name == "detect_pii":
            return await _detect(arguments)
        if name == "mask_pii":
            return await _mask(arguments)
        raise ValueError(f"unknown tool: {name}")

    return server


async def _detect(args: dict) -> list[TextContent]:
    payload = {"text": args["text"]}
    if "mode" in args:
        payload["mode"] = args["mode"]
    if "labels" in args:
        payload["labels"] = args["labels"]

    async with httpx.AsyncClient(timeout=TIMEOUT_SEC) as client:
        r = await client.post(f"{PII_FILTER_URL}/detect", json=payload)
        r.raise_for_status()
        body = r.json()
    import json
    return [TextContent(
        type="text",
        text=json.dumps({
            "entities": body["entities"],
            "entity_count": body["meta"]["entity_count"],
        }),
    )]


async def _mask(args: dict) -> list[TextContent]:
    payload = {"text": args["text"]}
    for k in ("mode", "labels", "mask_char"):
        if k in args:
            payload[k] = args[k]

    async with httpx.AsyncClient(timeout=TIMEOUT_SEC) as client:
        r = await client.post(f"{PII_FILTER_URL}/mask", json=payload)
        r.raise_for_status()
        body = r.json()
    import json
    return [TextContent(
        type="text",
        text=json.dumps({
            "masked_text": body["masked_text"],
            "entity_count": body["entity_count"],
        }),
    )]
```

- [ ] **Step 3: Write `mcp_server/pii_filter_mcp/__main__.py`**

```python
"""Entrypoint: `python -m pii_filter_mcp` runs the stdio MCP server."""
from __future__ import annotations

import asyncio

from mcp.server.stdio import stdio_server

from pii_filter_mcp.server import build_server


async def _run() -> None:
    server = build_server()
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Smoke install**

```bash
pip install -e mcp_server/
```

Expected: installs without error.

- [ ] **Step 5: Commit**

```bash
git add mcp_server/
git commit -m "feat(mcp): stdio MCP sidecar exposing detect_pii and mask_pii tools"
```

---

### Task 19: MCP server tests

**Files:**
- Create: `mcp_server/tests/test_tools.py`

- [ ] **Step 1: Write the failing test**

`mcp_server/tests/test_tools.py`:
```python
"""
Test the MCP tool layer in isolation by stubbing httpx.

Run from repo root:  pytest mcp_server/tests/ -v
"""
from __future__ import annotations

import json
import pytest
import httpx

from pii_filter_mcp import server as srv


class _FakeClient:
    def __init__(self, response_body: dict, *args, **kwargs):
        self._body = response_body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def post(self, url, json):
        self.last_url = url
        self.last_json = json
        return _FakeResp(self._body)


class _FakeResp:
    def __init__(self, body): self._body = body
    def raise_for_status(self): pass
    def json(self): return self._body


@pytest.fixture
def fake_detect_response():
    return {
        "text": "alice@example.com",
        "entities": [{
            "label": "private_email", "start": 0, "end": 17,
            "text": "alice@example.com", "score": 0.99,
        }],
        "meta": {"model": "x", "mode": "balanced",
                 "entity_count": 1, "processing_ms": 10},
    }


@pytest.fixture
def fake_mask_response():
    return {"masked_text": "[REDACTED]", "entity_count": 1, "processing_ms": 10}


async def test_detect_calls_correct_endpoint(monkeypatch, fake_detect_response):
    fakes: list[_FakeClient] = []
    def factory(*a, **kw):
        c = _FakeClient(fake_detect_response, *a, **kw)
        fakes.append(c)
        return c
    monkeypatch.setattr(httpx, "AsyncClient", factory)

    out = await srv._detect({"text": "alice@example.com"})

    assert fakes[0].last_url.endswith("/detect")
    assert fakes[0].last_json == {"text": "alice@example.com"}
    body = json.loads(out[0].text)
    assert body["entity_count"] == 1
    assert body["entities"][0]["label"] == "private_email"


async def test_mask_passes_optional_args(monkeypatch, fake_mask_response):
    captured: list[_FakeClient] = []
    def factory(*a, **kw):
        c = _FakeClient(fake_mask_response, *a, **kw)
        captured.append(c)
        return c
    monkeypatch.setattr(httpx, "AsyncClient", factory)

    out = await srv._mask({
        "text": "alice@example.com",
        "mode": "recall",
        "labels": ["private_email"],
        "mask_char": "***",
    })
    assert captured[0].last_url.endswith("/mask")
    assert captured[0].last_json == {
        "text": "alice@example.com",
        "mode": "recall",
        "labels": ["private_email"],
        "mask_char": "***",
    }
    body = json.loads(out[0].text)
    assert body["masked_text"] == "[REDACTED]"
```

- [ ] **Step 2: Run**

```bash
pytest mcp_server/tests/ -v
```

Expected: 2 passed.

- [ ] **Step 3: Commit**

```bash
git add mcp_server/tests/
git commit -m "test(mcp): isolated tool tests with stubbed httpx"
```

---

## Phase 7 — Documentation & Final Verification

### Task 20: Main README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

```markdown
# pii-filter

Self-hosted PII detection and masking REST API powered by `openai/privacy-filter`.

- Zero-config startup: `docker compose up`
- Auto-chunking for inputs of any length (token-aligned, no offset drift)
- Per-IP rate limiting (configurable)
- Optional MCP sidecar for Claude Desktop integration

## Quickstart

```bash
cp .env.example .env
docker compose up -d
# First boot downloads ~3GB; allow up to 5 min
curl -s http://localhost:8080/ready
# Once ready:
curl -X POST http://localhost:8080/detect \
  -H 'Content-Type: application/json' \
  -d '{"text": "Email me at alice@example.com"}'
```

## Endpoints

- `POST /detect` — return PII spans (and optionally masked text).
- `POST /mask` — convenience: return only the masked text.
- `POST /detect/batch` — batch up to `MAX_BATCH_SIZE` short items.
- `GET /health` — always 200 if process is alive; reports `model_loaded`.
- `GET /ready` — 200 once model is loaded; 503 otherwise.
- `GET /docs` — interactive OpenAPI UI.

### Span offsets

Spans use **Python code-point character indices** into the request text:
`text[entity.start:entity.end]` returns the surface form. Clients in Java/Swift/Rust should convert from their default string indexing.

## Configuration

All knobs are env vars (with defaults shown):

| Var | Default | Notes |
|-----|---------|-------|
| `DEVICE` | `cpu` | Set to `cuda` to use GPU |
| `DEFAULT_MODE` | `balanced` | `precise` / `balanced` / `recall` |
| `MAX_TEXT_LENGTH` | `524288` | Hard char cap on `/detect` |
| `CHUNK_SIZE_TOKENS` | `120000` | Max tokens per chunk |
| `CHUNK_OVERLAP_TOKENS` | `512` | Overlap between consecutive chunks |
| `SMART_SPLIT` | `true` | Nudge splits to paragraph boundaries |
| `MAX_CONCURRENT_INFERENCES` | `2` | In-flight inference cap (memory safety) |
| `MAX_BATCH_SIZE` | `32` | Items per `/detect/batch` call |
| `MAX_BATCH_TOTAL_TOKENS` | `200000` | Token budget per batch |
| `RATE_LIMIT_ENABLED` | `true` | Master switch |
| `RATE_LIMIT_PER_IP` | `60/10minutes` | slowapi format |

## Throughput tuning

Throughput is bounded by `MAX_CONCURRENT_INFERENCES`. The model singleton avoids 4GB-per-worker memory bloat; horizontal scaling is via more containers, not more workers per container.

For ≥10 req/s on short inputs (≤2k tokens), tune `MAX_CONCURRENT_INFERENCES` based on your CPU core count (typical good value: cores − 1).

## GPU

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up
```

Requires NVIDIA Container Toolkit on the host. The override pins `nvidia/cuda:12.4.1`; adjust the tag if your driver requires a different CUDA version.

## Security

- No data persisted. No external calls after first model download.
- Container runs as a non-root user (uid 10001).
- Auth is **not** built in; deployers should restrict network access (reverse proxy, NetworkPolicy, VPN). The per-IP rate limit is a brake against runaway clients, not an access-control mechanism.
- `X-Forwarded-For` is taken at face value — terminate untrusted proxies before reaching this service.

## MCP server

See `mcp_server/README.md`.

## Development

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## License

Apache 2.0 (matches the model license).
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README with quickstart, env reference, and tuning notes"
```

---

### Task 21: MCP README

**Files:**
- Create: `mcp_server/README.md`

- [ ] **Step 1: Write `mcp_server/README.md`**

```markdown
# pii-filter-mcp

stdio MCP server that exposes the pii-filter REST API as Claude Desktop tools.

## Install

```bash
pip install -e ./mcp_server
```

## Configure Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS):

```json
{
  "mcpServers": {
    "pii-filter": {
      "command": "python",
      "args": ["-m", "pii_filter_mcp"],
      "env": { "PII_FILTER_URL": "http://localhost:8080" }
    }
  }
}
```

Then start the pii-filter container (`docker compose up -d` in the repo root) and restart Claude Desktop.

## Tools exposed

- `detect_pii(text, mode?, labels?)` — returns `{ entities, entity_count }`.
- `mask_pii(text, mode?, labels?, mask_char?)` — returns `{ masked_text, entity_count }`.

## Configuration

| Var | Default |
|-----|---------|
| `PII_FILTER_URL` | `http://localhost:8080` |
| `PII_FILTER_TIMEOUT_SEC` | `60` |

## Development

```bash
pytest mcp_server/tests/
```
```

- [ ] **Step 2: Commit**

```bash
git add mcp_server/README.md
git commit -m "docs(mcp): install + Claude Desktop config snippet"
```

---

### Task 22: Final integration check

**Files:** none (verification step)

- [ ] **Step 1: Run the full test suite**

```bash
pytest -v
```

Expected: all tests pass. Capture the count.

- [ ] **Step 2: Build the Docker image**

```bash
docker build -t pii-filter:latest .
```

Expected: success.

- [ ] **Step 3: Validate compose configurations**

```bash
docker compose config > /dev/null
docker compose -f docker-compose.yml -f docker-compose.gpu.yml config > /dev/null
```

Expected: both succeed silently.

- [ ] **Step 4: Tag the release**

```bash
git tag -a v1.0.0 -m "v1.0.0: PII filter service + MCP sidecar"
```

- [ ] **Step 5: Print the summary**

Report to user:
- Test count (pass/fail)
- Image size: `docker images pii-filter:latest --format '{{.Size}}'`
- File count added: `git diff --stat <first-commit>..HEAD | tail -1`

---

## Spec Coverage Map

| Spec section | Implemented in task |
|--------------|---------------------|
| §1 Decision summary | (cross-cutting) |
| §2 Auth & rate limiting | Task 14 |
| §3 Batch endpoint | Task 13 |
| §4 Label filtering | Task 3, 11, 12, 13 |
| §5 Offset semantics | Task 4 (Pydantic), §11 (chunker) — documented in Task 20 README |
| §6 Chunker correctness | Task 7, 8, 9 |
| §7 Concurrency | Task 6 |
| §8 Health & readiness | Task 10 |
| §9 GPU support | Task 17 |
| §10 MCP sidecar | Task 18, 19, 21 |
| §11 Env var reference | Task 2 (config), Task 16 (compose), Task 20 (README) |
| §12 Project structure | Task 1 (scaffolding), all subsequent module tasks |
| §13 Smoke tests | Tests across Tasks 7-14 + 19 |
| §14 Resolved PRD open questions | (informational) |
| §15 New deps | Task 1 (`requirements.txt`), Task 18 (mcp `pyproject.toml`) |
| §16 Out of scope | (no implementation needed) |

---

## Notes for the Executor

- **Don't refactor while implementing.** If you spot something you think is wrong, raise it as a question — don't quietly redesign.
- **Tests are non-negotiable per task.** Each task ends with all its tests green.
- **Commits are per task, not per step.** The step-level checkpoints exist so you can pause and resume cleanly; the commit lands at the end.
- **The model is mocked everywhere except the integration check.** Don't try to download `openai/privacy-filter` during unit tests.
- **slowapi quirk:** every rate-limited handler MUST take `request: Request` as a route parameter, even if unused. The decorator inspects the signature.
