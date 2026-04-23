# PRD — PII Filter Service
**`pii-filter`** · Docker Compose · Ready-to-use REST API for PII detection & masking

---

## 1. Overview

### Problem Statement
Teams building data pipelines, LLM applications, or compliance workflows need a fast, self-hosted way to detect and optionally mask Personally Identifiable Information (PII) in text — without sending sensitive data to third-party APIs.

### Solution
A turnkey Docker Compose stack exposing a REST endpoint that accepts raw text and returns detected PII spans (with optional masked output), powered by OpenAI's `openai/privacy-filter` model running on-premises.

### Goals
- Zero-config startup: `docker compose up` → working endpoint
- Self-contained: model downloaded and cached at first boot
- Production-ready: health check, structured logging, graceful error handling
- Tunable: precision/recall tradeoff configurable via request params or env vars
- Developer-friendly: OpenAPI docs at `/docs`

---

## 2. Architecture

```
┌────────────────────────────────────────┐
│              Docker Compose            │
│                                        │
│  ┌─────────────┐    ┌───────────────┐  │
│  │   API       │    │  Model Worker │  │
│  │  (FastAPI)  │───▶│  (Transformers│  │
│  │  :8080      │    │   pipeline)   │  │
│  └─────────────┘    └───────────────┘  │
│         │                              │
│  ┌──────▼──────┐                       │
│  │ HF Cache    │ (volume: model files) │
│  │  Volume     │                       │
│  └─────────────┘                       │
└────────────────────────────────────────┘
```

### Services

| Service | Image | Role |
|---------|-------|------|
| `api` | `python:3.11-slim` | FastAPI HTTP server, request validation, response formatting |
| *(single-service option)* | same | API + model in one container for simpler deployments |

> **Note:** For v1, API and model inference run in the same container to keep the compose file minimal. A two-service split (API + worker with a queue) is planned for v2.

### Model
- **`openai/privacy-filter`** from Hugging Face
- 1.5B params total, ~50M active (sparse MoE)
- Runs comfortably on CPU; GPU optional via env var
- 128k token context window
- Apache 2.0 license

---

## 3. Detected PII Categories

The model outputs spans for 8 categories:

| Label | Examples |
|-------|---------|
| `private_person` | Names, usernames |
| `private_email` | alice@example.com |
| `private_phone` | +1-555-123-4567 |
| `private_address` | 123 Main St, Paris 75001 |
| `account_number` | IBAN, credit card numbers |
| `private_url` | URLs linked to individuals |
| `private_date` | Birthdates, appointment dates |
| `secret` | API keys, passwords, tokens |

---

## 4. Long-Text Chunking

### Problem
The model's context window is hard-capped at **128,000 tokens**. Naive truncation would silently drop PII in long documents. Naive splitting at a fixed character boundary risks cutting a PII span in half (e.g. slicing `"Alice Sm" | "ith"`), causing the model to miss it entirely.

### Strategy: Overlapping Token-Aligned Chunks

```
Original text (very long):
├─────────────────────────────────────────────────────────────┤

Chunk 1 (120k tokens):
├──────────────────────────────────────┤
                             ├── overlap (512 tok) ──┤

Chunk 2 (120k tokens):
                    ├──────────────────────────────────────────┤
                             ├── overlap (512 tok) ──┤

Chunk 3 ...
```

**Key design decisions:**

1. **Token-aligned splits** — chunking is done in token space (via the model's own tokenizer), not character space. Each chunk is decoded back to text, preserving correct token boundaries.

2. **120k token chunks** — 8k tokens of headroom below the 128k limit, reserved for any tokenizer edge cases and to keep memory pressure low.

3. **512-token overlap** — consecutive chunks share a 512-token window at their boundary. Any PII span that straddles the split point will be fully visible in at least one chunk.

4. **Character offset tracking** — each chunk records its `char_start` offset in the original text. All span `start`/`end` values returned by the model (relative to the chunk) are shifted by `char_start` before merging.

5. **Overlap deduplication** — after merging all chunks' entities, spans that:
   - have overlapping character ranges **and**
   - share the same label
   …are collapsed into a single entity, keeping the one with the higher confidence score.

### Algorithm (pseudocode)

```python
def detect_long_text(text, tokenizer, model, chunk_size=120_000, overlap=512):
    tokens = tokenizer.encode(text)

    if len(tokens) <= chunk_size:
        return run_inference(text)          # Fast path: fits in one pass

    chunks = []
    step = chunk_size - overlap
    for start_tok in range(0, len(tokens), step):
        end_tok = min(start_tok + chunk_size, len(tokens))
        chunk_tokens = tokens[start_tok:end_tok]
        chunk_text = tokenizer.decode(chunk_tokens)

        # Map token offset → character offset in original text
        char_start = len(tokenizer.decode(tokens[:start_tok]))
        chunks.append((chunk_text, char_start))

    all_entities = []
    for chunk_text, char_start in chunks:
        entities = run_inference(chunk_text)
        for e in entities:
            all_entities.append({**e, "start": e["start"] + char_start,
                                       "end":   e["end"]   + char_start})

    return deduplicate_spans(all_entities)


def deduplicate_spans(entities):
    # Sort by start offset, then by descending score
    entities.sort(key=lambda e: (e["start"], -e["score"]))
    seen, result = [], []
    for e in entities:
        overlaps = any(
            e["label"] == s["label"] and e["start"] < s["end"] and e["end"] > s["start"]
            for s in seen
        )
        if not overlaps:
            result.append(e)
            seen.append(e)
    return result
```

### Split Boundary Heuristic (optional enhancement)

When `SMART_SPLIT=true` (env var), the chunker additionally nudges the split point to the nearest **paragraph boundary** (`\n\n`) within ±200 tokens of the calculated split. This reduces mid-sentence splits and improves span coherence at boundaries with negligible overhead.

### Configuration

| Env var | Default | Description |
|---------|---------|-------------|
| `CHUNK_SIZE_TOKENS` | `120000` | Max tokens per chunk (must be ≤ 128000) |
| `CHUNK_OVERLAP_TOKENS` | `512` | Overlap window between consecutive chunks |
| `SMART_SPLIT` | `true` | Nudge split to nearest paragraph boundary |

### Response changes for chunked inputs

The `meta` block in `/detect` responses gains two additional fields when chunking is triggered:

```json
"meta": {
  "model": "openai/privacy-filter",
  "mode": "balanced",
  "entity_count": 14,
  "processing_ms": 1840,
  "chunks_processed": 3,       // ← added when input > CHUNK_SIZE_TOKENS
  "input_tokens": 287432       // ← added when input > CHUNK_SIZE_TOKENS
}
```

When `chunks_processed` is absent (or `1`), the input fit in a single pass.

---

## 5. API Specification

### Base URL
```
http://localhost:8080
```

### Endpoints

---

#### `POST /detect`
Detect PII spans in text. Returns structured span data.

**Request body**
```json
{
  "text": "My name is Alice Smith and my email is alice@example.com",
  "mode": "balanced",
  "mask": false
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | `string` | required | Input text (max 512k chars) |
| `mode` | `enum` | `"balanced"` | Operating point: `"precise"`, `"balanced"`, `"recall"` |
| `mask` | `boolean` | `false` | If true, include masked text in response |
| `mask_char` | `string` | `"[REDACTED]"` | Replacement string when `mask=true` |

**Response `200 OK`**
```json
{
  "text": "My name is Alice Smith and my email is alice@example.com",
  "entities": [
    {
      "label": "private_person",
      "start": 11,
      "end": 22,
      "text": "Alice Smith",
      "score": 0.97
    },
    {
      "label": "private_email",
      "start": 38,
      "end": 55,
      "text": "alice@example.com",
      "score": 0.99
    }
  ],
  "masked_text": "My name is [REDACTED] and my email is [REDACTED]",
  "meta": {
    "model": "openai/privacy-filter",
    "mode": "balanced",
    "entity_count": 2,
    "processing_ms": 142
  }
}
```

---

#### `POST /mask`
Convenience shorthand — same as `POST /detect` with `mask=true`. Returns only the masked text.

**Request body**
```json
{
  "text": "Call me at +33 6 12 34 56 78",
  "mode": "balanced"
}
```

**Response `200 OK`**
```json
{
  "masked_text": "Call me at [REDACTED]",
  "entity_count": 1,
  "processing_ms": 89
}
```

---

#### `GET /health`
Liveness + readiness check.

**Response `200 OK`**
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cpu"
}
```

---

#### `GET /docs`
Interactive OpenAPI/Swagger UI (auto-generated by FastAPI).

---

### Operating Modes

| Mode | Behavior | Use case |
|------|----------|----------|
| `precise` | Higher precision, may miss some PII | When false positives are costly (e.g. legal doc review) |
| `balanced` | Default tradeoff | General purpose |
| `recall` | Higher recall, may over-redact | When missing PII is costly (e.g. GDPR export) |

---

## 6. Docker Compose Spec

### `docker-compose.yml`

```yaml
version: "3.9"

services:
  pii-filter:
    build: .
    image: pii-filter:latest
    container_name: pii-filter
    ports:
      - "${PORT:-8080}:8080"
    environment:
      - HF_HOME=/app/model_cache
      - DEVICE=${DEVICE:-cpu}                       # set to "cuda" for GPU
      - DEFAULT_MODE=${DEFAULT_MODE:-balanced}
      - LOG_LEVEL=${LOG_LEVEL:-info}
      - MAX_TEXT_LENGTH=${MAX_TEXT_LENGTH:-524288}
      - CHUNK_SIZE_TOKENS=${CHUNK_SIZE_TOKENS:-120000}
      - CHUNK_OVERLAP_TOKENS=${CHUNK_OVERLAP_TOKENS:-512}
      - SMART_SPLIT=${SMART_SPLIT:-true}
    volumes:
      - pii_model_cache:/app/model_cache
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 120s    # Allow time for model download on first boot
    restart: unless-stopped

volumes:
  pii_model_cache:
    driver: local
```

### `.env` (example)
```env
PORT=8080
DEVICE=cpu              # or cuda
DEFAULT_MODE=balanced
LOG_LEVEL=info
MAX_TEXT_LENGTH=524288
CHUNK_SIZE_TOKENS=120000
CHUNK_OVERLAP_TOKENS=512
SMART_SPLIT=true
```

---

## 7. Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY app/ ./app/

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

### `requirements.txt`
```
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
transformers>=4.51.0
torch>=2.2.0
pydantic>=2.0.0
```

---

## 8. Project Structure

```
pii-filter/
├── docker-compose.yml
├── Dockerfile
├── .env.example
├── requirements.txt
├── README.md
└── app/
    ├── main.py          # FastAPI app, routes
    ├── model.py         # Model loading, single-pass inference
    ├── chunker.py       # Long-text splitting, offset mapping, deduplication
    ├── schemas.py       # Pydantic request/response models
    └── config.py        # Settings from env vars
```

---

## 9. Model Loading Strategy

- Model is loaded **once at startup** into a module-level singleton.
- `GET /health` returns `model_loaded: false` until ready (useful for k8s readiness probes).
- First boot downloads ~3GB model files to the named volume `pii_model_cache`.
- Subsequent boots load from cache (seconds, not minutes).
- GPU support: if `DEVICE=cuda`, model uses `device_map="auto"`.

---

## 10. Non-Functional Requirements

| Concern | Requirement |
|---------|------------|
| **Latency (short)** | < 500ms p95 for inputs up to 2,000 tokens on modern CPU |
| **Latency (chunked)** | Linear scale: ~500ms × number of chunks; no artificial timeout |
| **Throughput** | ≥ 10 req/s on 4-core CPU with batching |
| **Startup time** | < 30s (warm cache), < 5min (cold, model download) |
| **Memory** | ~4GB RAM (CPU, FP32) · ~2GB VRAM (GPU, BF16) |
| **Max input** | Unlimited via auto-chunking; configurable hard cap via `MAX_TEXT_LENGTH` |
| **Chunk correctness** | All PII spans fully within a chunk boundary must be detected; spans straddling a boundary must be detected in the overlap window |
| **Security** | No data persisted; no external calls after model download; runs as non-root user |
| **Logging** | Structured JSON logs; input text NOT logged by default |

---

## 11. Error Handling

| HTTP Code | Scenario |
|-----------|----------|
| `200` | Success |
| `422` | Validation error (missing field, text too long) |
| `503` | Model not yet loaded (startup) |
| `500` | Inference error |

---

## 12. Out of Scope (v1)

- Authentication / API key management
- Async job queue for very large documents
- Fine-tuning interface
- Batch endpoint (`POST /detect/batch`)
- Streaming output
- Kubernetes Helm chart
- Metrics / Prometheus exporter

These are candidates for v2.

---

## 13. Milestones

| Milestone | Deliverable |
|-----------|-------------|
| **M1** | `app/model.py` — model loads, basic single-pass inference works |
| **M2** | `app/chunker.py` — token-aligned splitting, offset mapping, span deduplication |
| **M3** | `app/main.py` — `/detect`, `/mask`, `/health` endpoints (chunking transparent to caller) |
| **M4** | Dockerfile builds, container runs end-to-end |
| **M5** | `docker-compose.yml` with volume, healthcheck, env vars |
| **M6** | README with quickstart, curl examples, env var reference |
| **M7** | Smoke tests (pytest + httpx): single-pass inputs, multi-chunk inputs, boundary-spanning PII |

---

## 14. Open Questions

1. **Batching:** Should v1 support a `POST /detect/batch` with an array of texts for throughput optimization?
2. **Auth:** Is an optional `API_KEY` env var needed even for v1?
3. **Label filtering:** Should callers be able to request only specific PII categories (e.g. `labels: ["private_email", "private_phone"]`)?
4. **GPU base image:** Should we ship a `docker-compose.gpu.yml` override using `nvidia/cuda` base image?
5. **Output format:** Should entity offsets be byte-based or character-based? (Relevant for multilingual text.)
6. **Chunk overlap tuning:** 512 tokens is a conservative default. Should the overlap scale with `CHUNK_SIZE_TOKENS`, or remain fixed? Longest realistic PII span (e.g. a full address) is rarely more than 50 tokens, so 512 is likely already generous.
7. **Streaming progress:** For very large documents (10M+ chars = many chunks), should the API support SSE streaming to report `chunk N of M` progress events while results accumulate?
