# PII Filter — Design (v1)

**Date:** 2026-04-23
**Baseline PRD:** [`docs/PRD-pii-filter-service.md`](../../PRD-pii-filter-service.md)
**Status:** Approved for implementation planning

This document captures the design decisions resolved during brainstorming. It is a **delta on the baseline PRD** — anything not contradicted here remains as specified in the PRD. Sections below override or extend the PRD where they overlap.

---

## 1. Decision Summary

| Area | Decision |
|------|----------|
| Auth | None |
| Rate limiting | Per-IP, default `60/10min`, env-tunable, can be disabled |
| Batch endpoint | Yes, `POST /detect/batch`, no chunking allowed per item |
| Label filtering | Optional `labels?: string[]` on `/detect`, `/mask`, batch items |
| Span offsets | Python code-point characters |
| Chunk overlap | Fixed at 512 tokens |
| SSE / streaming | No |
| GPU support | `docker-compose.gpu.yml` override (additive) |
| Concurrency | Single model + bounded semaphore, inference via threadpool |
| Chunker correctness | `tokenizer(..., return_offsets_mapping=True)` (no decode round-trip) |
| Health | `start_period: 600s`; new `/ready` endpoint; `/health` unchanged |
| MCP server | Yes, sidecar package in same repo |

---

## 2. Auth & Rate Limiting

### Auth
v1 ships with no authentication. The service is intended to sit behind whatever access control the deployer already operates (reverse proxy, k8s NetworkPolicy, VPN, internal-only listener). This preserves the "docker compose up and it works" promise.

### Rate limiting
A per-IP rate limit is enforced via [`slowapi`](https://github.com/laurentS/slowapi) (FastAPI-native, in-memory by default).

- **Default:** `60 requests / 10 minutes` per source IP.
- **Scope:** identified by `X-Forwarded-For` (first hop) when present, else the direct peer IP.
- **Response on limit:** `429 Too Many Requests` with a `Retry-After` header.
- **Trusted-proxy parsing:** the deployer is responsible for terminating untrusted proxies. We document this in the README (`X-Forwarded-For` is taken at face value).

#### Env vars
| Var | Default | Description |
|-----|---------|-------------|
| `RATE_LIMIT_ENABLED` | `true` | Master switch |
| `RATE_LIMIT_PER_IP` | `60/10minutes` | slowapi-format string |

The rate limit applies to all inference endpoints (`/detect`, `/mask`, `/detect/batch`). `/health`, `/ready`, and `/docs` are exempt.

---

## 3. Batch Endpoint

### `POST /detect/batch`

Accepts an array of texts; returns a parallel array of per-item results.

**Request body**
```json
{
  "items": [
    { "text": "Call me at +33 6 12 34 56 78", "mode": "balanced" },
    { "text": "alice@example.com", "labels": ["private_email"] }
  ],
  "mask": false
}
```

Each item has the same fields as a `/detect` request body (`text`, `mode?`, `labels?`, `mask_char?`). Top-level `mask` applies to every item.

**Response `200 OK`**
```json
{
  "results": [
    {
      "status": "ok",
      "entities": [ ... ],
      "masked_text": null,
      "meta": { "entity_count": 1, "processing_ms": 89 }
    },
    {
      "status": "error",
      "error": { "code": "inference_failed", "message": "..." }
    }
  ],
  "meta": {
    "model": "openai/privacy-filter",
    "batch_size": 2,
    "processing_ms": 142
  }
}
```

Per-item response **does not** echo the input `text` (it's already in the request, and order matches `items`). This keeps batch responses compact.

### Error semantics — hybrid

- **Request-level errors** (malformed JSON, missing `items`, batch size > `MAX_BATCH_SIZE`, batch token total > `MAX_BATCH_TOTAL_TOKENS`, model not loaded): return `4xx` / `503` for the whole batch, no `results` array.
- **Per-item errors** (item too long for non-chunking batch, inference failure on one item): return `200` with that item's `status: "error"`. Other items succeed normally.

### Constraints

| Constraint | Default | Behavior on violation |
|------------|---------|------------------------|
| `MAX_BATCH_SIZE` | `32` items | Request-level `422` |
| `MAX_BATCH_TOTAL_TOKENS` | `200_000` | Request-level `422` |
| Per-item token limit | `CHUNK_SIZE_TOKENS` (`120_000`) | Per-item `422` with `error.code: "item_too_long"` and message hinting `/detect` for chunked input |

**Why no chunking in batch:** keeps the per-item mental model simple, bounds memory predictably (no item explodes into N chunks queued in flight), and matches the contract that batch is for many short texts. Long texts use single `/detect` and chunk transparently there.

### Processing model

Items are processed **sequentially** through the same `MAX_CONCURRENT_INFERENCES` semaphore as single-call requests. A 32-item batch with 2 concurrent inference slots will interleave fairly with any concurrent single requests. Order of `results` matches order of `items`.

---

## 4. Label Filtering

Optional `labels?: string[]` on `/detect`, `/mask`, and per item in `/detect/batch`.

- **Omitted or `null`:** return all 8 labels (default).
- **Provided:** server runs full inference, then filters spans server-side before responding. No model-level optimization (the model is a single-pass tagger over all labels).
- **Validation:** every label must be one of the 8 known values (`private_person`, `private_email`, `private_phone`, `private_address`, `account_number`, `private_url`, `private_date`, `secret`). Unknown label → `422` with the offending value named in the error.

Applies identically in `/mask` (only spans of requested labels get masked) and in batch items.

---

## 5. Offset Semantics

Span `start` and `end` are **Python code-point character indices** into the request text (`len(text[:span.start])` model). This matches the natural output of HF fast tokenizers' `offset_mapping` and what FastAPI / Pydantic produce.

Documented explicitly in OpenAPI descriptions and README. Clients in languages with different default string indexing (Java/Swift/Rust) should convert by re-encoding `text` to a code-point sequence first.

---

## 6. Chunker Correctness Fix

The PRD pseudocode uses `tokenizer.decode(tokens[:start_tok])` to compute `char_start`. This drifts whenever `decode(encode(text)) != text` (BPE normalization, byte-level tokenizer markers, NFC/NFKC), accumulating per chunk and producing wrong span offsets in the response.

### Replacement algorithm

```python
def detect_long_text(text, tokenizer, model, chunk_size=120_000, overlap=512):
    assert tokenizer.is_fast, "Fast tokenizer required for offset_mapping"

    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    tokens = encoding["input_ids"]
    offsets = encoding["offset_mapping"]   # [(char_start, char_end), ...]

    if len(tokens) <= chunk_size:
        return run_inference(text)

    all_entities = []
    step = chunk_size - overlap
    for start_tok in range(0, len(tokens), step):
        end_tok = min(start_tok + chunk_size, len(tokens))
        char_start = offsets[start_tok][0]
        char_end   = offsets[end_tok - 1][1]

        # Optional: SMART_SPLIT nudges char_start/char_end to the nearest
        # paragraph boundary within ±200 tokens of the calculated split.
        char_start, char_end = maybe_smart_split(text, offsets, start_tok, end_tok, char_start, char_end)

        chunk_text = text[char_start:char_end]
        entities = run_inference(chunk_text)
        for e in entities:
            all_entities.append({**e,
                                 "start": e["start"] + char_start,
                                 "end":   e["end"]   + char_start})

        if end_tok == len(tokens):
            break

    return deduplicate_spans(all_entities)
```

**Key changes from PRD:**
- No `decode()` round-trip. Source text is sliced directly using offsets from the original encoding.
- Asserts fast tokenizer at startup; fail fast if a slow tokenizer is configured.
- Deduplication logic is unchanged from the PRD.

The `SMART_SPLIT` heuristic still applies — it now nudges character positions instead of token positions, but the ±200-token window semantics are preserved.

---

## 7. Concurrency Model

### Single model singleton + bounded inference

- The model is loaded once at startup into a module-level singleton.
- A single `asyncio.Semaphore(MAX_CONCURRENT_INFERENCES)` gates all inference calls (single, batch items, and chunks within a chunked request all share the same semaphore).
- Inference itself is invoked via `await asyncio.to_thread(run_inference, ...)` so the FastAPI event loop never blocks on CPU-bound work.

### Why this shape

- Predictable memory: one ~4GB model, period. Operators who want more throughput run more containers (horizontal), not more workers per container.
- Queue depth is bounded by the inference semaphore + uvicorn's incoming request queue; under sustained overload, requests wait, then time out at the client. This is preferable to OOMing the container.
- Threading (vs. multi-processing) keeps the model singleton intact; `torch` releases the GIL during ops, so threads do achieve real parallelism for inference.

### Env var
| Var | Default | Description |
|-----|---------|-------------|
| `MAX_CONCURRENT_INFERENCES` | `2` | Concurrent inferences allowed per container |

### Throughput target (revised from PRD §10)

> ≥10 req/s on 4-core CPU **for short inputs (≤2k tokens), with `MAX_CONCURRENT_INFERENCES` tuned to the host**. Tuning guidance lives in the README.

The PRD's original wording said "≥10 req/s … with batching." We do **not** implement model-level (in-process) batching in v1 — the `/detect/batch` endpoint processes items sequentially through the inference semaphore, which improves client ergonomics but not raw throughput. Throughput at this target depends on `MAX_CONCURRENT_INFERENCES` × CPU parallelism the model can extract via the GIL-released `torch` ops.

---

## 8. Health & Readiness

Two endpoints, distinct semantics:

| Endpoint | Behavior | Use |
|----------|----------|-----|
| `GET /health` | Always `200` while the process is alive. Body includes `model_loaded: bool`, `device: str`. | Liveness probes; "is the process up?" |
| `GET /ready` | `200` once the model is loaded; `503` until then (with `{"status":"loading"}`). | k8s readiness probes; "should I send traffic?" |

Compose `start_period` is bumped from 120s to **600s** to cover cold-start model download (~3 GB) on slow connections. The healthcheck still calls `/health`.

Both endpoints are exempt from rate limiting.

---

## 9. GPU Support

Ship `docker-compose.gpu.yml` as a Compose override.

```yaml
# docker-compose.gpu.yml
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

GPU users: `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up`. CPU users get the existing path unchanged. The `DEVICE` env var still controls the runtime device choice.

A second `Dockerfile.gpu` is **not** required: the override only changes the base image and device reservation; the application code path is identical.

> **Caveat:** the GPU override pins the CUDA base image. Users on different host CUDA driver versions may need to adjust the tag. Documented in the README.

---

## 10. MCP Server (sidecar)

Ship a thin stdio MCP server in the same repo, packaged separately so MCP-uninterested users carry no extra dependency.

### Project structure addition

```
pii-filter/
├── app/                    # existing FastAPI service
├── mcp_server/             # new
│   ├── pyproject.toml      # separately publishable
│   ├── pii_filter_mcp/
│   │   ├── __init__.py
│   │   └── __main__.py     # mcp.Server with two tools
│   └── README.md           # Claude Desktop config snippet
└── docker-compose.yml
```

### Tools exposed

| Tool | Args | Returns |
|------|------|---------|
| `detect_pii` | `text: str`, `mode?: "precise"\|"balanced"\|"recall"`, `labels?: string[]` | `{ entities: [...], entity_count: int }` |
| `mask_pii` | `text: str`, `mode?`, `labels?`, `mask_char?` | `{ masked_text: str, entity_count: int }` |

Both are direct mappings to the REST endpoints with the same defaults. Errors propagate as MCP tool errors.

### Configuration

| Var | Default | Description |
|-----|---------|-------------|
| `PII_FILTER_URL` | `http://localhost:8080` | Where the MCP server reaches the REST API |
| `PII_FILTER_TIMEOUT_SEC` | `60` | HTTP timeout for tool calls |

### Claude Desktop registration (README snippet)

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

### Out of scope for the MCP server in v1

- Batch tool (the LLM doesn't naturally batch — it calls per-message)
- Streaming responses (MCP tool results are one-shot)
- SSE/HTTP transport variant (stdio only for v1)

---

## 11. Updated Env Var Reference (delta only)

| Var | Default | Source |
|-----|---------|--------|
| `RATE_LIMIT_ENABLED` | `true` | §2 |
| `RATE_LIMIT_PER_IP` | `60/10minutes` | §2 |
| `MAX_BATCH_SIZE` | `32` | §3 |
| `MAX_BATCH_TOTAL_TOKENS` | `200000` | §3 |
| `MAX_CONCURRENT_INFERENCES` | `2` | §7 |
| `PII_FILTER_URL` | `http://localhost:8080` | §10 (MCP only) |
| `PII_FILTER_TIMEOUT_SEC` | `60` | §10 (MCP only) |

All vars from the baseline PRD remain valid.

---

## 12. Updated Project Structure (delta from PRD §8)

```
pii-filter/
├── docker-compose.yml
├── docker-compose.gpu.yml         # NEW (§9)
├── Dockerfile
├── .env.example
├── requirements.txt
├── README.md
├── app/
│   ├── main.py                    # routes incl. /detect, /mask, /detect/batch, /health, /ready
│   ├── model.py                   # singleton + semaphore-gated inference
│   ├── chunker.py                 # offset_mapping-based, no decode round-trip (§6)
│   ├── ratelimit.py               # NEW — slowapi setup
│   ├── schemas.py                 # incl. Batch* models
│   └── config.py
└── mcp_server/                    # NEW (§10)
    ├── pyproject.toml
    └── pii_filter_mcp/
        ├── __init__.py
        └── __main__.py
```

---

## 13. Smoke Tests (additions to PRD §13 M7)

Beyond the PRD's existing test plan:
- `/detect/batch` happy path with mixed item shapes (some with `labels`, some without).
- `/detect/batch` with one item exceeding `CHUNK_SIZE_TOKENS` → `200` with that item as `status: error`, others succeed.
- `/detect/batch` exceeding `MAX_BATCH_TOTAL_TOKENS` → request-level `422`.
- Rate limit: `61` requests in <10 min from same IP → 61st returns `429`.
- Label filtering: request with `labels: ["private_email"]` returns only emails; spans of other labels are absent from response.
- Chunker offset correctness: input containing multibyte text (CJK + emoji) at chunk boundaries — verify returned offsets exactly match `text[start:end]` for every span across all chunks.
- `/ready` returns `503` until the model finishes loading on a fresh boot.
- MCP server: integration test stubbing `PII_FILTER_URL` to a fake server, verifies `detect_pii` and `mask_pii` tools round-trip correctly.

---

## 14. Resolved PRD Open Questions

| PRD §14 # | Question | Resolution | This doc § |
|-----------|----------|------------|------------|
| 1 | Batch endpoint? | Yes | §3 |
| 2 | Auth? | No, but rate limit per IP | §2 |
| 3 | Label filtering? | Yes | §4 |
| 4 | GPU base image override? | Yes | §9 |
| 5 | Char or byte offsets? | Code-point characters | §5 |
| 6 | Overlap tuning? | Fixed | (no change) |
| 7 | SSE streaming? | No | (out of scope) |

---

## 15. New Dependency Additions

`requirements.txt` additions on top of PRD §7:
- `slowapi>=0.1.9` — rate limiting

`mcp_server/pyproject.toml` dependencies:
- `mcp>=1.0.0` — Anthropic MCP Python SDK
- `httpx>=0.27.0` — REST client to call pii-filter

---

## 16. Out of Scope (v1) — Reaffirmed

Same as PRD §12, with the following moved into v1:
- Batch endpoint (now in §3)
- Label filtering (now in §4)
- MCP server (now in §10)

Still explicitly out of v1:
- Authentication (rate-limit only)
- SSE/streaming
- Async job queue
- Fine-tuning interface
- Kubernetes Helm chart
- Prometheus metrics exporter
