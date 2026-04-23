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

## GPU (experimental)

A `docker-compose.gpu.yml` override is shipped as a **template** for GPU deployment. It reserves an NVIDIA GPU device but does not by itself produce a working image — the CUDA runtime base it references lacks Python 3.11 and the app code.

To run on GPU, you will need one of:

1. **Author a `Dockerfile.gpu`** (`FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04`), install Python 3.11 + `requirements.txt`, and copy `app/` — then reference it via `build: { dockerfile: Dockerfile.gpu }` in `docker-compose.gpu.yml`.
2. **Or** modify the base `Dockerfile` to install CUDA-enabled `torch` wheels and drop the `image:` override from `docker-compose.gpu.yml`, keeping only the `deploy.resources.reservations.devices` block.

Both paths require NVIDIA Container Toolkit on the host. PRs welcome for a ready-made `Dockerfile.gpu`.

## Security

- No data persisted. No external calls after first model download.
- Container runs as a non-root user (uid 10001).
- Auth is **not** built in; deployers should restrict network access (reverse proxy, NetworkPolicy, VPN). The per-IP rate limit is a brake against runaway clients, not an access-control mechanism.
- `X-Forwarded-For` is taken at face value — terminate untrusted proxies before reaching this service.

## MCP server

See [`mcp_server/README.md`](mcp_server/README.md).

> **Install in a separate venv from the REST service.** The MCP SDK pulls in `sse-starlette` which requires a newer `starlette` than `fastapi==0.115.0` allows. Mixing both in one venv triggers a `pip check` warning and can break the FastAPI app on accidental upgrades. In production this is a non-issue: the REST service runs in Docker, while Claude Desktop launches the MCP server in its own Python environment.

## Development

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest tests/        # backend tests
pytest mcp_server/tests/  # MCP server tests
```

## License

Apache 2.0 (matches the model license).
