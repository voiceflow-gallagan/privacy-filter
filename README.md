# pii-filter

Self-hosted PII detection and masking REST API powered by `openai/privacy-filter`.

- Zero-config startup: `docker compose up`
- Auto-chunking for inputs of any length (token-aligned, no offset drift)
- Per-IP rate limiting (configurable)
- Built-in MCP server at `/mcp` (streamable-HTTP) for Claude Desktop and other MCP clients

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
- `POST /mcp` — MCP streamable-HTTP endpoint (tools: `detect_pii`, `mask_pii`).

### Span offsets

Spans use **Python code-point character indices** into the request text:
`text[entity.start:entity.end]` returns the surface form. Clients in Java/Swift/Rust should convert from their default string indexing.

## Configuration

All knobs are env vars (with defaults shown):

| Var | Default | Notes |
|-----|---------|-------|
| `DEVICE` | `cpu` | Set to `cuda` to use GPU |
| `DEFAULT_MODE` | `balanced` | Score threshold applied to model spans: `precise` ≥ 0.85, `balanced` ≥ 0.55, `recall` keeps all. Request-level `mode` overrides this. |
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

The MCP server is mounted in the same FastAPI process at `/mcp` over the
streamable-HTTP transport. There is no separate sidecar and no extra container
to deploy.

### Tools exposed

- `detect_pii(text, mode?, labels?)` — returns `{ entities, entity_count }`.
- `mask_pii(text, mode?, labels?, mask_char?)` — returns `{ masked_text, entity_count }`.

Both dispatch to the same in-process detection pipeline as `POST /detect`; no
HTTP self-loopback.

### Configure an MCP client

Point any streamable-HTTP MCP client at `http://<host>:8080/mcp`. For Claude
Desktop, add an entry like:

```json
{
  "mcpServers": {
    "pii-filter": {
      "url": "http://localhost:8080/mcp"
    }
  }
}
```

(The exact schema for remote MCP servers varies by client version — consult
your client's docs if the shape above is not accepted.)

> **Rate limiting.** `/mcp` is **not** rate-limited in v1. Operators are
> expected to front it with network-level controls (reverse proxy, VPN,
> auth). Per-IP limits still apply to the public REST routes.

## Development

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest tests/
```

## License

Apache 2.0 (matches the model license).
