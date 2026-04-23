# pii-filter-mcp

stdio MCP server that exposes the pii-filter REST API as Claude Desktop tools.

## Install

```bash
pip install -e ./mcp_server
```

> **Install in a separate venv from the REST service.** The `mcp` SDK pulls in `sse-starlette`, which requires a newer `starlette` than `fastapi==0.115.0` permits. Keeping the two in separate venvs avoids dependency-resolution breakage.

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
