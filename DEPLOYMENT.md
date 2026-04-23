# Deployment: VPS + Cloudflare (Flexible SSL)

This recipe puts `pii-filter` on a VPS behind Cloudflare with a custom subdomain. Cloudflare handles TLS at the edge; the VPS only needs to speak plain HTTP on a port Cloudflare can proxy to.

- **Edge URL:** `https://pii.yourdomain.com/mcp` (Cloudflare terminates TLS on standard 443)
- **Origin URL:** `http://<VPS_IP>:8080` (Cloudflare connects to your VPS on port 8080)
- **App changes required:** none. The app stays HTTP-only on its default port.

---

## 1. Choose a Cloudflare-supported origin port

Cloudflare's free/pro plans only proxy to a fixed set of HTTP ports on your origin. The safe picks for `pii-filter` are:

| Port | Notes |
|------|-------|
| `8080` | Default. Works with `docker-compose.yml` out of the box. |
| `8880` | Alternative if `8080` is already used by something else. |

Other allowed HTTP origin ports: `80, 2052, 2082, 2086, 2095`. Do **not** use `3402`, `443`, or any unlisted port — Cloudflare's proxy will refuse.

---

## 2. Run the container

```bash
git clone https://github.com/voiceflow-gallagan/privacy-filter.git
cd privacy-filter
cp .env.example .env
# Optional: edit .env. Leave PORT=8080 unless you picked 8880 above.
docker compose up -d
```

First boot downloads ~3 GB of model weights to the `pii_model_cache` volume. Wait until `/ready` returns 200:

```bash
curl -s http://localhost:8080/ready
# {"status":"ready"}
```

---

## 3. VPS firewall — allow Cloudflare only

Open the origin port **only** to Cloudflare's IP ranges so the service is not directly reachable from the open internet.

Cloudflare publishes its ranges at [cloudflare.com/ips](https://www.cloudflare.com/ips/). Fetch them fresh every run — they change:

```bash
# IPv4
for ip in $(curl -s https://www.cloudflare.com/ips-v4); do
  sudo ufw allow from "$ip" to any port 8080 proto tcp
done
# IPv6
for ip in $(curl -s https://www.cloudflare.com/ips-v6); do
  sudo ufw allow from "$ip" to any port 8080 proto tcp
done
sudo ufw reload
```

Verify:

```bash
sudo ufw status | grep 8080 | wc -l    # ~25-30 rules
```

**Important:** this locks the origin port to Cloudflare-only — you won't be able to `curl http://<VPS_IP>:8080/health` from your laptop anymore. That's the intent. Health-check via the Cloudflare URL once DNS is live: `curl https://pii.yourdomain.com/health`.

If you use a different firewall (iptables, firewalld, cloud-provider security group), apply the same Cloudflare-IP allowlist pattern.

---

## 4. Cloudflare dashboard setup

### 4a. DNS record

**DNS → Records → Add record**

| Field | Value |
|-------|-------|
| Type | `A` |
| Name | `pii` (resolves to `pii.yourdomain.com`) |
| IPv4 address | Your VPS public IP |
| Proxy status | **Proxied** (orange cloud ☁️) |

Wait for DNS to propagate (usually seconds on Cloudflare).

### 4b. SSL/TLS mode

**SSL/TLS → Overview → Set to `Flexible`**

This tells Cloudflare to speak HTTPS to the client but plain HTTP to your origin. No certs needed on the VPS.

> **Security tradeoff:** traffic between Cloudflare and your VPS is HTTP. The firewall rules in §3 mitigate this (only Cloudflare-range IPs can reach your origin), but if that is not enough for your threat model, upgrade to **Full (Strict)** and install a Cloudflare Origin Certificate on the VPS (see §7).

### 4c. Origin Rule — rewrite origin port

By default, Cloudflare connects to origin on port 80 (Flexible) or 443 (Full). You need to tell it to use 8080 instead.

**Rules → Origin Rules → Create rule**

| Field | Value |
|-------|-------|
| Rule name | `pii-filter origin port` |
| If... | `Hostname` `equals` `pii.yourdomain.com` |
| Then... | ☑ Rewrite to... → `HTTP` port `8080` |

Save and deploy.

---

## 5. Verify

From anywhere on the public internet:

```bash
curl -s https://pii.yourdomain.com/ready
# {"status":"ready"}

curl -s -X POST https://pii.yourdomain.com/detect \
  -H 'Content-Type: application/json' \
  -d '{"text":"Email alice@example.com"}' | jq
```

You should get a JSON response with a `private_email` entity.

---

## 6. Claude Desktop config

Open `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) and add:

```json
{
  "mcpServers": {
    "pii-filter": {
      "url": "https://pii.yourdomain.com/mcp"
    }
  }
}
```

Restart Claude Desktop. The `detect_pii` and `mask_pii` tools appear under the server.

---

## 7. (Optional) Upgrade to Full (Strict) with a Cloudflare Origin Certificate

When you want encryption between Cloudflare and your VPS too:

1. **SSL/TLS → Origin Server → Create Certificate.** Accept defaults (RSA 2048, 15 years). Cloudflare generates a cert+key pair for `*.yourdomain.com`.
2. Save the cert and private key on the VPS, e.g. `/etc/ssl/cloudflare/origin.crt` + `/etc/ssl/cloudflare/origin.key`.
3. Put a lightweight TLS terminator in front of pii-filter (Caddy or Nginx) that serves HTTPS on port 8080 using the Cloudflare origin cert, reverse-proxying to the container. The `docker-compose.yml` gets one extra service; the pii-filter container itself still listens on HTTP internally.
4. Flip **SSL/TLS → Overview → Full (Strict)**.

This is a drop-in upgrade — no code changes, no Let's Encrypt dance.

---

## 8. Troubleshooting

| Symptom | Likely cause |
|---------|--------------|
| `526 Invalid SSL certificate` at the edge | SSL mode is `Full (Strict)` but origin doesn't have a valid cert. Either switch to `Flexible`, or install an Origin Certificate per §7. |
| `522 Connection timed out` | VPS firewall blocks Cloudflare — re-run §3. |
| `521 Web server is down` | Container not running or `PORT` mismatch. `docker compose ps`, `curl localhost:8080/health` from the VPS itself. |
| `502 Bad gateway` | Origin Rule port mismatch — check §4c. |
| `/mcp` returns 421 Misdirected | Old bug with FastMCP Host check; fixed in v1.0.0. Rebuild if you're on an older commit. |

---

## 9. Operational notes

- **Rate limiting:** the `/detect`, `/mask`, `/detect/batch` endpoints are rate-limited per source IP. With Cloudflare in front, the client IP comes through in the `X-Forwarded-For` header — the service reads it, so rate limiting works correctly. The `/mcp` endpoint is **not** rate-limited in v1; Cloudflare's edge WAF / rate limiting is the recommended control if you need one.
- **Logs:** `docker compose logs -f pii-filter`. Input text is not logged by default (spec §10 guarantee).
- **Upgrades:** `git pull && docker compose up -d --build`. The model cache volume survives rebuilds.
- **Cloudflare IP range drift:** Cloudflare updates their IP ranges occasionally. Re-run the firewall script in §3 every few months, or automate it via cron.
