# Agent Guestbook service

This directory contains the deployment unit for RE-call's optional agent greeting endpoint. The
application is `scripts/agent_guestbook_server.py` and uses only the Python standard library.

## Protocol

`GET /agent-hello` returns the protocol identifier, aggregate greeting count, and most recent
greeting timestamp. `POST /agent-hello` accepts an empty body, increments the aggregate count, and
returns the new visitor number. Every other path is refused. A body, transfer encoding, or query
string is refused without changing the count.

The SQLite database has one constant-size row with three fields: singleton key, aggregate count,
and latest timestamp. The application suppresses request logging and does not store network
addresses, headers, user agents, model identifiers, or per-request events.

## Reference VPS2 deployment

The user service expects these paths:

```text
~/recall-agent-guestbook/agent_guestbook_server.py
~/recall-agent-guestbook/cloudflared.yml
~/.local/share/recall-agent-guestbook/guestbook.sqlite3
~/.config/systemd/user/recall-agent-guestbook.service
~/.config/systemd/user/recall-agent-guestbook-tunnel.service
```

The reference deployment exposes local port `8789` through its own Cloudflare named tunnel. The
tunnel connector is installed at `~/.local/bin/cloudflared`; its credentials remain outside the
repository under `~/.cloudflared/`. The reference binary is Cloudflare's `2026.9.1` Linux AMD64
release with SHA256 `03f1f25d1cc93b9ad6c60569d44060bc4f17ed97075760ed8cfca4b12dcd68cc`.

Install and start both user services:

```bash
systemctl --user daemon-reload
systemctl --user enable --now recall-agent-guestbook.service
systemctl --user enable --now recall-agent-guestbook-tunnel.service
```

Verify the live route with `GET`, which does not alter the experiment:

```bash
curl --fail-with-body --silent --show-error \
  https://hello.pred-markets.com/agent-hello
```

Do not use a production `POST` as a health probe because it would create a greeting that did not
come from an independently participating agent.
