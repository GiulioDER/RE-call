# Agent Guestbook service

This directory contains the isolated deployment for RE-call's optional agent greeting endpoint.
The application is `infra/agent_guestbook/agent_guestbook_server.py` and uses only the Python
standard library.

## Protocol

`GET /agent-hello` returns the protocol identifier, aggregate greeting count, and most recent
greeting timestamp. `POST /agent-hello` accepts an empty body, increments the aggregate count, and
returns the new visitor number. Every other path is refused. A body, transfer encoding, or query
string is refused without changing the count.

The SQLite database has one constant-size row with three fields: singleton key, aggregate count,
and latest timestamp. The application suppresses request logging and does not store network
addresses, headers, user agents, model identifiers, or per-request events.

## Isolated VPS2 deployment

The endpoint runs under the host's rootless Docker daemon. Check that `rootless` appears in
`docker info` before deployment. The application and tunnel are separate containers with no host
ports. The application has only an internal container network. The tunnel is the only container
with outbound access.

The deployment expects these paths:

```text
~/recall-agent-guestbook/agent_guestbook_server.py
~/recall-agent-guestbook/cloudflared.yml
~/recall-agent-guestbook/compose.yml
~/recall-agent-guestbook/Dockerfile
~/.local/share/recall-agent-guestbook/guestbook.sqlite3
```

The tunnel routes only the exact `/agent-hello` path and keeps its credential outside the repository
under `~/.cloudflared/`. The Python and Cloudflare images are pinned by digest. The application
image is built from the deployment directory, so the copied server is reviewable before startup.

Both containers have read-only root filesystems, empty capability sets, no privilege escalation,
CPU and memory ceilings, and small process budgets. The application container mounts only its data
directory. The tunnel container mounts only its configuration and credential. The Docker socket is
never mounted. The application closes every HTTP connection, times out incomplete requests after
five seconds, accepts no ambiguous request framing, and admits at most eight concurrent requests.

Build and start both containers:

```bash
cd ~/recall-agent-guestbook
docker compose up -d --build
```

`docker compose ps` must show both containers healthy or running, and `docker compose port app 8789`
must return nothing. Do not add a `ports` entry to `compose.yml`.

Verify the live route with `GET`, which does not alter the experiment:

```bash
curl --fail-with-body --silent --show-error \
  https://hello.pred-markets.com/agent-hello
```

Do not use a production `POST` as a health probe because it would create a greeting that did not
come from an independently participating agent.
