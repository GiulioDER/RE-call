# RE-call Hosted 1.0 VPS2 runbook

This runbook provisions the frozen hosted API without placing secrets in the repository. A new
system account, PostgreSQL database and roles, systemd unit, Cloudflare hostname, and protected
environment file require the VPS2 operator path.

## Provisioning order

1. Create the `recall-aml` system account with no interactive login.
2. Create the dedicated `recall_aml` database, an object owner role, and a serving role without DDL.
3. Install the frozen repository commit into `/opt/recall-aml/current` and install
   `.[hosted]` into `/opt/recall-aml/venv`.
4. As the object owner, apply the 1,024 dimension Voyage schema:

   ```sh
   recall --migration-dsn "$RECALL_MIGRATION_DSN" --table recall_aml_chunks schema --dim 1024 apply
   ```

5. Generate serving grants with `recall schema grants --role recall_aml_serving`, review them, and
   apply them as the object owner.
6. Install `infra/systemd/hosted.env.example` as `/etc/recall-aml/hosted.env`, substitute secrets,
   set owner `root:recall-aml`, and set mode `0640`.
7. Install `infra/systemd/recall-aml.service`, reload systemd, and start the unit.
8. Verify `http://127.0.0.1:18004/health` and `/version` locally.
9. Add the hostname route to the existing Cloudflare Tunnel, keeping the final catch-all rule last.
10. Verify HTTPS from outside VPS2, then run the contract, concurrency, isolation, restart, and
    deletion suites against that exact endpoint.

## Freeze receipt

Before submission record the release tag, Git commit, wheel or image SHA256, redacted environment
SHA256, `/version` response, database schema status, Cloudflare route, and systemd unit digest under
`docs/results/aml-hosted-v1/`. Never place the environment contents or API key in an artifact.

Do not run Full unless every mechanical promotion gate in the committed preregistration passes.
