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
6. When the admission run opens, install `infra/systemd/hosted.env.example` as
   `/etc/recall-aml/hosted.env`. Supply the admission-time `RECALL_AML_DATABASE_URL` and
   `RECALL_AML_API_KEY`, substitute the remaining secrets, set owner `root:recall-aml`, and set
   mode `0640`. Their absence before the run opens is expected. The service remains stopped until
   both values exist. Before admission, load the available target-host settings and run
   `python scripts/aml_hosted_preflight.py --phase prepare`. After both admission values arrive,
   run the same command with `--phase launch`. The preflight reports only presence states and
   never prints configuration values.
7. Install `infra/systemd/recall-aml.service`, reload systemd, and start the unit.
8. Verify `http://127.0.0.1:18004/health` and `/version` locally.
9. Add the hostname route to the existing Cloudflare Tunnel, keeping the final catch-all rule last.
10. Verify HTTPS from outside VPS2, then run the contract, concurrency, isolation, restart, and
    deletion suites against that exact endpoint.

## Freeze receipt

Build the wheel, then create the immutable release manifest before submission:

```sh
python -m build --wheel
python -m scripts.aml_release_manifest \
  --wheel dist/recall_rag-0.13.0-py3-none-any.whl \
  --commit "$(git rev-parse HEAD)" \
  --variant A4_pack_7000 \
  --output /var/lib/recall-aml/release-manifest.json
```

The generator refuses a mismatched commit, tracked checkout changes, missing artifacts, an unknown
variant, and an existing output path. It binds the wheel, service unit, Cloudflare template,
environment template, compiler source and prompt, dependency lock, project metadata, and immutable
preregistration by SHA256. It records credential variable names but never reads or serializes their
values.

Before submission also record the release tag, the manifest itself, `/version` response, database
schema status, and verified Cloudflare route under `docs/results/aml-hosted-v1/`. Never place the
environment contents or API key in an artifact.

Do not run Full unless every mechanical promotion gate in the committed preregistration passes.
