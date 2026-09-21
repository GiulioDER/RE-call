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
   `/etc/recall-aml/hosted.env`. Supply the admission-time `RECALL_AML_DATABASE_URL`,
   `RECALL_AML_API_KEY`, and its single-tenant binding `RECALL_AML_AUTHORIZED_USER_ID`,
   substitute the remaining secrets, set owner `root:recall-aml`, and set
   mode `0640`. Their absence before the run opens is expected. The service remains stopped until
   all three values exist. Before admission, load the available target-host settings and run
   `python scripts/aml_hosted_preflight.py --phase prepare`. After all three admission values arrive,
   run the same command with `--phase launch`. The preflight reports only presence states and
   never prints configuration values.
7. Install `infra/systemd/recall-aml.service`, reload systemd, and start the unit.
8. Verify `http://127.0.0.1:18004/health` and `/version` locally.
9. Add the hostname route to the existing Cloudflare Tunnel, keeping the final catch-all rule last.
10. Verify HTTPS from outside VPS2, then run the contract, concurrency, isolation, restart, and
    deletion suites against that exact endpoint.

The registered external reliability run is duration bound and defaults to thirty minutes:

```sh
python scripts/aml_hosted_verify.py --base-url https://memory.example.com --mode all \
  --soak-minutes 30
```

The output records every request count, actual elapsed time, error count, and nearest rank latency
percentile. Health calls are not included in the Add or Search distributions.

## Freeze receipt

Build the wheel, then create the immutable release manifest before submission:

```sh
python -m build --wheel
python -m scripts.aml_release_manifest \
  --wheel dist/recall_rag-0.14.0-py3-none-any.whl \
  --commit "$(git rev-parse HEAD)" \
  --variant C6_code4_exact_bm25 \
  --output /var/lib/recall-aml/release-manifest.json
```

The generator refuses a mismatched commit, tracked checkout changes, missing artifacts, an unknown
variant, and an existing output path. It binds the wheel, service unit, Cloudflare template,
environment template, compiler source and prompt, dependency lock, project metadata, and immutable
preregistration by SHA256. For the Code4 candidate it also binds the canonical BM25
implementation bytes, the embedding profile, the word window size and stride, and the lexical
retrieval switch.
It records credential variable names but never reads or serializes their values.

`C6_code4_exact_bm25` is the selected CAMBench Coding candidate for the next official AML
compatibility Smoke. It stores 160 word content only windows, embeds them with Voyage Code4,
computes exact dense top 100, and fuses dense and canonical BM25 ranks with one stable source
session plus segment ordering profile. It has no compiler, facet generator, reranker, graph path,
or evidence packer. User and session isolation remain mandatory. The local and isolated endpoint
receipts establish run readiness only. Do not describe the version as an official AML result until
the platform Smoke, Full evaluation, and organizer review have completed for this exact frozen
version.

Before submission also record the release tag, the manifest itself, `/version` response, database
schema status, and verified Cloudflare route under `docs/results/aml-hosted-v1/`. Never place the
environment contents or API key in an artifact.

Do not run Full unless every mechanical promotion gate in the committed preregistration passes.
