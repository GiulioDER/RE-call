# Runbook — the overnight embedder-gap run (vast.ai)

Target: a **vast.ai instance**, which is a Docker container, not a VM — so Postgres is installed
natively, **not** via `docker compose`. Every BEIR corpus is public; nothing sensitive leaves your
infrastructure, and the private memory corpus is not part of this run at all.

Reference instance (2026-07-26): `45933936` — EPYC 7763, 64 vCPU, 258 GB, 2 TB disk, $0.569/hr.

Each step ends with **how you know it worked**. Several of these fail by producing a plausible
result rather than an error, which is why the checks are not optional.

---

## 1. Connect

```bash
ssh -p <PORT> root@95.253.220.115
```

✅ **Right if:** `nproc` prints 64 and `free -g` shows ~258 GB.

## 2. System packages + Postgres 17 + pgvector (native, no docker)

```bash
apt-get update && apt-get install -y curl ca-certificates gnupg lsb-release git unzip python3-venv python3-dev build-essential && install -d /usr/share/postgresql-common/pgdg && curl -fsSL -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc https://www.postgresql.org/media/keys/ACCC4CF8.asc && echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] https://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list && apt-get update && apt-get install -y postgresql-17 postgresql-17-pgvector
```

Start it **without systemd** (containers have no systemd):

```bash
pg_ctlcluster 17 main start && su postgres -c "psql -c \"CREATE USER recall WITH PASSWORD 'recall' SUPERUSER;\"" && su postgres -c "createdb -O recall recall" && su postgres -c "psql -d recall -c 'CREATE EXTENSION IF NOT EXISTS vector;'"
```

✅ **Right if:** this prints `(1 row)` containing `vector`:

```bash
PGPASSWORD=recall psql -h 127.0.0.1 -U recall -d recall -c "select extname from pg_extension where extname='vector';"
```

⚠️ If `vector` is missing, **every dense arm silently scores near zero** and the whole run is
worthless. Do not continue past this check.

💡 If `pg_ctlcluster` fails with a locale error: `apt-get install -y locales && locale-gen en_US.UTF-8`.

## 3. Repo and Python

```bash
git clone --branch research/vocab-gap-predictor https://github.com/GiulioDER/RE-call.git /root/recall && cd /root/recall && python3 -m venv .venv && ./.venv/bin/pip install -U pip && ./.venv/bin/pip install -e '.[fastembed]' voyageai
```

✅ **Right if:** this prints `ok 20000` (the module loads *and* carries the restated cap):

```bash
cd /root/recall && ./.venv/bin/python -c "from recall.eval.gap_run import MAX_DOCS; print('ok', MAX_DOCS)"
```

## 4. BEIR datasets

CQADupStack ships as one zip whose subforums become separate corpora, so it needs renaming — the
runner looks for `cqadupstack-tex`, the zip gives `cqadupstack/tex`.

```bash
mkdir -p /root/recall/beir && cd /root/recall/beir && for d in nfcorpus scifact scidocs fiqa arguana cqadupstack; do curl -sSLO "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/$d.zip" && unzip -q -o "$d.zip" && rm "$d.zip"; done && for f in cqadupstack/*/; do mv "$f" "cqadupstack-$(basename $f)"; done && rmdir cqadupstack
```

✅ **Right if:** this prints **17**:

```bash
ls -d /root/recall/beir/*/ | wc -l && ls /root/recall/beir/*/corpus.jsonl | wc -l
```

## 5. Voyage key — test it before the night, not during

```bash
export VOYAGE_API_KEY='<paste-your-key>'
```

✅ **Right if:** this prints a dimension (e.g. `1024`) and **not** a traceback:

```bash
cd /root/recall && ./.venv/bin/python -c "from recall.embeddings import VoyageEmbedder; print(len(VoyageEmbedder().embed(['smoke test'])[0]))"
```

⚠️ **Check your Voyage quota first.** At `MAX_DOCS=20000` this run embeds roughly **290 000
documents** ≈ tens of millions of tokens through the API. That is the one cost that can surprise
you — the box is ~$5, the API might not be. Confirm your plan covers it before step 7.

## 6. Smoke-test one corpus

`nfcorpus` is the smallest (3.6k docs, used whole) and finishes in minutes. **Do not skip it.**

```bash
cd /root/recall && ./.venv/bin/python -m recall.eval.gap_run --beir-root ./beir --out ./results/gap --dsn 'postgresql://recall:recall@127.0.0.1:5432/recall' --datasets nfcorpus
```

✅ **Right if** `cat /root/recall/results/gap/nfcorpus.json` shows `"status": "ok"`, and:
- `scores.local.hybrid` and `scores.cloud.hybrid` are both in (0, 1) **and differ from each other**
- `predictors.oov_rate` is a number, not `NaN`
- `manifest.documents_written` is ~3 633

⚠️ **Local and cloud identical to three decimals ⇒ the cloud embedder was never used.** Fix the key
before burning the night.

## 7. The full run

```bash
cd /root/recall && nohup ./.venv/bin/python -m recall.eval.gap_run --beir-root ./beir --out ./results/gap --dsn 'postgresql://recall:recall@127.0.0.1:5432/recall' > run.log 2>&1 &
```

Expect **4–10 hours** for the remaining 16 corpora (~290k documents, each embedded twice). Fully
resumable: `nfcorpus` is already done and gets skipped; if the box dies, re-run the identical
command and finished corpora are not repeated. A *failed* corpus is deliberately **not** retried —
add `--retry-failed` on purpose, never by reflex.

```bash
tail -f /root/recall/run.log
```

## 8. Analysis

```bash
cd /root/recall && ./.venv/bin/python -c "
import json, glob
from recall.eval.gap_study import analyse_records
records = [json.load(open(p)) for p in glob.glob('results/gap/*.json') if 'summary' not in p]
print(json.dumps(analyse_records(records, arm='hybrid'), indent=2))
" | tee results/gap/analysis.json
```

### Reading it — decided in advance, so nothing here is a choice

| what you see | what it means |
|---|---|
| `underpowered: true` | n < 12. A null means **"could not tell"**, not "no effect" |
| `control_collinearity` near ±1 | the local score absorbs that predictor; its partial correlation is unstable and its p is not to be trusted |
| `haystack_confound.spearman` high | corpus SIZE is moving the gap, not vocabulary — report it as a confound, do not tune it away |
| `holm_p < 0.05` | that predictor beats the null: a rule applicable to a corpus you have never retrieved from |
| no `holm_p < 0.05` | *"skip the vocabulary analysis, just measure your local embedder"* — publishable, in §3's register |
| `responses.gap` vs `responses.headroom` disagree | publish the disagreement; it says which quantity the predictor tracked |
| `predictor_correlations` all high | the three predictors are one measurement, and Holm over three was the wrong correction |

## 9. Bring it home, then destroy the box

```bash
scp -P <PORT> -r root@95.253.220.115:/root/recall/results/gap ~/Documents/recall-vocab-predictor/results/
```

The JSON records are the artifact. Nothing else on the instance matters — destroy it.
