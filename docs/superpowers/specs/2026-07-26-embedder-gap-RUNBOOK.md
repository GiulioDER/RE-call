# Runbook — the overnight embedder-gap run

Everything here runs on a **rented box**, not VPS2 or VPS3. Every BEIR corpus is public, so nothing
sensitive leaves your infrastructure; the private memory corpus is not part of this run at all (it
is excluded from the primary analysis anyway — see the spec's PREREGISTRATION section).

Each step ends with **how you know it worked**, because several of these fail by producing a
plausible result rather than an error.

---

## 0. Box

Any Linux box with **4+ cores, 8 GB RAM, 20 GB disk**. No GPU needed — `fastembed` runs bge-small
on CPU via ONNX. Expect **4–8 hours** for all 17 corpora.

## 1. Repo and Python

```bash
git clone --branch research/vocab-gap-predictor https://github.com/GiulioDER/RE-call.git ~/recall && cd ~/recall && python3 -m venv .venv && ./.venv/bin/pip install -e '.[fastembed]' voyageai
```

✅ **Right if:** `./.venv/bin/python -c "import recall.eval.gap_run; print('ok')"` prints `ok`.

## 2. Postgres + pgvector

```bash
cd ~/recall && docker compose up -d && sleep 10 && docker compose ps
```

✅ **Right if:** the container is `running` **and** this prints `('vector',)`:

```bash
cd ~/recall && ./.venv/bin/python -c "import psycopg; print(psycopg.connect('postgresql://recall:recall@localhost:5432/recall').execute(\"select extname from pg_extension where extname='vector'\").fetchone())"
```

⚠️ If it prints `None`, pgvector is missing and **every dense arm will silently score near zero** —
stop here, do not run overnight.

## 3. BEIR datasets

CQADupStack ships as one zip whose subforums become separate corpora, so it needs renaming — the
runner looks for `cqadupstack-tex`, the zip gives `cqadupstack/tex`.

```bash
cd ~/recall && mkdir -p beir && cd beir && for d in nfcorpus scifact scidocs fiqa arguana cqadupstack; do curl -sSLO "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/$d.zip" && unzip -q -o "$d.zip" && rm "$d.zip"; done && for f in cqadupstack/*/; do mv "$f" "cqadupstack-$(basename $f)"; done && rmdir cqadupstack
```

✅ **Right if:** this prints **17** and every line ends in `corpus.jsonl`:

```bash
cd ~/recall/beir && ls -d */ | wc -l && ls */corpus.jsonl | head -20
```

## 4. Voyage key

```bash
export VOYAGE_API_KEY='<paste-your-key>'
```

✅ **Right if:** this prints a `1024` (or your model's dim) and **not** a traceback:

```bash
cd ~/recall && ./.venv/bin/python -c "from recall.embeddings import VoyageEmbedder; print(len(VoyageEmbedder().embed(['smoke test'])[0]))"
```

⚠️ Do this **before** the overnight run. A bad key fails on corpus 1 and every corpus after it,
and you will find out in the morning.

## 5. Smoke-test one small corpus first

`nfcorpus` is the smallest (3.6k docs) and finishes in minutes. **Do not skip this.**

```bash
cd ~/recall && ./.venv/bin/python -m recall.eval.gap_run --beir-root ./beir --out ./results/gap --dsn 'postgresql://recall:recall@localhost:5432/recall' --datasets nfcorpus
```

✅ **Right if:** `cat ~/recall/results/gap/nfcorpus.json` shows `"status": "ok"`, both
`scores.local.hybrid` and `scores.cloud.hybrid` are **between 0 and 1 and not equal to each
other**, and `predictors.oov_rate` is not `NaN`.

⚠️ If local and cloud are *identical* to three decimals, the cloud embedder probably was not used —
check the key before burning the night.

## 6. The full run

```bash
cd ~/recall && nohup ./.venv/bin/python -m recall.eval.gap_run --beir-root ./beir --out ./results/gap --dsn 'postgresql://recall:recall@localhost:5432/recall' > run.log 2>&1 &
```

Resumable: `nfcorpus` is already done and will be skipped. If the box dies, re-run the identical
command — finished corpora are not repeated. A failed corpus is **not** retried automatically
(it would burn the night twice); add `--retry-failed` deliberately.

✅ **Watch with:** `tail -f ~/recall/run.log`

## 7. Analysis

```bash
cd ~/recall && ./.venv/bin/python -c "
import json, glob
from recall.eval.gap_study import analyse_records
records = [json.load(open(p)) for p in glob.glob('results/gap/*.json') if 'summary' not in p]
print(json.dumps(analyse_records(records, arm='hybrid'), indent=2))
" | tee results/gap/analysis.json
```

### Reading it — decided in advance, so there is nothing to choose here

| what you see | what it means |
|---|---|
| `underpowered: true` | n fell below 12. A null means **"could not tell"**, not "no effect". Report it that way. |
| any `control_collinearity` near ±1 | the local score absorbs that predictor; its partial correlation is unstable and its p-value is not to be trusted |
| `holm_p < 0.05` for a predictor | it beats the null — a rule applicable to a corpus you have never retrieved from |
| no `holm_p < 0.05` | *"skip the vocabulary analysis, just measure your local embedder"* — publishable, in §3's register |
| `responses.gap` and `responses.headroom` disagree | publish the disagreement; it says which quantity the predictor was tracking |
| `predictor_correlations` all high | the three predictors are one measurement, and Holm over three was the wrong correction |

## 8. Bring the results home

```bash
scp -r '<user>@<box>:~/recall/results/gap' ~/Documents/recall-vocab-predictor/results/
```

Then destroy the box. The JSON records are the artifact; nothing else on it matters.
