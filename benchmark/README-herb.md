# HERB benchmark — reproducible local-sandbox runbook

Benchmarks nexus retrieval against the HERB benchmark (Salesforce AI Research,
"Benchmarking Deep Search over Heterogeneous Enterprise Data", arXiv 2506.23139)
in a **fully isolated, local-only nexus sandbox** — never touching the real
knowledgebase, Chroma Cloud, or Voyage.

## Isolation model

`benchmark/herb_sandbox.sh` wraps any command in an environment where all nexus
state lives under `.herb-sandbox/` (gitignored) and the local 768-dim embedder
is used instead of Voyage:

| Surface | Real KB | HERB sandbox |
|---|---|---|
| config / catalog / T2 / logs / daemon sockets | `~/.config/nexus/` | `.herb-sandbox/config/` (`NEXUS_CONFIG_DIR`) |
| T3 vector store | Chroma Cloud | local ChromaDB `.herb-sandbox/chroma/` (`NX_LOCAL_CHROMA_PATH`, `NX_LOCAL=1`) |
| embedder | Voyage `voyage-context-3` (1024d) | `bge-768` = `BAAI/bge-base-en-v1.5` (768d) |
| cloud creds | present | scrubbed (`CHROMA_API_KEY` etc. unset) |

The wrapper provisions the embedder on first run and ensures the isolated T2/T3
daemons are up. Nothing the benchmark does is visible to the real KB, and a
`nx collection list` (un-wrapped) will never show `herb-*`.

## Run it

```bash
pip install -e ".[dev]"                                              # or: python -m venv .venv && .venv/bin/pip install rich
SB=./benchmark/herb_sandbox.sh

$SB .venv/bin/python -m benchmark.run_herb --mode prep               # offline: load + flatten + stats (free, no index)
$SB .venv/bin/python -m benchmark.run_herb --mode index              # materialize 1462 SearchFlow artifacts -> local bge-768 (~30 min)
$SB .venv/bin/python -m benchmark.run_herb --mode eval               # retrieval recall@10 (cheap) -> results/herb/<product>-retrieval-<sha>.json
$SB .venv/bin/python -m benchmark.run_herb --mode eval --answer-arm nx_answer --limit 5   # real composed arm (slow: ~30s-4min/question)
```

## Telemetry (versioned, for science)

Every `eval` run writes `results/herb/<product>-<arm>-<assetops_sha>.json` (committed)
containing:

- **`manifest`** — the reproducibility envelope: UTC timestamp, `nx_version`,
  `embedding_model`, `chroma_backend` (local/cloud), the sandbox paths, **both
  repo SHAs** (`assetops_sha`, `herb_sha`), product, collection, `k`, answer arm,
  question/artifact counts. A number is never orphaned from the config that made it.
- **`summary`** — `mean_recall_at_k`, `recall_by_type`, `count_by_type`, and total
  retrieval/answer wall-time.
- **`per_question`** — gold citations, retrieved ids, `recall_at_k`,
  `retrieval_ms`, `answer_ms` per question (the raw rows for re-analysis).
- **`herb_dataset`** — also dumped as a sibling `.pk` (gitignored) in HERB's exact
  shape, so `python ~/git/HERB/code/evaluate.py --output_file <file>.pk` scores it
  with HERB's own F1 / Likert / unanswerable metrics.

Additional nexus-side logs land under `.herb-sandbox/config/logs/`
(`t2_daemon.log`, and `nx_answer_step_*` structlog events when the `nx_answer`
arm runs). nexus also records per-run plan metrics in its T2 `nx_answer_runs`
table inside the sandbox.

## Known limitations (honest)

- **Indexing is subprocess-bound.** `--mode index` calls `nx store put` once per
  artifact; each reloads bge-768, so throughput is ~50 artifacts/min regardless of
  local-vs-cloud. A production loader should batch within one process.
- **Recall@k is the model-agnostic signal.** The `retrieval` answer arm is a
  recall-only baseline (entity F1 will be ~0). Use `--answer-arm nx_answer` for a
  real answer, but mind the dataset caveats below.
- **Dataset caveats** (see the indexed critique in T3, `knowledge`): GPT-4o
  circularity on `content` questions, ~53 templated question patterns, 8 products
  with the same question text in both answerable/unanswerable pools, real-OSS-PR
  parametric leakage. Treat absolute scores as opaque for cross-system comparison.
