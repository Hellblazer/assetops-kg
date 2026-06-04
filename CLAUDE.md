# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A knowledge-graph loader and benchmark harness for IBM's [AssetOpsBench](https://github.com/IBM/AssetOpsBench) industrial-asset scenarios. The thesis: loading the same data into a graph and answering with **deterministic Cypher traversals** (zero LLM tokens) beats IBM's GPT-4 + flat-document baseline (65% → 99%). This repo is the loader + source-data specifics; the graph engine itself is the external `samyama` package (`SamyamaClient.embedded()`), part of the [samyama-graph](https://github.com/samyama-ai/samyama-graph) ecosystem.

There is no application server to "run" — the deliverables are the ETL pipeline, the benchmark runners, and the MCP server. Everything talks to an embedded Samyama graph.

## Commands

```bash
pip install -e ".[dev]"          # install with pytest, ruff
pytest                            # all tests (testpaths=tests)
pytest tests/test_scenarios.py -k graph_dep_001   # single test
ruff check .                      # lint (line-length 100, target py310)

# Benchmarks (each loads data via ETL, then evaluates)
python -m benchmark.run_ibm_scenarios --data-dir ../AssetOpsBench   # IBM's 139 scenarios → 99%
python -m benchmark.run_samyama                                     # 40 synthetic KG scenarios → 100%
python -m benchmark.run_hf_benchmark                               # full 467 HuggingFace configs
python -m benchmark.run_nlq --provider openai --model gpt-4        # LLM-generates-Cypher comparison arm
python -m benchmark.run_baseline                                   # GPT-4 + flat-data baseline (no graph)

python -m demo.demo                                                # narrated walkthrough
```

Most runners take `--category`, `--output results/<name>.json`, and `--data-dir`. The IBM data is NOT vendored — clone `IBM/AssetOpsBench` separately and point `--data-dir` at it (default `~/projects/Madhulatha-Sandeep/AssetOpsBench`).

## Architecture

Three layers, all over an embedded Samyama graph:

**`etl/`** — ETL into the graph. `loader.py` is the click-CLI orchestrator for synthetic data (EAMLite → CouchDB sensors → FMSR failure modes → work orders → embeddings, in that dependency order). `ibm_loader.py:load_ibm_data` loads the real IBM dataset (site/equipment hierarchy, sensors, failure modes, MONITORS edges, work-order/alert/anomaly CSVs). The `EQUIPMENT_MAP` (asset-tag → human name) in `ibm_loader.py` is the canonical IBM-data translation. `embedding_gen.py` produces 384-dim `all-MiniLM-L6-v2` embeddings stored on `FailureMode` nodes for vector similarity.

**`benchmark/`** — evaluation harnesses, one per comparison arm (see commands above). `handlers/router.py:route_scenario` dispatches the 467 HuggingFace scenarios to `rule_logic_handler` / `fmsr_handler` / `phm_handler` via `_CONFIG_MAP` / `_TYPE_MAP` keyed on scenario config/type strings. `rubric_judge.py` re-scores NLQ trajectories on the AssetOpsBench 3-axis LLM-as-judge rubric (task completion / retrieval accuracy / result verification), running each 3× for variance.

**`mcp_server/`** — `server.py` (FastMCP) exposes graph queries as MCP tools, registered in four groups under `tools/`: asset, failure, impact, analytics. Holds one global embedded client on the `industrial` graph.

**`scenarios/`** — 40 hand-authored synthetic scenarios (7 category files) used by `run_samyama`. Each scenario: `id`, `category`, `description`, `expected_tools`, `expected_output_contains`, `difficulty`, `requires_graph`. `tests/test_scenarios.py` enforces structure, unique IDs, and per-category counts — **if you add/remove scenarios, update `EXPECTED_COUNTS` in that test.**

**`schema/industrial_kg.cypher`** — the documented graph schema (ISO 14224 + ISA-95; 14 node labels, 21 edge types in the v2 467-scenario expansion). Reference, not executed by the loaders.

## Graph names

Different entry points use different graph names against the same embedded engine: `industrial` (synthetic ETL + MCP server), `ibm` (`run_ibm_scenarios`). Don't assume one global graph.

## Results

`results/*.json` are versioned (committed, `_v2`…`_v5` suffixes track reruns). Raw per-run log streams (`results/arxiv-runs/`) and the `data/` directory are gitignored. The published numbers and methodology live in `docs/` (`results.md`, `methodology.md`, `reproducibility-guide.md`, `information-leakage-analysis.md`).

## Conventions

Apache-2.0 (see LICENSE). `from __future__ import annotations` + `logging` (not `print`) in library modules; benchmark/demo CLIs use `rich.Console` for user-facing output. `click` for the ETL CLI, `argparse` for benchmark runners.
