# HERB benchmark — reproducible local-sandbox runbook

Benchmarks nexus retrieval against the HERB benchmark (Salesforce AI Research,
"Benchmarking Deep Search over Heterogeneous Enterprise Data", arXiv 2506.23139)
in a **fully isolated, local-only nexus sandbox** (PG engine, `nx init` local mode,
bge-768) — never touching the operator's live service, config, or any cloud endpoint.

## Isolation model

`benchmark/herb_sandbox.sh` gives every command a virgin HOME under `.herb-sandbox/home`
(gitignored) and an `env -i` allowlist:

| Surface | Live install | HERB sandbox |
|---|---|---|
| `HOME`, `NEXUS_CONFIG_DIR` | the operator's | `.herb-sandbox/home`, `.herb-sandbox/home/.config/nexus` |
| service + Postgres | managed endpoint or the local service | its own cluster and engine, loopback ports the OS allocates |
| embedder | Voyage in cloud mode | `bge-768` = `BAAI/bge-base-en-v1.5` (768d), `NX_LOCAL=1` |
| credentials | present | none: no `NX_SERVICE_*`, `VOYAGE_API_KEY`, `ANTHROPIC_API_KEY` |
| autostart unit | the uid's launchd/systemd domain | never registered (`nx init -y --no-autostart`) |
| telemetry | on | `NX_NO_TELEMETRY=1` |

`up` reads the sandbox service URL back and refuses unless it is loopback. The one thing
the sandbox does not have is a `claude` login, which `nx_answer` needs: set
`HERB_SANDBOX_LLM=1` and wrap the call in the nexus repo's credential helper (below); the
token reaches only that run's environment. Without it indexing can never spend (its hooks
can queue LLM aspect extraction).

## Run it

```bash
SB=./benchmark/herb_sandbox.sh
$SB up                                                   # provision + start (idempotent, ~1 min warm)
$SB run python3 -m benchmark.run_herb --mode prep        # offline: load + flatten + stats
$SB run python3 -m benchmark.run_herb --mode index       # 1559 artifacts through the store note path (~1.5 min)
$SB run python3 -m benchmark.herb_ladder --build-gold    # rebuild gold fixtures from ~/git/HERB
$SB run python3 -m benchmark.herb_ladder                 # the ladder + script/oracle baselines (~2 min, free)
HERB_SANDBOX_LLM=1 python3 <nexus>/tests/e2e/lib/claude_credentials.py run -- \
    $SB run python3 -m benchmark.herb_ladder --nx-answer --cap-usd 2.0   # + the nx_answer product rung (spends)
$SB run python3 -m pytest tests                      # unit + corpus-dependent tests
$SB down                                                 # stop the service and its Postgres
$SB destroy                                              # down, then remove .herb-sandbox
```

The harness python must have `rich` (and `pytest` for tests); the sandbox puts the caller's
`python3` first on PATH. `nx` is the installed one; `NX_BIN` overrides.

- P2 needs a nexus with `nexus.plans.resolve_step` (nexus `develop` until it ships in a release).
- `HERB_NEXUS_REV=<sha> NX_PYTHON=<venv>/bin/python3 NX_BIN=<venv>/bin/nx $SB run python3 -m benchmark.herb_ladder --p2-resolve` scores only P2 (spend 0), writes `results/herb/p2-resolve-<date>.json`, and patches only the P2 rung, `p2_resolve` and `manifest.patched_rungs.P2` of the scorecard (never `step0-baselines.json`).
- P2 refuses the installed nexus interpreter and any env whose `HOME`/`NEXUS_CONFIG_DIR` is not under `.herb-sandbox`: set `NX_PYTHON`/`NX_BIN` to a venv with the nexus under test, and `HERB_NEXUS_REV` to its git revision (recorded beside the wheel version).

## What the ladder measures

| Rung | Metric | Graded against |
|---|---|---|
| P1 | `mentions` link recall / precision | 37 artifact -> CUST pairs (`fixtures/herb/artifact_mentions.json`) |
| P2 | resolution accuracy | the 29 SearchFlow member customers (`p2_member_customers.json`) |
| P3 | hop-1 precision | question -> affected customers (`question_to_affected_customers.json`) |
| E2E script | company recall / precision | HERB `ground_truth`, 7 company questions (the fixed hand-written script, a diagnostic baseline) |
| E2E nx_answer | company recall / precision / F1 | same; **the product rung**. Companies are read from `final_text` against the 10-name registry universe |

The 7 company questions split into 5 "issue" lookups and 2 "maximum number of
resolved/unresolved bugs" aggregations; the summary reports `all7`, `issue5` and `agg2`.
`results/herb/step0-baselines.json` carries every baseline with its RDR-147 re-plan
prediction beside the measured value, the nx and engine versions, and each nx_answer run's
cost. `results/herb/herb-scorecard.json` is the compact rung list.

Every read is `nx search --json` (full chunk text). `-c` truncates each hit at 200
characters; do not use it. Any nx failure raises `NxError`; nothing returns "" on error.

## The RDR-147 MVV (`--mvv`, nexus-bwct4.1)

The MVV run (nexus-irndi) uses this mode; it spends (cap 4 USD total, `MVV_CAP_USD`) and refuses to start unless
the sandbox guard is satisfied (`HERB_SANDBOX=1`, `HOME` and `NEXUS_CONFIG_DIR` under `.herb-sandbox`, no
`NX_SERVICE_*`, `NX_PYTHON` a sandbox venv, `HERB_NEXUS_REV` set) **and the resolution switch is on**:

```bash
NX_ANSWER_RESOLUTION_REPAIR=1 HERB_NEXUS_REV=<sha> HERB_SANDBOX_LLM=1 \
    python3 <nexus>/tests/e2e/lib/claude_credentials.py run -- \
    $SB run python3 -m benchmark.herb_ladder --mvv
python3 -m benchmark.herb_ladder --mvv --resume [DATE]   # after a stop: keeps the valid rows, same cap
python3 -m benchmark.herb_ladder --copy-scorecard-to <nexus>/docs/rdr/evidence/rdr-147   # then commit it in nexus
```

**The switch is environment-only.** nexus reads `NX_ANSWER_RESOLUTION_REPAIR` from the process environment and
never from config, and it is off unless set to an on value (`1`, `true`, `yes`, `on`). The sandbox's `env -i`
allowlist (`herb_sandbox.sh`, checked by `--self-test`) passes it, and `NX_ANSWER_COVERAGE`, through when set.
`--mvv` refuses before the first sweep, call or ledger row unless it is an on value in the harness's own
environment and coverage is not off (`mvv_switch_violations`). The `entity.collections` pattern is the one
thing still in the sandbox's own config; whoever runs the MVV writes it (the harness does not), and the
branch-entered columns and the held-out reading below say when it is missing.

Rules, as the code applies them (`join_completeness`, `precision_gate`, `score_mvv`):

1. **Two runs per question**, labelled `a` and `b`, under one spend cap (`b` gets what `a` left). The cap is
   cumulative across `--resume`: each call is written to the cost ledger (label `mvv`, with `stage` a, b or
   heldout and `mvv_run` the run date) and a resume starts from what the ledger already charged. A call whose
   cost is unknown, or that failed, is charged its per-call budget (0.5 USD): a cap an unreadable cost can slip
   under is no cap.
2. **Grown plans are disabled before every run.** `sweep_grown_plans` snapshots the id of every plan
   (disabled ones too), then soft-disables every active plan that is not a curated template
   (`HttpPlanLibrary.set_plan_disabled`, once each; a template whose declaration says `grown` counts as grown,
   as in nexus) and refuses a list that may be partial. Each row records the id and tags of each plan disabled.
   **Validity (Sam, 2026-11-04):** a run answered by the plan it grew itself is VALID (the inline planner
   saves and reports its own plan; the row says `plan_grown_by_call`), `plan_id` 0 or none is valid, and a run
   is **invalid** only when the reported plan existed in the pre-run snapshot and is a grown or user plan the
   sweep should have disabled (or its row cannot be read). A run made without a plan library is never valid.
3. **A failed call is a row.** An `NxError` from a call (exit, timeout, no envelope) becomes an invalid row
   carrying the error text and the run goes on. Rows are appended to `results/herb/mvv-<date>.jsonl` as they
   land; `--resume` keeps the valid rows (the last row per label and question wins), reruns the rest and
   continues under the cumulative cap. A failed sweep still stops the run (no call starts on a sweep that may
   have missed a grown plan); the rows so far are on disk.
4. **Branch-entered columns.** Each row copies from the envelope's `repair` (the RepairReport,
   `coverage.repair`), tolerantly (an absent field is "not entered" or empty): `resolution_entered`,
   `resolution_reason`, `resolution_tokens`, `anchor_key_ids`, `resolved`, `missing`, `ambiguous`,
   `joined_added`, `items_merged`, the repair `outcome`, `joined_dropped` (`<key>@<item>:<reason>` pairs,
   parsed from the right), `tokens_dropped` (keys the 12-key cap left unresolved), `anchors_scanned`,
   `anchors_without_pattern` and the `*_truncated` counts. Only a boolean `true` counts as entered.
5. **Join completeness (gated).** Domain: issue questions whose `anchor_key_ids` is non-empty in that run.
   A question passes when its answer names every company those anchor ids give in the registry
   (`cust_to_company.json`, case-insensitive) and `resolution_entered` is true. Anchor ids the registry
   cannot map, or that nexus cut on the wire, do not pass. At least 4 of the 5 issue questions must be in
   the domain in each run; fewer is a vacuous run and a fail. A usable question outside the domain says why
   (`domain_exclusion`): empty anchors beside evidence ids read "boundary not reached or branch not
   considered" with the plan and `resolution_reason`. An in-domain question carries a cause per anchor id
   (`id_causes`: resolved, missing, ambiguous, cap-dropped, unaccounted). Reported, never gated: completeness
   over all hydrated evidence ids, and a resolved-only completeness (companies of the resolved ids only), which
   separates cap and missing losses from answer-composition losses.
6. **Precision (gated).** Per run, the mean over the 5 issue questions of per-question precision
   (`score_set`), at least 0.95, in both runs. The pooled ratio and the per-question values are reported.
7. **Three verdicts.** `passed`; `failed` (a run made of valid rows failed a gate: a finding about the branch);
   `invalid` (a run is missing, or a row did not run validly or was skipped: not a measurement, rerun with
   `--resume`, never a finding). A failed run on valid rows settles the gate even if the other run is
   invalid. The scorecard rungs read GREEN, RED or INVALID.
8. **A question passes only where `resolution_entered` is true** (a pass through the semantic repair alone
   does not count).
9. **Held-out single-hop set.** `fixtures/herb/single_hop_heldout.json` is frozen: the first 20 SearchFlow
   answerable questions whose type is not `company`, in file order, with `herb_index` (their position in
   `answerable_questions`; HERB questions carry no id), ground truth and citations, the HERB commit and a
   sha256 of the questions. `load_heldout` refuses an edited file, `--freeze-heldout` never overwrites it.
   Each question runs once behind the same sweep. The count is reported as entered of valid of asked, with the
   skipped and invalid counts and why the others did not enter (`not_entered_reasons`), and is marked **not
   interpretable** when no held-out run was valid or when the branch entered no issue question in either
   run (a mis-set switch or pattern would read the same as a correct zero).
10. **Exact assertions only for deterministic rungs** (P2, P3, the script, both oracles: `PREDICTIONS`).
    nx_answer and aggregation rows are recorded with their run label and never asserted exactly
    (`NX_ANSWER_PREDICTIONS`, `asserted_exactly: false` in the comparison).
11. **Reported, not gated:** issue5 recall, precision and F1 per run beside the script's issue5 F1
    (`0.7642857142857142`, `SCRIPT_ISSUE5_F1`, asserted exactly against the recorded baseline); agg2 recall,
    precision and F1 per run.
12. **Output.** `--mvv` re-measures no other rung. It writes `results/herb/mvv-<date>.json` (rows, held-out
    rows, verdict, score, meta with the nexus revision, versions, plan service URL and spend) beside the
    row file `mvv-<date>.jsonl`, and patches only the MVV rungs, the `mvv` section and
    `manifest.patched_rungs.MVV` into `herb-scorecard.json`. `step0-baselines.json` and the E2E-nx_answer
    rungs are left as the Step 0 run measured them (`tests/test_herb_ladder.py` reads its nx_answer record).

The harness tests (`tests/test_herb_harness.py`, unmarked) pin all of this on fixture rows and fakes: no
interpreter, no sandbox, no service, no model.

## Telemetry

`--mode eval` writes `results/herb/<product>-<arm>-<assetops_sha>.json` with a `manifest`
(UTC time, `nx_version`, `engine_release_version`, embedding model, both repo SHAs, k, arm,
counts), a `summary` (recall@10/50/100, precision@10, hit@10 by type) and `per_question`
rows, plus a HERB-shaped `.pk` for `~/git/HERB/code/evaluate.py`.

## Known limitations (honest)

- **Recall@k is the model-agnostic signal for the `retrieval` arm**, which is recall-only
  (entity F1 ~0). The ladder's company rungs are the entity-level measure.
- **P3 precision is company-level.** The gold counts every member of a gold company as
  affected even when that customer was not named in the issue, so it overstates the
  customer-level figure.
- **Dataset caveats** (see the indexed critique in T3, `knowledge`): GPT-4o circularity on
  `content` questions, ~53 templated question patterns, 8 products with the same question
  text in both answerable/unanswerable pools, real-OSS-PR parametric leakage. Treat
  absolute scores as opaque for cross-system comparison.
- HERB ships two `urls` records under one id (the NLTK repo); the second is stored as
  `<id>#dup2` so catalog identity (collection, title) keeps both.
