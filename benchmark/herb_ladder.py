"""RDR-147 benchmark ladder — per-phase observable metrics over HERB SearchFlow.

The single end-to-end metric (company recall) only moves when all three RDR-147
phases land, so it gives no intermediate signal. This module decomposes it into a
ladder of isolated sub-metrics and emits ``herb-scorecard.json`` plus
``step0-baselines.json`` (the Step 0 baselines with their predictions beside them).

    Phase             metric                         graded against
    ----------------  -----------------------------  ------------------------------------
    P1 (linker)       mentions-link recall/precision 37 artifact -> CUST pairs (Slack)
    P2 (resolve)      resolution accuracy            the 29 SearchFlow member customers
    P3 (filter)       hop-1 precision                question -> affected customers
    E2E script        company recall / precision     HERB ground_truth (diagnostic rung)
    E2E nx_answer     company recall / precision     HERB ground_truth (THE product rung)

The E2E product rung is nx_answer. The hand-written script
(``multihop_resolve.answer_company``) is a separate baseline row: a script that needs no
nexus change can move a metric, so it can never stand in for the product (RDR-147
re-plan 2026-11-02, finding 2).

Every deterministic metric (P2, P3, the script, both oracles) is asserted EXACT by the paired
pytest. nx_answer values vary from run to run: they are recorded with their run label and compared
to a gate or a prediction, never asserted exactly. Until the corresponding nexus feature exists the
metric computes to its RED baseline; the harness is the oracle, not the implementation.

The RDR-147 MVV (``--mvv``, nexus-bwct4.1) runs every issue question twice (labels a and b) with
every grown plan soft-disabled before each run, then scores join completeness and precision (verdict
passed, failed or invalid), and counts how often the resolution branch fires on a frozen single-hop
held-out set. It writes ``results/herb/mvv-<date>.json`` and patches only the MVV rungs into the scorecard
(``emit_mvv``); the Step 0 baselines are never rewritten by it. The rules are in
``benchmark/README-herb.md`` (section "The RDR-147 MVV").

Run inside the isolated sandbox (see herb_sandbox.sh):
    ./benchmark/herb_sandbox.sh run python3 -m benchmark.herb_ladder --build-gold
    ./benchmark/herb_sandbox.sh run python3 -m benchmark.herb_ladder
    # with the nx_answer product rung (spends money; needs the claude CLI token):
    HERB_SANDBOX_LLM=1 python3 <nexus>/tests/e2e/lib/claude_credentials.py run -- \\
        ./benchmark/herb_sandbox.sh run python3 -m benchmark.herb_ladder --nx-answer --cap-usd 2.0
    # the MVV: two labelled runs per question + the held-out set (spends, cap 4 USD total; refuses unless
    # NX_ANSWER_RESOLUTION_REPAIR is on, which nexus reads from the environment only). Rows go to
    # results/herb/mvv-<date>.jsonl as they land; --resume [DATE] continues a stopped run:
    NX_ANSWER_RESOLUTION_REPAIR=1 HERB_NEXUS_REV=<sha> HERB_SANDBOX_LLM=1 \\
        python3 <nexus>/tests/e2e/lib/claude_credentials.py run -- \\
        ./benchmark/herb_sandbox.sh run python3 -m benchmark.herb_ladder --mvv
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import pwd
import re
import shutil
import subprocess
import textwrap
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from rich.console import Console

from benchmark.multihop_resolve import (  # type: ignore
    CORPUS,
    HERB_DIR,
    _RE_CUST,
    _RE_COMPANY,
    SEARCH_TIMEOUT_S,
    NxError,
    answer_company,
    companies_for_ids,
    resolve_company,
    search_hits,
)
from benchmark.run_herb import (  # type: ignore
    _NXA_SENTINEL,
    CONEXUS_PYTHON,
    corpus_arg,
    flatten_artifacts,
    nx_answer_call,
    service_versions,
)

console = Console()
err_console = Console(stderr=True)

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "herb"
_GOLD_CUST_COMPANY = _FIXTURE_DIR / "cust_to_company.json"
_GOLD_MEMBERS = _FIXTURE_DIR / "p2_member_customers.json"
_GOLD_MENTIONS = _FIXTURE_DIR / "artifact_mentions.json"
_GOLD_Q_CUSTOMERS = _FIXTURE_DIR / "question_to_affected_customers.json"
_GOLD_HELDOUT = _FIXTURE_DIR / "single_hop_heldout.json"
_RESULTS_DIR = Path(__file__).parent.parent / "results" / "herb"
_SCORECARD = _RESULTS_DIR / "herb-scorecard.json"
_BASELINES = _RESULTS_DIR / "step0-baselines.json"

PRODUCT = "SearchFlow"

# The E2E baseline the Problem Statement declared broken (single-hop nx_answer, June).
_E2E_BASELINE = 0.000

# Predictions written down BEFORE the first measured run (RDR-147 re-plan memo
# plan-rdr147-replan-2026-11-02 section 4, k=20, bge-768). op "==" is a point estimate,
# "<" an upper bound. Embedded beside the measured value in step0-baselines.json.
#
# PREDICTIONS is the exact-assert table: deterministic rungs only (P2, P3, the script, both oracles).
# nx_answer values vary per run (an LLM drives the plan), so they live in NX_ANSWER_PREDICTIONS: recorded
# beside the measurement, never asserted exactly (nexus-bwct4.1, RDR-147 fix check).
PREDICTIONS: dict[str, dict[str, Any]] = {
    "script_fixed.recall.all7": {"value": 0.40, "op": "=="},
    "script_fixed.recall.issue5": {"value": 0.52, "op": "=="},
    "script_fixed.recall.agg2": {"value": 0.10, "op": "=="},
    "p3_hop1_precision": {"value": 0.55, "op": "=="},
    "p2_resolution_accuracy_29": {"value": 1.0, "op": "=="},
    "oracle_cited.recall.issue5": {"value": 1.0, "op": "=="},
    "oracle_cited.precision.issue5": {"value": 0.61, "op": "=="},
    "oracle_cited.precision.agg2": {"value": 0.10, "op": "=="},
}
NX_ANSWER_PREDICTIONS: dict[str, dict[str, Any]] = {
    "nx_answer.recall.all7": {"value": 0.10, "op": "=="},
    "nx_answer.precision.all7": {"value": 0.30, "op": "<"},
}

# The MVV rules (RDR-147 section Minimum Viable Validation; Sam's readings 2026-11-03, nexus-bwct4.1).
RUN_LABELS = ("a", "b")          # every issue question runs twice
ISSUE_QUESTIONS = 5              # the issue questions the gates are taken over
DOMAIN_FLOOR = 4                 # join completeness: at least this many issue questions in the domain, per run
MAX_DOMAIN_MISSES = 1            # join completeness: at most this many in-domain questions may fail, per run
                                 # (Sam 2026-11-05, T2 nexus/rdr147-mvv-bar-revision-2026-11-05: assessor variance)
PRECISION_GATE = 0.95            # per-run mean of per-question precision over the questions that named a company
HELDOUT_LIMIT = 20               # the single-hop held-out set: the first 20 non-company SearchFlow questions
MVV_CAP_USD = 4.0                # spend cap of the MVV run
# The script's issue5 F1 (step0-baselines.json): the diagnostic baseline the nx_answer F1 is reported beside.
# A deterministic rung, asserted exactly by the paired pytest.
SCRIPT_ISSUE5_F1 = 0.7642857142857142
GROWN_PLAN_REASON = "herb-mvv: a plan grown from one question captures later ones"
PLAN_SWEEP_LIMIT = 1000          # plans listed per sweep; a list that fills this is refused as possibly partial
#: The RepairReport fields copied into every nx_answer row (nexus-qpimy, with its fix round 1). Read
#: tolerantly: a report from an older nexus lacks the later ones. ``repair_outcome`` is the report's
#: ``outcome``; the ``*_truncated`` counts say how many entries nexus cut from a key list on the wire.
REPAIR_FIELDS = ("resolution_entered", "resolution_reason", "resolution_tokens", "anchor_key_ids", "resolved",
                 "missing", "ambiguous", "joined_added", "items_merged", "repair_outcome", "joined_dropped",
                 "tokens_dropped", "anchors_scanned", "anchors_without_pattern", "anchor_key_ids_truncated",
                 "resolved_truncated", "missing_truncated", "ambiguous_truncated", "joined_dropped_truncated",
                 "tokens_dropped_truncated", "generated_on_join")
_TRUNCATED_LISTS = ("anchor_key_ids", "resolved", "missing", "ambiguous", "joined_dropped", "tokens_dropped")
MVV_LEDGER_LABEL = "mvv"         # every MVV call's cost-ledger label (nexus-irndi); the stage is beside it
PER_CALL_BUDGET_USD = 0.5        # nx_answer's soft per-call budget; also what a call of unknown cost is charged
MIN_CALL_USD = 0.02              # below this much left under the cap, no call starts
RESULT_VERDICTS = ("passed", "failed", "invalid")   # invalid: not a measurement (see score_mvv)
ON_VALUES = frozenset({"1", "true", "yes", "on"})   # nexus.plans.switches
OFF_VALUES = frozenset({"0", "false", "off", "no"})
RESOLUTION_SWITCH = "NX_ANSWER_RESOLUTION_REPAIR"
COVERAGE_SWITCH = "NX_ANSWER_COVERAGE"

DEFAULT_CAP_USD = 2.0

#: The head of a ``final_text`` that says nx_answer stopped on an error: a degraded result, not an answer. nexus
#: returns these IN BAND (the call exits 0 and prints an envelope), so without this check a budget-exhausted or
#: errored call after spend reads as a valid zero-precision measurement of the branch. Sources, nexus 8096a1451:
#: ``[non-answer: <shape>]`` with shape ``error`` (plan-match, plan-execution, single-step and input errors;
#: core.py ``_error_result``, :13464-13468) or ``planner_error`` (core.py:13853-13857); and the budget-exhausted
#: marker ``[budget exhausted (time|cost) after step N of M ...]`` (core.py:12538, built at :13520). The coverage
#: non-answers (``[non-answer: uncovered]``, ``[non-answer: no_evidence]``, answer_shape.py:295-312) and a partial
#: answer (``[insufficient-evidence: ...]``) are findings about what the branch retrieved, so they stay real rows.
_IN_BAND_ERROR_RE = re.compile(r"\s*(?:\[non-answer: (?:error|planner_error)\]|\[budget exhausted)")
IN_BAND_ERROR_NOTICE_CHARS = 240   # how much of the notice a row's invalid_reason carries
_RUN_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


# --------------------------------------------------------------------------- #
# HERB source access
# --------------------------------------------------------------------------- #


def _load_product() -> dict[str, Any]:
    return json.loads((HERB_DIR / "data/products" / f"{PRODUCT}.json").read_text())


def _is_aggregation(question: str) -> bool:
    """The two 'maximum number of resolved/unresolved bugs' questions are aggregations over
    the whole bug set, not lookups; the extract+resolve template cannot answer them."""
    return "maximum number" in question.lower()


def company_questions(product: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for q in product["answerable_questions"]:
        if q.get("type") != "company":
            continue
        out.append({
            "question": q["question"],
            "gold": sorted(q["ground_truth"]),
            "citations": sorted(set(q.get("citations", []))),
            "kind": "aggregation" if _is_aggregation(q["question"]) else "issue",
        })
    return out


def held_out_questions(product: dict[str, Any], limit: int = HELDOUT_LIMIT) -> list[dict[str, Any]]:
    """The single-hop held-out set: the answerable questions whose type is not ``company``, in file
    order, the first *limit* of them. HERB gives questions no id, so each carries ``herb_index``, its
    position in the product's ``answerable_questions``."""
    out = []
    for i, q in enumerate(product["answerable_questions"]):
        if q.get("type") == "company":
            continue
        out.append({"herb_index": i, "question": q["question"], "type": q.get("type"),
                    "ground_truth": q["ground_truth"], "citations": q.get("citations", [])})
        if len(out) == limit:
            break
    return out


def _heldout_digest(questions: list[dict[str, Any]]) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(questions, sort_keys=True).encode()).hexdigest()


def freeze_heldout(product: dict[str, Any] | None = None, path: Path | None = None) -> dict[str, Any]:
    """Write the held-out fixture once. It is frozen: an existing file is never rewritten (``NxError``);
    change the set only by a decision, by deleting the file and the digest pinned in the tests."""
    path = path or _GOLD_HELDOUT
    if path.exists():
        raise NxError(f"{path} exists: the held-out set is frozen")
    product = product or _load_product()
    qs = held_out_questions(product)
    try:
        commit = subprocess.run(["git", "-C", str(HERB_DIR), "rev-parse", "HEAD"], capture_output=True,
                                timeout=30, check=True).stdout.decode().strip()
    except Exception:  # noqa: BLE001 - provenance only; a source that is not a checkout records none
        commit = ""
    fx = {"product": PRODUCT, "source": f"data/products/{PRODUCT}.json", "herb_commit": commit,
          "selection": f"answerable_questions whose type is not 'company', in file order, first {HELDOUT_LIMIT}",
          "n_non_company_in_source": sum(1 for q in product["answerable_questions"] if q.get("type") != "company"),
          "sha256": _heldout_digest(qs), "questions": qs}
    path.write_text(json.dumps(fx, indent=2) + "\n")
    return fx


def load_heldout(path: Path | None = None) -> list[dict[str, Any]]:
    """The frozen held-out questions. ``NxError`` when the file is missing, or its digest does not match
    its questions (an edited fixture is not the frozen set)."""
    path = path or _GOLD_HELDOUT
    if not path.exists():
        raise NxError(f"{path} is missing: the frozen held-out set is committed with the harness")
    fx = json.loads(path.read_text())
    if _heldout_digest(fx["questions"]) != fx.get("sha256"):
        raise NxError(f"{path}: the questions do not match the stored digest; the held-out set is frozen")
    return fx["questions"]


# --------------------------------------------------------------------------- #
# Gold-truth fixtures — built once, deterministically, from the HERB source.
# --------------------------------------------------------------------------- #


def build_gold_fixtures() -> dict[str, Any]:
    """Derive the gold fixtures from the HERB source.

    cust_to_company (120): the GLOBAL customer registry ``data/metadata/customers_data.json``,
    the authoritative CUST->company truth. Used to invert company gold into affected customers.

    p2_member_customers (29): the SearchFlow member customers (``product["customers"]``) with
    their company. P2 resolution is graded over these 29, the only customer records the corpus
    holds. (The June scorecard divided 29 resolutions by the 120-row registry and printed
    0.2417 for a perfect resolver.) Every gold company of all 7 company questions is reachable
    through a member, so scoping to members loses no question.

    artifact_mentions (37): artifact id -> CUST ids its own text names, over the indexed
    artifacts (slack, documents, transcripts, urls, prs). All 37 pairs are Slack messages;
    the customer and employee records are entities, not artifacts. This is the P1 gold: the
    linker's job is an artifact--mentions-->customer edge, not CUST->company.

    question_to_affected_customers: each company question's ``ground_truth`` company names
    inverted against the GLOBAL registry (affected sets 13..81; product-scoped inversion gives
    empty sets for some questions and makes P3 degenerate). Note it counts every member of a
    gold company as affected even when that customer was not named in the issue, so P3
    precision is company-level and overstates the customer-level figure.
    """
    registry = json.loads((HERB_DIR / "data/metadata/customers_data.json").read_text())
    cust_to_company: dict[str, str] = {
        c["id"]: c["company"] for c in registry if c.get("id") and c.get("company")
    }
    product = _load_product()

    members = {cid: cust_to_company[cid] for cid in product["customers"] if cid in cust_to_company}

    mentions: dict[str, list[str]] = {}
    for rec in flatten_artifacts(product):
        ids = sorted(set(_RE_CUST.findall(rec.text)))
        if ids:
            mentions.setdefault(rec.id, [])
            mentions[rec.id] = sorted(set(mentions[rec.id]) | set(ids))

    company_to_custs: dict[str, list[str]] = {}
    for cid, comp in cust_to_company.items():
        company_to_custs.setdefault(comp.lower(), []).append(cid)
    q_to_custs: dict[str, list[str]] = {}
    for q in company_questions(product):
        gold_companies = {g.lower() for g in q["gold"]}
        q_to_custs[q["question"]] = sorted(
            {cid for comp in gold_companies for cid in company_to_custs.get(comp, [])}
        )

    _FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for path, obj in (
        (_GOLD_CUST_COMPANY, cust_to_company),
        (_GOLD_MEMBERS, members),
        (_GOLD_MENTIONS, mentions),
        (_GOLD_Q_CUSTOMERS, q_to_custs),
    ):
        path.write_text(json.dumps(obj, indent=2, sort_keys=True))
    console.print(
        f"[green]gold fixtures written[/green]: {len(cust_to_company)} cust->company, "
        f"{len(members)} members, {len(mentions)} artifact mentions "
        f"({sum(len(v) for v in mentions.values())} pairs), {len(q_to_custs)} questions"
    )
    return {"cust_to_company": cust_to_company, "members": members,
            "mentions": mentions, "q_to_custs": q_to_custs}


def _load_gold() -> dict[str, Any]:
    paths = (_GOLD_CUST_COMPANY, _GOLD_MEMBERS, _GOLD_MENTIONS, _GOLD_Q_CUSTOMERS)
    if not all(p.exists() for p in paths):
        return build_gold_fixtures()
    c, m, a, q = (json.loads(p.read_text()) for p in paths)
    return {"cust_to_company": c, "members": m, "mentions": a, "q_to_custs": q}


# --------------------------------------------------------------------------- #
# Scoring primitives
# --------------------------------------------------------------------------- #


def score_set(pred: set[str], gold: set[str]) -> dict[str, float | int]:
    """Recall, precision and F1 of a set of company names. Names compare case-insensitively.
    An empty prediction scores precision 0.0 (not undefined), so an answer that names no company
    cannot look better than one that names wrong ones."""
    p = {x.lower() for x in pred}
    g = {x.lower() for x in gold}
    hit = len(p & g)
    recall = hit / len(g) if g else 0.0
    precision = hit / len(p) if p else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"recall": recall, "precision": precision, "f1": f1, "hit": hit,
            "n_pred": len(p), "n_gold": len(g)}


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    """Macro-average recall/precision/f1 over all 7 questions, the 5 issue questions and the 2
    aggregation questions (reported separately)."""
    groups = {
        "all7": rows,
        "issue5": [r for r in rows if r["kind"] == "issue"],
        "agg2": [r for r in rows if r["kind"] == "aggregation"],
    }
    out: dict[str, dict[str, float | int]] = {}
    for name, rs in groups.items():
        n = len(rs)
        out[name] = {"n": n} | {
            m: (sum(r[m] for r in rs) / n if n else 0.0) for m in ("recall", "precision", "f1")
        }
    return out


def extract_companies(text: str, universe: list[str]) -> set[str]:
    """Company names from the registry that appear in an answer's prose (whole word, any case)."""
    found = set()
    for name in universe:
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])", text or "", re.IGNORECASE):
            found.add(name)
    return found


# --------------------------------------------------------------------------- #
# Per-phase metrics
# --------------------------------------------------------------------------- #


def score_p1_link_recall_precision(
    artifact_mentions: dict[str, list[str]],
) -> tuple[float | None, float | None]:
    """P1: does the entity-linker create artifact--mentions-->customer edges?

    Gold is the 37 artifact->CUST pairs. RED (None) until the entity-linker hook (RDR-147
    Step 2) exists and the corpus is re-indexed with it enabled; when it lands:
    recall = gold pairs with a ``mentions`` link from the artifact's catalog document to the
    customer record's document / 37; precision = correct ``mentions`` edges / edges created.
    """
    return None, None


# The P2 rung resolves the member customers through nexus's `resolve` step (RDR-147 Step 3,
# nexus-v9v5r): ONE `resolve_keys` call over every token, in the conexus interpreter, then one
# batched `store_get_many` for the resolved records' text. It needs a nexus that has
# `nexus.plans.resolve_step` (nexus develop until it ships in a release).
#
# Two guards keep it off a live tenant. `p2_sandbox_violations` runs in the host before any spawn and
# again in the driver from the driver's own environment, before any client exists; the driver then
# refuses a non-loopback service URL before any request.


def p2_sandbox_violations(env: Mapping[str, str], interpreter: Any, real_home: Any) -> list[str]:
    """Reasons P2 must not reach a nexus service from *env* with *interpreter*; empty means it may.

    Stdlib only and self-contained: the driver embeds this function's source and runs it itself.
    HERB_SANDBOX=1 alone is not enough (it can be exported by hand): NEXUS_CONFIG_DIR and HOME must
    both lie under a ``.herb-sandbox`` directory (symlinks and ``..`` resolved), the interpreter must
    not be the one installed under *real_home*'s nexus tools directory (it would read the operator's
    own service config), no NX_SERVICE_* endpoint or token may be set (nexus reads them before
    the config dir, so they would reach whatever service they name, a loopback live one included),
    and HERB_NEXUS_REV must name the nexus revision being measured."""
    import os
    import re

    bad: list[str] = []
    if env.get("HERB_SANDBOX") != "1":
        bad.append("HERB_SANDBOX is not 1")
    for key in ("NEXUS_CONFIG_DIR", "HOME"):
        val = env.get(key)
        if not val:
            bad.append(f"{key} is not set")
        elif ".herb-sandbox" not in os.path.realpath(val).split(os.sep):
            bad.append(f"{key}={val} does not lie under a .herb-sandbox directory")
    tools = os.path.realpath(os.path.join(str(real_home), ".local", "share", "nexus", "tools"))
    interp_dir = os.path.realpath(os.path.dirname(os.path.abspath(str(interpreter))))
    if interp_dir == tools or interp_dir.startswith(tools + os.sep):
        bad.append(f"interpreter {interpreter} is the installed nexus ({tools}); set NX_PYTHON to a sandbox venv")
    for key in sorted(k for k in env if k.startswith("NX_SERVICE_")):
        bad.append(f"{key} is set; the sandbox's own config names its service")
    if not re.fullmatch(r"[0-9a-f]{7,40}", env.get("HERB_NEXUS_REV", "")):
        bad.append("HERB_NEXUS_REV is not set to the nexus git revision under test (hex, 7-40 chars)")
    return bad


def _real_home() -> str:
    """The operator's home from the password database, not from $HOME (which the sandbox swaps)."""
    return pwd.getpwuid(os.getuid()).pw_dir


_RESOLVE_DRIVER = textwrap.dedent(
    """
    from __future__ import annotations
    import json, os, pwd, sys, time
    from urllib.parse import urlparse

    @@GUARD@@

    bad = p2_sandbox_violations(os.environ, sys.executable, pwd.getpwuid(os.getuid()).pw_dir)
    if bad:
        raise SystemExit("resolve driver refuses to run: " + "; ".join(bad))

    import importlib.metadata
    from nexus.catalog.http_catalog_client import HttpCatalogClient
    from nexus.mcp.core import store_get_many
    from nexus.plans.resolve_step import resolve_keys

    class Spy:
        def __init__(self, inner):
            self.inner, self.lookups = inner, 0
        def find_by_title_exact(self, *a, **kw):
            self.lookups += 1
            return self.inner.find_by_title_exact(*a, **kw)
        def get_manifests(self, *a, **kw):
            return self.inner.get_manifests(*a, **kw)

    resolve_invocations = []

    def counted_resolve_keys(catalog, tokens):
        resolve_invocations.append(len(tokens))
        return resolve_keys(catalog, tokens)

    tokens = json.load(open(sys.argv[1]))
    cat = HttpCatalogClient()
    url = str(cat._base_url)
    print("resolve service url: " + url, file=sys.stderr, flush=True)
    if urlparse(url).hostname not in ("127.0.0.1", "localhost", "::1"):
        raise SystemExit("refusing non-loopback service url " + url)
    spy = Spy(cat)
    t0 = time.perf_counter()
    res = counted_resolve_keys(spy, tokens)
    resolve_s = time.perf_counter() - t0
    flat_ids, flat_colls, owner = [], [], []
    for tok, bt in res["by_token"].items():
        for i, c in zip(bt["ids"], bt["collections"]):
            flat_ids.append(i); flat_colls.append(c); owner.append(tok)
    texts = {tok: "" for tok in res["by_token"]}
    if flat_ids:
        got = store_get_many(flat_ids, flat_colls, max_chars_per_doc=100000, structured=True)
        for tok, body in zip(owner, got["contents"]):
            texts[tok] += (body or "") + "\\n"
    print("@@SENTINEL@@" + json.dumps({
        "service_url": url, "result": res, "texts": texts,
        "lookups": spy.lookups, "resolve_calls": len(resolve_invocations), "resolve_s": resolve_s,
        "nexus_revision": os.environ["HERB_NEXUS_REV"],
        "nexus_dist_version": importlib.metadata.version("conexus"),
    }))
    """
).replace("@@GUARD@@", textwrap.dedent(inspect.getsource(p2_sandbox_violations))).replace("@@SENTINEL@@", _NXA_SENTINEL)


def resolve_p2_tokens(tokens: list[str]) -> dict[str, Any]:
    """ONE ``resolve_keys`` call over *tokens* against the sandbox catalog, via the conexus interpreter.

    Refuses (``NxError``) before spawning anything unless ``p2_sandbox_violations`` is empty.
    Returns ``{service_url, result, texts, lookups, resolve_calls, resolve_s, nexus_revision,
    nexus_dist_version}``: ``result`` is the step's own output (``tumblers, ids, collections, by_token,
    missing, ambiguous``), ``texts`` the joined chunk text of each resolved token's record,
    ``resolve_calls`` the number of ``resolve_keys`` invocations the driver counted. Raises ``NxError``
    on a subprocess failure or a missing result line."""
    import tempfile

    bad = p2_sandbox_violations(os.environ, CONEXUS_PYTHON, _real_home())
    if bad:
        raise NxError("resolve_p2_tokens refuses to reach a nexus service: " + "; ".join(bad)
                      + ". Run it through benchmark/herb_sandbox.sh with NX_PYTHON/NX_BIN set to a sandbox "
                        "venv and HERB_NEXUS_REV set; never against the operator's own or the managed service")
    if not CONEXUS_PYTHON.exists():
        raise NxError(f"nx interpreter not found at {CONEXUS_PYTHON}; set NX_PYTHON")
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(tokens, tf)
        tmp = tf.name
    try:
        proc = subprocess.run(
            [str(CONEXUS_PYTHON), "-c", _RESOLVE_DRIVER, tmp],
            capture_output=True, timeout=SEARCH_TIMEOUT_S, check=True,
        )
    except subprocess.CalledProcessError as e:
        raise NxError(f"resolve driver failed (exit {e.returncode}): {(e.stderr or b'').decode(errors='replace')[-600:]}") from e
    except subprocess.TimeoutExpired as e:
        raise NxError(f"resolve driver timed out after {SEARCH_TIMEOUT_S}s") from e
    finally:
        os.unlink(tmp)
    for line in proc.stdout.decode(errors="replace").splitlines():
        if line.startswith(_NXA_SENTINEL):
            return json.loads(line[len(_NXA_SENTINEL):])
    raise NxError("resolve driver printed no result line")


def score_p2_resolution(
    members: dict[str, str],
    resolver: Callable[[list[str]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """P2: given a CORRECT CUST id, does the ``resolve`` step return the right record?

    Hop 1 is bypassed (gold ids fed directly). Every member id goes to *resolver* in ONE call (the
    nexus ``resolve_keys`` core). A token is right when it resolved (a ``by_token`` entry, not in
    ``missing``/``ambiguous``) and its record text opens with ``Customer {cid}:`` and names the gold
    company. That correctness check is HERB-scoped: it checks the ingest format of
    ``run_herb.flatten_artifacts`` (``Customer {cid}: contact ..., at company {company}.``), not a
    general resolve contract. P2 passes only at accuracy 1.0 with ``missing`` and ``ambiguous`` both
    empty. Records the token count, the driver-counted resolve calls, catalog lookups, wall time and
    the nexus revision and wheel version the driver ran. An error from nx raises."""
    tokens = list(members)
    if not tokens:
        return {"accuracy": 0.0, "passed": False, "n_tokens": 0, "hits": 0, "wrong": [],
                "missing": [], "ambiguous": {}, "resolve_calls": 0,
                "nexus_revision": None, "nexus_dist_version": None}
    t0 = time.perf_counter()
    out = (resolver or resolve_p2_tokens)(tokens)
    wall_s = time.perf_counter() - t0
    res, texts = out["result"], out.get("texts", {})
    missing, ambiguous = list(res.get("missing", [])), dict(res.get("ambiguous", {}))
    resolved = res.get("by_token", {})
    hits, wrong = 0, []
    for cid, gold_company in members.items():
        text = texts.get(cid, "")
        own_record = re.search(rf"(?m)^\s*Customer {re.escape(cid)}:", text)
        m = _RE_COMPANY.search(text)
        pred = m.group(1).strip() if m else None
        if cid in resolved and own_record and pred and pred.lower() == gold_company.strip().lower():
            hits += 1
        else:
            wrong.append(cid)
    accuracy = hits / len(members)
    return {
        "accuracy": accuracy,
        "passed": accuracy == 1.0 and not missing and not ambiguous,
        "n_tokens": len(tokens),
        "n_distinct_tokens": len(set(tokens)),
        "resolve_calls": out["resolve_calls"],
        "catalog_lookups": out.get("lookups"),
        "hits": hits,
        "wrong": wrong,
        "missing": missing,
        "ambiguous": ambiguous,
        "service_url": out.get("service_url"),
        "resolve_s": out.get("resolve_s"),
        "wall_s": wall_s,
        "nexus_revision": out["nexus_revision"],
        "nexus_dist_version": out["nexus_dist_version"],
    }


def score_p2_resolution_accuracy(members: dict[str, str]) -> float:
    """P2 accuracy through the ``resolve`` step (see ``score_p2_resolution`` for the full record)."""
    return score_p2_resolution(members)["accuracy"]


def p2_rung(detail: dict[str, Any]) -> "Rung":
    """The P2 scorecard row: GREEN only when ``detail['passed']`` (accuracy 1.0, nothing missing or ambiguous)."""
    return Rung("P2", "resolution_accuracy_29_members", detail["accuracy"], "==1.0, missing=[] ambiguous={}",
                "GREEN" if detail["passed"] else "RED")


def p2_resolve_entry(detail: dict[str, Any], versions: dict[str, str], timestamp_utc: str,
                     file: str | None) -> dict[str, Any]:
    """The scorecard's ``p2_resolve`` entry: the same shape on the full run (``file`` None, no per-run
    file is written) and on ``--p2-resolve``. Carries the nx/engine versions and the nexus revision."""
    return {**detail, **versions, "timestamp_utc": timestamp_utc, "file": file}


def patch_scorecard_p2(card: dict[str, Any], detail: dict[str, Any], versions: dict[str, str],
                       timestamp_utc: str, file: str) -> dict[str, Any]:
    """Replace the P2 rung and ``p2_resolve`` of *card* in place and mark the manifest.

    Raises ``ValueError`` unless the card holds exactly one P2 rung. The card's own manifest versions
    describe the other rungs and stay as measured; ``manifest.patched_rungs.P2`` names what was
    replaced and the versions and nexus revision it was measured on."""
    idx = [i for i, r in enumerate(card["rungs"]) if r["phase"] == "P2"]
    if len(idx) != 1:
        raise ValueError(f"expected exactly one P2 rung in the scorecard, found {len(idx)}")
    rung = p2_rung(detail)
    card["rungs"][idx[0]] = asdict(rung)
    card["p2_resolve"] = p2_resolve_entry(detail, versions, timestamp_utc, file)
    card["manifest"].setdefault("patched_rungs", {})["P2"] = {
        "metric": rung.metric, "by": "--p2-resolve", "file": file, "timestamp_utc": timestamp_utc,
        "nexus_revision": detail["nexus_revision"], "nexus_dist_version": detail["nexus_dist_version"],
        "nx_version": versions.get("nx_version"), "engine_release_version": versions.get("engine_release_version"),
    }
    return card


def score_p3_detail(q_to_custs: dict[str, list[str]], k: int = 20) -> list[dict[str, Any]]:
    """P3 per question: of the CUST ids hop 1 (full chunk text) surfaced, what fraction are affected."""
    rows = []
    for question, gold in q_to_custs.items():
        gold_set = set(gold)
        hits = search_hits(question, k)
        surfaced = {cid for h in hits for cid in _RE_CUST.findall(h.get("content", ""))}
        rows.append({
            "question": question,
            "n_hits": len(hits),
            "n_surfaced": len(surfaced),
            "n_gold": len(gold_set),
            "n_correct": len(surfaced & gold_set),
            "precision": (len(surfaced & gold_set) / len(surfaced)) if surfaced else 0.0,
        })
    return rows


def p3_by_kind(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    """Mean hop-1 precision over all 7 questions, the 5 issue lookups and the 2 aggregations.
    The aggregation questions surface almost no customer ids, which drags the overall mean."""
    groups = {
        "all7": rows,
        "issue5": [r for r in rows if not _is_aggregation(r["question"])],
        "agg2": [r for r in rows if _is_aggregation(r["question"])],
    }
    return {name: {"n": len(rs), "mean_precision": (sum(r["precision"] for r in rs) / len(rs)) if rs else 0.0,
                   "mean_surfaced": (sum(r["n_surfaced"] for r in rs) / len(rs)) if rs else 0.0}
            for name, rs in groups.items()}


def score_p3_hop1_precision(q_to_custs: dict[str, list[str]], k: int = 20) -> float:
    rows = score_p3_detail(q_to_custs, k)
    return sum(r["precision"] for r in rows) / len(rows) if rows else 0.0


def _row(q: dict[str, Any], pred: set[str], **extra: Any) -> dict[str, Any]:
    return {"question": q["question"], "kind": q["kind"], "gold": q["gold"],
            "pred": sorted(pred), **score_set(pred, set(q["gold"])), **extra}


def run_script_rows(questions: list[dict[str, Any]], k: int = 20) -> list[dict[str, Any]]:
    """Baseline 1: the fixed hand-written script, no filter."""
    rows = []
    for q in questions:
        hits = search_hits(q["question"], k)
        ids = sorted({cid for h in hits for cid in _RE_CUST.findall(h.get("content", ""))})
        rows.append(_row(q, companies_for_ids(ids), n_hits=len(hits), cust_ids=ids))
    return rows


def run_oracle_rows(questions: list[dict[str, Any]], product: dict[str, Any]) -> list[dict[str, Any]]:
    """Baseline 3 (ceiling): hop 1 is the question's CITED artifacts, read from the HERB source;
    hops 2-4 are the script's (extract CUST ids, resolve each through the exact title lookup).
    This is the answer extract+resolve gives if retrieval were perfect."""
    text_by_id: dict[str, str] = {}
    for rec in flatten_artifacts(product):
        text_by_id[rec.id] = text_by_id.get(rec.id, "") + "\n" + rec.text
    rows = []
    for q in questions:
        ids = sorted({cid for a in q["citations"] for cid in _RE_CUST.findall(text_by_id.get(a, ""))})
        rows.append(_row(q, companies_for_ids(ids), n_cited=len(q["citations"]), cust_ids=ids))
    return rows


def run_oracle_retrieval_rows(questions: list[dict[str, Any]], k: int = 300) -> list[dict[str, Any]]:
    """Informative: the script with its hop-1 retrieval (depth k) restricted to the cited artifacts."""
    rows = []
    for q in questions:
        pred = answer_company(q["question"], k=k, only_artifacts=set(q["citations"]))
        rows.append(_row(q, pred, k=k))
    return rows


_COST_LEDGER = _RESULTS_DIR / "step0-nx-answer-costs.jsonl"


def _append_cost_ledger(label: str, question: str, cost: float | None, **extra: Any) -> None:
    """Append-only record of every nx_answer call's spend: ``cost_usd`` is what the call was CHARGED against
    the cap (its recorded cost, or the per-call budget when the cost is unknown or the call failed).
    *extra* names the MVV stage and run (``stage``, ``mvv_run``) and whether the cost was recorded."""
    _COST_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with _COST_LEDGER.open("a") as f:
        f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "run": label,
                            "cost_usd": cost, "question": question[:90], **extra}) + "\n")


def ledger_spent(run_id: str | None) -> float:
    """What the MVV run *run_id* has already been charged: the sum of ``cost_usd`` over the ledger rows
    labelled ``mvv`` with ``mvv_run == run_id``. 0.0 for no run id or no ledger. A resume starts its cap
    from this, so the cap is cumulative over every invocation of one run.

    A call writes a RESERVE row (``reserve_id``, ``settled`` false, charged its per-call budget) before it starts
    and a SETTLED row with the same ``reserve_id`` after it returns. The last row per ``reserve_id`` counts, so a
    settled call is charged once, at its actual cost, and a call that never settled (an interrupt, a decode
    error) stays charged at its reserve. Rows without a ``reserve_id`` are summed as they stand."""
    if run_id is None or not _COST_LEDGER.exists():
        return 0.0
    total = 0.0
    reserves: dict[str, float] = {}
    for line in _COST_LEDGER.read_text().splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        cost = e.get("cost_usd") if isinstance(e, dict) else None
        if (isinstance(e, dict) and e.get("run") == MVV_LEDGER_LABEL and e.get("mvv_run") == run_id
                and isinstance(cost, (int, float)) and not isinstance(cost, bool)):
            rid = e.get("reserve_id")
            if isinstance(rid, str):
                reserves[rid] = float(cost)
            else:
                total += float(cost)
    return total + sum(reserves.values())


@dataclass
class Budget:
    """One spend cap shared by every call of a run. A call of unknown cost, or one that failed, is charged its
    per-call budget: a cap that an unreadable cost could slip under is no cap."""

    cap: float
    spent: float = 0.0
    unknown_cost_calls: int = 0

    @property
    def remaining(self) -> float:
        return self.cap - self.spent

    @property
    def exhausted(self) -> bool:
        return self.remaining <= MIN_CALL_USD

    def per_call(self) -> float:
        return min(PER_CALL_BUDGET_USD, self.remaining)

    def charge(self, cost: Any, per_call: float) -> float:
        if cost is None:
            self.unknown_cost_calls += 1
            charged = per_call
        else:
            charged = float(cost)
        self.spent += charged
        return charged


class MvvRows:
    """The MVV's rows as they land: one JSON line per row, appended at once, so a crash or a stopped sweep
    loses no paid row. ``load`` returns the rows by (``run_label``, ``question``); a later row replaces an
    earlier one for the same key, and a line cut short by a crash is skipped."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def exists(self) -> bool:
        return self.path.exists()

    def append(self, row: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def load(self) -> dict[tuple[str, str], dict[str, Any]]:
        out: dict[tuple[str, str], dict[str, Any]] = {}
        if not self.path.exists():
            return out
        for line in self.path.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and "run_label" in row and "question" in row:
                out[(row["run_label"], row["question"])] = row
        return out


def mvv_switch_violations(env: Mapping[str, str]) -> list[str]:
    """Reasons the MVV must not spend from *env*; empty means it may. nexus reads ``NX_ANSWER_RESOLUTION_REPAIR``
    from the process environment only (never from config): the resolution branch enters only when it is an on
    value, and a run without it is a guaranteed fail after the money is spent. Coverage
    (``NX_ANSWER_COVERAGE``) must not be off, or an unrecognised value (nexus reads that as off): the branch
    lives inside the coverage boundary."""
    bad: list[str] = []
    raw = str(env.get(RESOLUTION_SWITCH, "")).strip().lower()
    if raw not in ON_VALUES:
        bad.append(f"{RESOLUTION_SWITCH} is {raw!r}, not an on value {sorted(ON_VALUES)}: nexus reads it from the "
                   "process environment only, so the resolution branch could not enter (herb_sandbox.sh passes it "
                   "through its allowlist)")
    cov = str(env.get(COVERAGE_SWITCH, "")).strip().lower()
    if cov and cov not in ON_VALUES:
        bad.append(f"{COVERAGE_SWITCH} is {cov!r}: coverage is off (or unrecognised, which nexus reads as off), "
                   "and the resolution branch lives inside the coverage boundary")
    return bad


def mvv_llm_violations(env: Mapping[str, str]) -> list[str]:
    """Reasons nx_answer could not reach its LLM from *env*; empty means it can. Without ``claude`` on the PATH
    nexus does not fail: the inline planner and the coverage assessment fall away, a single-step template
    answers with a raw listing at exit 0, and every row reads as a valid zero (the 2026-11-05 first MVV run,
    ``results/herb/mvv-2026-11-05.no-claude-on-path.jsonl``). The token's value never enters a message."""
    bad: list[str] = []
    if shutil.which("claude", path=env.get("PATH", "")) is None:
        bad.append("claude is not on the PATH nx_answer will see: herb_sandbox.sh adds its directory when "
                   "HERB_SANDBOX_LLM=1 and claude is on the caller's PATH (or CLAUDE_BIN names it)")
    if not str(env.get("CLAUDE_CODE_OAUTH_TOKEN", "")).strip():
        bad.append("CLAUDE_CODE_OAUTH_TOKEN is not set: run under the nexus credential helper with "
                   "HERB_SANDBOX_LLM=1 (README-herb.md)")
    return bad


# --------------------------------------------------------------------------- #
# The RDR-147 MVV: grown plans, the RepairReport columns, two runs per question
# --------------------------------------------------------------------------- #
#
# A plan grown from one question captures later questions (RDR-147 research 9: 2 of 10 spike runs ran
# on grown plan 14). Before EVERY nx_answer run every grown plan in the sandbox is soft-disabled
# (HttpPlanLibrary.set_plan_disabled) and the ids of ALL plans are snapshotted. Sam's rule (2026-11-04): a
# run answered by the plan it grew itself is VALID (the defect was capture by an EARLIER question's plan,
# which the sweep prevents), plan_id 0 (nexus saved nothing) is valid, and a run is INVALID only when the
# plan it reports existed before the run and is a grown or user plan the sweep should have disabled.


def _declared_provenance(plan: Mapping[str, Any]) -> str | None:
    """The ``answer_kind.provenance`` a plan row declares in its ``plan_json`` (a JSON string or an object),
    or None for no readable declaration. Never raises."""
    pj = plan.get("plan_json")
    if isinstance(pj, str):
        try:
            pj = json.loads(pj)
        except ValueError:
            return None
    decl = pj.get("answer_kind") if isinstance(pj, Mapping) else None
    prov = decl.get("provenance") if isinstance(decl, Mapping) else None
    return prov if isinstance(prov, str) else None


def plan_is_grown(plan: Mapping[str, Any] | None) -> bool:
    """True unless the plan row is a curated template. A plan is a template only when its tags carry
    ``builtin-template`` and not ``grown`` AND its declaration (``plan_json.answer_kind.provenance``) does
    not say ``grown``: nexus (``nexus.plans.answer_kind``) reads a row as grown when either its tags or its
    declaration say so. A plan with no tags, no row, or the ``grown`` tag (the nx_answer grow site writes
    ``ad-hoc,grown``) is grown. Unlike nexus's eligibility, a tagged template with NO declaration reads as a
    template here (nexus calls that ``unattested``); the seed loader writes the declaration, so on a seeded
    sandbox the difference does not arise."""
    if plan is None:
        return True
    tags = {t.strip() for t in str(plan.get("tags") or "").split(",")}
    if "grown" in tags or "builtin-template" not in tags:
        return True
    return _declared_provenance(plan) == "grown"


@dataclass(frozen=True)
class Sweep:
    """What one pre-run sweep did: the ids it disabled, the id and tags of each (so a disabled user plan is
    visible), and the ids of EVERY plan that existed before the run (disabled ones included)."""

    disabled: list[int]
    records: list[dict[str, Any]]
    snapshot: frozenset[int]


def sweep_grown_plans(library: Any, reason: str = GROWN_PLAN_REASON) -> Sweep:
    """Snapshot every plan id, then soft-disable every active grown plan, once each.

    *library* is an ``HttpPlanLibrary`` (or the sandbox adapter). Two lists are read: all plans
    (``include_disabled=True``, the snapshot) and the ACTIVE ones (the ones to disable; a disabled plan is not
    disabled again). ``NxError`` when a plan will not disable, or when either list filled ``PLAN_SWEEP_LIMIT``
    (it may be partial, and a grown plan past the cut would still match): a run never starts on a sweep that
    may have missed one."""
    everything = library.list_plans(limit=PLAN_SWEEP_LIMIT, include_disabled=True)
    active = library.list_plans(limit=PLAN_SWEEP_LIMIT, include_disabled=False)
    for rows in (everything, active):
        if len(rows) >= PLAN_SWEEP_LIMIT:
            raise NxError(f"plan list reached its limit of {PLAN_SWEEP_LIMIT}: a grown plan may have been missed")
    snapshot = frozenset(int(r["id"]) for r in [*everything, *active])
    grown = [r for r in active if plan_is_grown(r)]
    records = [{"id": int(r["id"]), "tags": str(r.get("tags") or "")} for r in grown]
    for rec in records:
        if not library.set_plan_disabled(rec["id"], reason=reason):
            raise NxError(f"plan {rec['id']} did not disable: a run on it would not be a measurement")
    return Sweep([r["id"] for r in records], records, snapshot)


def _created_at(row: Mapping[str, Any]) -> datetime | None:
    """A plan row's ``created_at`` as an aware datetime, or None when it is absent or unreadable."""
    raw = row.get("created_at")
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        at = datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return at if at.tzinfo else at.replace(tzinfo=timezone.utc)


def plan_validity(library: Any, plan_id: Any, snapshot: frozenset[int],
                  started_at: datetime | None = None) -> tuple[bool, str, bool]:
    """(valid, reason, grown_by_call) for a run that reports *plan_id*, given the pre-run *snapshot* of plan ids.

    No plan row (``None``, or ``0``: nexus saved nothing) is valid. An id absent from the snapshot was created
    by this very call (the inline planner grew and saved it): valid, ``grown_by_call``. An id in the snapshot
    is read back: a row created at or after *started_at* was written by this very call (nexus re-grows a plan
    under an id it already had: the 2026-11-05 held-out rows on plans 13 and 14, created during their own
    calls), valid and ``grown_by_call``; otherwise a grown or user plan is invalid (the sweep should have
    disabled it), a row that cannot be read is invalid, a curated template is valid. An id that is not an
    integer is invalid."""
    if plan_id is None:
        return True, "", False
    if isinstance(plan_id, bool) or not isinstance(plan_id, int) or plan_id < 0:
        return False, f"plan id {plan_id!r} is not a plan id", False
    if plan_id == 0:
        return True, "", False
    if plan_id not in snapshot:
        return True, "", True
    row = library.get_plan(plan_id)
    if row is None:
        return False, f"plan row {plan_id} not found: the run cannot be shown to be off a grown plan", False
    created = _created_at(row)
    if started_at is not None and created is not None and created >= started_at:
        return True, "", True
    if plan_is_grown(row):
        return False, f"run on grown plan {plan_id} (it existed before the run: the sweep should have disabled it)", False
    return True, "", False


def _names(value: Any) -> list[str]:
    """The names in a report list: a list's items, a mapping's keys, else none."""
    if isinstance(value, Mapping):
        return [str(k) for k in value]
    if isinstance(value, (list, tuple, set, frozenset)):
        return [str(x) for x in value]
    return []


def parse_dropped_pair(entry: Any) -> tuple[str, str, str] | None:
    """(key, item, reason) of one ``joined_dropped`` entry ``<key>@<item id>:<reason>``, or None. Parsed from the
    right: the reason has no ``:`` and the item id no ``@``, but a key may hold either (``a@b:c``)."""
    if not isinstance(entry, str) or ":" not in entry:
        return None
    head, reason = entry.rsplit(":", 1)
    if "@" not in head:
        return None
    key, item = head.rsplit("@", 1)
    return key, item, reason


def repair_columns(report: Any) -> dict[str, Any]:
    """The RepairReport fields an nx_answer row carries (``REPAIR_FIELDS``), read tolerantly: nexus's Step 4 adds
    them to the report, a report without them (an older nexus, or no repair) is a branch that did not enter,
    and a field its fix round added reads as empty on a report that predates it. ``resolution_entered`` is True
    only for a boolean True; ``joined_dropped`` is the list of ``<key>@<item>:<reason>`` pairs (an older nexus's
    count is copied as given); ``repair_outcome`` is the report's ``outcome``."""
    rep = report if isinstance(report, Mapping) else {}

    def listy(v: Any) -> list[Any]:
        return list(v) if isinstance(v, (list, tuple, set, frozenset)) else []

    def keep(v: Any) -> Any:
        return dict(v) if isinstance(v, Mapping) else listy(v)

    def count(v: Any) -> int:
        return v if isinstance(v, int) and not isinstance(v, bool) else 0

    def text(v: Any) -> str:
        return v if isinstance(v, str) else ""

    dropped = rep.get("joined_dropped")
    dropped = list(dropped) if isinstance(dropped, (list, tuple, set, frozenset)) else (
        dropped if isinstance(dropped, int) and not isinstance(dropped, bool) else [])
    cols = {
        "resolution_entered": rep.get("resolution_entered") is True,
        "resolution_reason": text(rep.get("resolution_reason")),
        "resolution_tokens": count(rep.get("resolution_tokens")),
        "anchor_key_ids": [str(x) for x in listy(rep.get("anchor_key_ids"))],
        "resolved": keep(rep.get("resolved")),
        "missing": keep(rep.get("missing")),
        "ambiguous": keep(rep.get("ambiguous")),
        "joined_added": count(rep.get("joined_added")),
        "items_merged": count(rep.get("items_merged")),
        "repair_outcome": text(rep.get("outcome")),
        "joined_dropped": dropped,
        "tokens_dropped": [str(x) for x in listy(rep.get("tokens_dropped"))],
        "anchors_scanned": count(rep.get("anchors_scanned")),
        "anchors_without_pattern": count(rep.get("anchors_without_pattern")),
        # RDR-147 option 1 (Sam 2026-11-05): the answer was generated on an exact-key join the re-check did not cover
        "generated_on_join": rep.get("generated_on_join") is True,
    }
    for name in _TRUNCATED_LISTS:
        cols[f"{name}_truncated"] = count(rep.get(f"{name}_truncated"))
    return cols


def in_band_error(final_text: Any) -> str | None:
    """The notice (its first ``IN_BAND_ERROR_NOTICE_CHARS`` characters) when *final_text* opens with an error-shape
    marker nexus returns in band, else None. See ``_IN_BAND_ERROR_RE`` for what counts and what does not."""
    if not isinstance(final_text, str) or _IN_BAND_ERROR_RE.match(final_text) is None:
        return None
    return final_text.strip()[:IN_BAND_ERROR_NOTICE_CHARS]


def _guarded_nx_answer(question: str, plan_library: Any | None, budget: float, max_steps: int,
                       timeout: int, before_call: Callable[[], None] | None = None,
                       ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """One nx_answer run behind the grown-plan sweep. Returns (envelope or None when the call failed, the columns
    every row carries: plan, validity, plans disabled before the run with their tags, the RepairReport fields).

    A failed call (``NxError``: exit, timeout, no envelope) is an INVALID row carrying the error text, never an
    exception: one bad call must not lose the paid rows around it. So is a call that returned an in-band error
    notice (``in_band_error``): it is not a measurement of the branch, and the envelope still comes back so its
    cost is charged. A failed SWEEP still raises: no run starts on a sweep that may have missed a grown plan (the
    rows so far are on disk; resume continues). Without a plan library the run is recorded but never valid:
    nothing showed it was off a grown plan. *before_call* runs after the sweep and just before the call (the
    spend reserve): a sweep that raises has reserved nothing."""
    sweep = sweep_grown_plans(plan_library) if plan_library is not None else None
    base = {"plans_disabled_before": sweep.disabled if sweep else [],
            "plans_disabled_records": sweep.records if sweep else []}
    if before_call is not None:
        before_call()
    # Truncated to the second: the engine stamps created_at in whole seconds.
    started_at = datetime.now(timezone.utc).replace(microsecond=0)
    try:
        env = nx_answer_call(question, corpus_arg(PRODUCT), max_steps, timeout, budget_usd=budget)
    except NxError as e:
        return None, {**base, "plan_id": None, "valid": False, "invalid_reason": f"nx_answer error: {e}",
                      "plan_grown_by_call": False, "error": str(e), **repair_columns(None)}
    notice = in_band_error(env.get("final_text"))
    if notice is not None:
        return env, {**base, "plan_id": env.get("plan_id"), "valid": False,
                     "invalid_reason": f"nx_answer returned an error notice: {notice}",
                     "plan_grown_by_call": False, "error": notice, **repair_columns(env.get("repair"))}
    grown_by_call = False
    if sweep is not None:
        valid, why, grown_by_call = plan_validity(plan_library, env.get("plan_id"), sweep.snapshot, started_at)
    else:
        valid, why = False, "plan not checked: no plan library, grown plans were not disabled"
    return env, {**base, "plan_id": env.get("plan_id"), "valid": valid, "invalid_reason": why,
                 "plan_grown_by_call": grown_by_call, **repair_columns(env.get("repair"))}


def _kept(prior: Mapping[tuple[str, str], Mapping[str, Any]] | None, label: str, question: str):
    """The row a resume keeps for (*label*, *question*): a prior row that ran validly, else None."""
    row = (prior or {}).get((label, question))
    return dict(row) if row is not None and _usable(row) else None


def _execute(question: str, *, label: str, ledger_label: str, run_id: str | None, plan_library: Any | None,
             budget: Budget, max_steps: int, timeout: int) -> tuple[dict[str, Any], dict[str, Any], Any, float]:
    """One guarded call, charged to *budget* and written to the cost ledger. Returns (envelope or {}, row
    columns, the recorded cost or None, the cost charged).

    The ledger gets a RESERVE row (the per-call budget) after the sweep and before the call, and a settled row
    after it: ``ledger_spent`` never undercounts a paid call. Any exception out of the call (an interrupt, a
    decode error that is not an ``NxError``) leaves the reserve in place and charges it to *budget*."""
    per_call = budget.per_call()
    reserve_id = uuid.uuid4().hex
    reserved = False

    def reserve() -> None:
        nonlocal reserved
        _append_cost_ledger(ledger_label, question, per_call, stage=label, mvv_run=run_id,
                            reserve_id=reserve_id, settled=False)
        reserved = True

    try:
        env, cols = _guarded_nx_answer(question, plan_library, per_call, max_steps, timeout, before_call=reserve)
    except BaseException:
        if reserved:
            budget.spent += per_call
        raise
    cost = None if env is None else env.get("cost_usd")
    charged = budget.charge(cost, per_call)
    _append_cost_ledger(ledger_label, question, charged, stage=label, mvv_run=run_id, reserve_id=reserve_id,
                        settled=True, cost_recorded=cost is not None, error="error" in cols)
    return env or {}, cols, cost, charged


def run_nx_answer_rows(
    questions: list[dict[str, Any]],
    universe: list[str],
    cap_usd: float,
    max_steps: int = 6,
    timeout: int = 420,
    label: str = "run",
    plan_library: Any | None = None,
    *,
    budget: Budget | None = None,
    ledger_label: str | None = None,
    run_id: str | None = None,
    sink: MvvRows | None = None,
    prior: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Baseline 2 (the product): nx_answer over the corpus, scored on the companies named in its
    answer, recall AND precision. Spend is charged to *budget* (a fresh one of *cap_usd* by default): each
    run's recorded cost_usd, or its per-call budget when the cost is unknown or the call failed. Once the cap
    is reached the remaining questions are not run and the baseline is marked incomplete.

    With *plan_library* every grown plan is disabled before each run and each row is marked valid or
    invalid (``valid``, ``invalid_reason``); without one no row is valid. Every row carries its run
    ``label`` and the RepairReport columns (``REPAIR_FIELDS``). Each row is appended to *sink* as it lands;
    a row of *prior* that ran validly is kept and not re-run (a resume)."""
    budget = budget if budget is not None else Budget(cap_usd)
    ledger_label = ledger_label or label
    start, unknown0 = budget.spent, budget.unknown_cost_calls
    rows: list[dict[str, Any]] = []

    def land(row: dict[str, Any]) -> None:
        rows.append(row)
        if sink is not None:
            sink.append(row)

    for q in questions:
        kept = _kept(prior, label, q["question"])
        if kept is not None:
            rows.append(kept)
            continue
        if budget.exhausted:
            land(_row(q, set(), skipped="cost cap reached", cost_usd=0.0, run_label=label,
                      valid=False, invalid_reason="skipped: cost cap reached"))
            continue
        env, cols, cost, charged = _execute(q["question"], label=label, ledger_label=ledger_label, run_id=run_id,
                                            plan_library=plan_library, budget=budget, max_steps=max_steps,
                                            timeout=timeout)
        pred = extract_companies(env.get("final_text") or "", universe)
        ev_ids = sorted({cid for t in env.get("evidence_texts") or [] for cid in _RE_CUST.findall(t or "")})
        land(_row(q, pred, final_text=env.get("final_text"), run_label=label,
                  step_count=env.get("step_count"), cost_usd=cost, cost_charged_usd=charged,
                  coverage_state=env.get("coverage_state"), coverage_reason=env.get("coverage_reason"),
                  n_evidence_chunks=len(env.get("chunks") or []),
                  evidence_cust_ids=ev_ids,
                  evidence_companies=sorted(companies_for_ids(ev_ids)), **cols))
        console.print(f"  nx_answer[{label}] cost {cost} (charged {charged:.4f}, total {budget.spent:.4f}) "
                      f"recall={rows[-1]['recall']:.2f} precision={rows[-1]['precision']:.2f} "
                      f"valid={rows[-1]['valid']} entered={rows[-1]['resolution_entered']}  {q['question'][:50]}")
    meta = {
        "total_cost_usd": round(budget.spent - start, 6),
        "cumulative_cost_usd": round(budget.spent, 6),
        "cap_usd": budget.cap,
        "runs_with_unknown_cost": budget.unknown_cost_calls - unknown0,
        "complete": all("skipped" not in r for r in rows),
    }
    return rows, meta


def run_mvv_nx_answer(
    questions: list[dict[str, Any]],
    universe: list[str],
    cap_usd: float,
    plan_library: Any,
    max_steps: int = 6,
    timeout: int = 420,
    *,
    budget: Budget | None = None,
    run_id: str | None = None,
    sink: MvvRows | None = None,
    prior: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Every question twice, run ``a`` then run ``b``, under ONE spend cap: ``b`` gets what ``a`` left. The cap
    is cumulative over a resume: the budget starts from what the cost ledger already charged to *run_id*."""
    budget = budget if budget is not None else Budget(cap_usd, spent=ledger_spent(run_id))
    start = budget.spent
    rows: dict[str, list[dict[str, Any]]] = {}
    by_run: dict[str, dict[str, Any]] = {}
    for label in RUN_LABELS:
        rows[label], by_run[label] = run_nx_answer_rows(
            questions, universe, budget.cap, max_steps, timeout, label=label, plan_library=plan_library,
            budget=budget, ledger_label=MVV_LEDGER_LABEL, run_id=run_id, sink=sink, prior=prior)
    meta = {"total_cost_usd": round(budget.spent - start, 6), "cumulative_cost_usd": round(budget.spent, 6),
            "cap_usd": budget.cap,
            "runs_with_unknown_cost": sum(m["runs_with_unknown_cost"] for m in by_run.values()),
            "complete": all(m["complete"] for m in by_run.values()), "by_run": by_run}
    return rows, meta


def run_heldout_rows(
    heldout: list[dict[str, Any]],
    cap_usd: float,
    plan_library: Any,
    max_steps: int = 6,
    timeout: int = 420,
    label: str = "heldout",
    *,
    budget: Budget | None = None,
    ledger_label: str | None = None,
    run_id: str | None = None,
    sink: MvvRows | None = None,
    prior: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The single-hop held-out questions, each run ONCE with the resolution switch on (the environment's),
    behind the same grown-plan sweep. Rows record the plan, validity and RepairReport columns; they are not
    scored against a gold (the count is of runs where the branch entered)."""
    budget = budget if budget is not None else Budget(cap_usd)
    ledger_label = ledger_label or label
    start, unknown0 = budget.spent, budget.unknown_cost_calls
    rows: list[dict[str, Any]] = []

    def land(row: dict[str, Any]) -> None:
        rows.append(row)
        if sink is not None:
            sink.append(row)

    for q in heldout:
        base = {"herb_index": q["herb_index"], "question": q["question"], "type": q.get("type"), "run_label": label}
        kept = _kept(prior, label, q["question"])
        if kept is not None:
            rows.append(kept)
            continue
        if budget.exhausted:
            land({**base, "skipped": "cost cap reached", "cost_usd": 0.0, "valid": False,
                  "invalid_reason": "skipped: cost cap reached", **repair_columns(None)})
            continue
        env, cols, cost, charged = _execute(q["question"], label=label, ledger_label=ledger_label, run_id=run_id,
                                            plan_library=plan_library, budget=budget, max_steps=max_steps,
                                            timeout=timeout)
        land({**base, "final_text": env.get("final_text"), "step_count": env.get("step_count"),
              "cost_usd": cost, "cost_charged_usd": charged, "coverage_state": env.get("coverage_state"),
              "coverage_reason": env.get("coverage_reason"), **cols})
        console.print(f"  heldout cost {cost} (charged {charged:.4f}, total {budget.spent:.4f}) "
                      f"valid={cols['valid']} entered={cols['resolution_entered']}  {q['question'][:50]}")
    meta = {"total_cost_usd": round(budget.spent - start, 6), "cumulative_cost_usd": round(budget.spent, 6),
            "cap_usd": budget.cap, "runs_with_unknown_cost": budget.unknown_cost_calls - unknown0,
            "complete": all("skipped" not in r for r in rows)}
    return rows, meta


def _usable(row: Mapping[str, Any]) -> bool:
    """A row that counts: it ran, and its plan was shown not to be a grown one."""
    return row.get("valid") is True and not row.get("skipped")


def _id_causes(r: Mapping[str, Any], anchors: list[str], cust_to_company: Mapping[str, str],
               named: set[str]) -> dict[str, dict[str, Any]]:
    """Per anchor id, why its company is or is not in the answer: what the branch did with the key (``cap-dropped``
    when the 12-key cap left it unresolved, ``missing``, ``ambiguous``, ``resolved``, else ``unaccounted``),
    its registry company, whether the answer names it, and the reasons any joined item for it was dropped."""
    capped, missing = set(_names(r.get("tokens_dropped"))), set(_names(r.get("missing")))
    ambiguous, resolved = set(_names(r.get("ambiguous"))), set(_names(r.get("resolved")))
    dropped: dict[str, list[str]] = {}
    raw = r.get("joined_dropped")
    for entry in raw if isinstance(raw, (list, tuple)) else []:
        parsed = parse_dropped_pair(entry)
        if parsed:
            dropped.setdefault(parsed[0], []).append(parsed[2])
    out: dict[str, dict[str, Any]] = {}
    for a in anchors:
        cause = ("cap-dropped" if a in capped else "missing" if a in missing else "ambiguous" if a in ambiguous
                 else "resolved" if a in resolved else "unaccounted")
        company = cust_to_company.get(a)
        out[a] = {"cause": cause, "company": company, "named": company is not None and company.lower() in named,
                  "dropped": dropped.get(a, [])}
    return out


def _count(v: Any) -> int:
    return v if isinstance(v, int) and not isinstance(v, bool) else 0


def join_completeness(rows: list[dict[str, Any]], cust_to_company: Mapping[str, str], *,
                      floor: int = DOMAIN_FLOOR, n_issue: int = ISSUE_QUESTIONS,
                      max_misses: int = MAX_DOMAIN_MISSES) -> dict[str, Any]:
    """Join completeness of ONE run, gated (RDR-147 MVV).

    The domain is the issue questions whose ``anchor_key_ids`` is non-empty in this run: the branch scans
    the boundary's anchors, not all the evidence. A question passes when it is in the domain, its answer
    names every company its anchor ids give (through the registry), and ``resolution_entered`` is true
    (a pass through the semantic repair alone does not count). The run passes when ``n_issue`` issue rows
    all ran validly, at least ``floor`` are in the domain (fewer is a vacuous run, a fail), and at most
    ``max_misses`` in-domain questions fail (Sam 2026-11-05: one miss per run, the re-check's variance on
    the same joined item). Anchor ids cut on the wire cannot be graded: that question fails.

    The ``verdict`` is ``invalid`` (not a measurement) when a row did not run validly or the issue rows are not
    ``n_issue``; else ``failed`` or ``passed``. A usable question outside the domain says why (its
    ``domain_exclusion``: a question whose evidence holds ids but whose anchors are empty reads "boundary not
    reached or branch not considered", with its plan and reason); an in-domain question carries a cause per
    anchor id (``id_causes``) and a resolved-only completeness (``resolved_only``) beside the gated one.
    Completeness over ALL hydrated evidence ids and the resolved-only variant are reported, never gated."""
    issue = [r for r in rows if r.get("kind") == "issue"]
    questions = []
    for r in issue:
        usable = _usable(r)
        anchors = [str(a) for a in (r.get("anchor_key_ids") or [])]
        in_domain = usable and bool(anchors)
        companies = sorted({cust_to_company[a] for a in anchors if a in cust_to_company})
        named = {c.lower() for c in r.get("pred", [])}
        missing = sorted(c for c in companies if c.lower() not in named)
        cut = _count(r.get("anchor_key_ids_truncated"))
        complete = bool(companies) and not missing and not cut
        entered = r.get("resolution_entered") is True
        causes = _id_causes(r, anchors, cust_to_company, named) if in_domain else {}
        res_companies = sorted({cust_to_company[a] for a, c in causes.items()
                                if c["cause"] == "resolved" and a in cust_to_company})
        res_missing = sorted(c for c in res_companies if c.lower() not in named)
        exclusion = None
        if usable and not anchors:
            exclusion = {"evidence_holds_ids": bool(r.get("evidence_cust_ids")), "plan_id": r.get("plan_id"),
                         "resolution_reason": r.get("resolution_reason") or "",
                         "anchors_scanned": _count(r.get("anchors_scanned")),
                         "anchors_without_pattern": _count(r.get("anchors_without_pattern"))}
        if not usable:
            reason = r.get("invalid_reason") or "row did not run"
        elif not anchors:
            if exclusion["evidence_holds_ids"]:
                reason = (f"outside the domain: no anchor ids, but the evidence holds {len(r['evidence_cust_ids'])} "
                          f"id(s) (boundary not reached or branch not considered; plan_id={exclusion['plan_id']}, "
                          f"resolution_reason={exclusion['resolution_reason']!r}, "
                          f"anchors_scanned={exclusion['anchors_scanned']}, "
                          f"anchors_without_pattern={exclusion['anchors_without_pattern']})")
            else:
                reason = "no anchor ids: outside the domain"
        elif cut:
            reason = f"anchor ids cut on the wire ({cut} left out): cannot be graded"
        elif not companies:
            reason = "anchor ids not in the registry: cannot be graded"
        elif missing:
            reason = f"answer does not name {missing}"
        elif not entered:
            reason = "resolution branch did not enter"
        else:
            reason = ""
        questions.append({"question": r.get("question"), "in_domain": in_domain, "valid": usable,
                          "resolution_entered": entered, "anchor_key_ids": anchors,
                          "anchor_companies": companies, "named": sorted(named), "missing_companies": missing,
                          "complete": complete, "passed": in_domain and complete and entered, "reason": reason,
                          "id_causes": causes, "domain_exclusion": exclusion,
                          "resolved_only": {"companies": res_companies, "missing_companies": res_missing,
                                            "complete": bool(res_companies) and not res_missing}})
    n_in = sum(q["in_domain"] for q in questions)
    n_missed = sum(1 for q in questions if q["in_domain"] and not q["passed"])
    n_invalid = sum(not q["valid"] for q in questions)
    vacuous = n_in < floor
    ev = []
    for r in issue:
        if not _usable(r):
            continue
        ev_companies = sorted({cust_to_company[c] for c in r.get("evidence_cust_ids") or [] if c in cust_to_company})
        named = {c.lower() for c in r.get("pred", [])}
        miss = sorted(c for c in ev_companies if c.lower() not in named)
        ev.append({"question": r.get("question"), "n_evidence_ids": len(r.get("evidence_cust_ids") or []),
                   "evidence_companies": ev_companies, "missing_companies": miss, "complete": not miss})
    if len(issue) != n_issue or n_invalid:
        verdict = "invalid"
    elif vacuous or n_missed > max_misses:
        verdict = "failed"
    else:
        verdict = "passed"
    if len(issue) != n_issue:
        reason = f"{len(issue)} issue rows, expected {n_issue}"
    elif n_invalid:
        reason = f"{n_invalid} row(s) invalid or skipped: the run is not a measurement"
    elif vacuous:
        reason = f"vacuous: {n_in} of {len(issue)} issue questions in the domain, floor {floor}"
    else:
        reason = (f"{n_missed} in-domain question(s) failed, at most {max_misses} allowed" if verdict == "failed"
                  else f"{n_missed} in-domain miss(es), at most {max_misses} allowed" if n_missed else "")
    return {"n_issue": len(issue), "in_domain": n_in, "floor": floor, "vacuous": vacuous, "n_invalid": n_invalid,
            "n_missed": n_missed, "max_misses": max_misses,
            "passed": verdict == "passed", "verdict": verdict, "reason": reason, "questions": questions,
            "excluded_with_evidence_ids": sum(1 for q in questions
                                              if q["domain_exclusion"] and q["domain_exclusion"]["evidence_holds_ids"]),
            "resolved_only_completeness": {
                "n_in_domain": n_in, "n_complete": sum(1 for q in questions
                                                       if q["in_domain"] and q["resolved_only"]["complete"]),
                "gated": False},
            "evidence_completeness": {"n": len(ev), "n_complete": sum(e["complete"] for e in ev),
                                      "n_with_evidence": sum(1 for e in ev if e["n_evidence_ids"]),
                                      "questions": ev, "gated": False}}


def precision_gate(rows: list[dict[str, Any]], threshold: float = PRECISION_GATE,
                   n_issue: int = ISSUE_QUESTIONS) -> dict[str, Any]:
    """Precision of ONE run, gated: the MEAN of per-question precision (``score_set``) over the issue questions
    whose answer named at least one company, at least *threshold*. A question that named none is left out
    (Sam 2026-11-05: precision of the named companies is undefined there, and join completeness already counts
    the non-answer); its ``per_question`` value is None. A run in which no question named a company fails.
    The pooled ratio is reported beside it. The run is ``invalid`` (not a measurement) unless ``n_issue`` issue
    rows all ran validly; else ``passed`` or ``failed``."""
    issue = [r for r in rows if r.get("kind") == "issue"]
    valid = [r for r in issue if _usable(r)]
    per: list[float | None] = [float(r["precision"]) if r["n_pred"] else None for r in valid]
    named = [p for p in per if p is not None]
    mean = sum(named) / len(named) if named else 0.0
    n_pred = sum(r["n_pred"] for r in valid)
    pooled = sum(r["hit"] for r in valid) / n_pred if n_pred else 0.0
    if len(issue) != n_issue or len(valid) != n_issue:
        verdict = "invalid"
    else:
        verdict = "passed" if named and mean >= threshold - 1e-9 else "failed"
    # Reported, never gated (RDR-147 R9): precision of the answers only option 1 admitted, beside the rest.
    split: dict[str, dict[str, Any]] = {}
    for key, pick in (("generated_on_join", True), ("rechecked", False)):
        vals = [float(r["precision"]) for r in valid if r["n_pred"] and (r.get("generated_on_join") is True) is pick]
        split[key] = {"n": len(vals), "mean": sum(vals) / len(vals) if vals else None}
    return {"mean": mean, "pooled": pooled, "per_question": per, "n_issue": len(issue), "n_valid": len(valid),
            "n_named": len(named), "n_left_out": len(per) - len(named), "split": split,
            "threshold": threshold, "passed": verdict == "passed", "verdict": verdict}


def _combine(verdicts: list[str]) -> str:
    """Across runs a failed run on valid rows settles the gate (``failed``), then an invalid one (``invalid``:
    not a measurement), else ``passed``."""
    return "failed" if "failed" in verdicts else "invalid" if "invalid" in verdicts else "passed"


def heldout_report(rows: list[dict[str, Any]], runs: Mapping[str, Any], n_asked: int | None = None) -> dict[str, Any]:
    """The held-out count as entered of valid of asked, never as a bare number: ``n_valid``, ``n_skipped`` (cap),
    ``n_invalid`` (failed or on a grown plan), ``entered`` (valid rows where the branch entered), why the
    others did not (``not_entered_reasons``, by ``resolution_reason``), and whether the count reads at all.
    A zero is ``interpretable: false`` when no held-out run was valid, or when the branch entered no issue
    question in either run (the switch or the entity pattern may be mis-set, and a zero then says nothing)."""
    valid = [r for r in rows if _usable(r)]
    reasons: list[str] = []
    if not valid:
        reasons.append("no valid held-out run")
    if not any(runs[lb]["issue_entered"] for lb in RUN_LABELS if lb in runs):
        reasons.append("issue_entered is 0 in both runs: the switch or the entity pattern may be mis-set, "
                       "so a held-out zero is not evidence of anything")
    not_entered: dict[str, int] = {}
    for r in valid:
        if r.get("resolution_entered") is not True:
            key = r.get("resolution_reason") or r.get("repair_outcome") or "(no reason reported)"
            not_entered[key] = not_entered.get(key, 0) + 1
    return {"asked": len(rows) if n_asked is None else n_asked, "n": len(rows), "n_valid": len(valid),
            "n_skipped": sum(1 for r in rows if r.get("skipped")),
            "n_invalid": sum(1 for r in rows if not _usable(r) and not r.get("skipped")),
            "entered": sum(1 for r in valid if r.get("resolution_entered") is True),
            # RDR-147 R9: an entered join now answers; count answered held-out outcomes, not firings only
            "answered": sum(1 for r in valid if not str(r.get("final_text") or "").startswith("[non-answer:")),
            "answered_on_join": sum(1 for r in valid if r.get("generated_on_join") is True),
            "not_entered_reasons": not_entered, "interpretable": not reasons, "reasons": reasons}


def score_mvv(rows_by_label: Mapping[str, list[dict[str, Any]]], cust_to_company: Mapping[str, str],
              heldout_rows: list[dict[str, Any]], script_f1: float = SCRIPT_ISSUE5_F1,
              n_asked: int | None = None) -> dict[str, Any]:
    """The MVV verdict from the two labelled runs and the held-out runs.

    Gated: join completeness and precision, each passing in BOTH runs. The ``verdict`` is one of
    ``passed``, ``failed`` (a run made of valid rows failed a gate: the finding), or ``invalid`` (a run is
    missing, or a row did not run validly or was skipped: not a measurement, rerun it; it is never a finding
    about the branch). ``passed`` is True only for a ``passed`` verdict. Reported only: issue5
    recall/precision/F1 beside the script's F1, agg2 recall/precision/F1, per-run counts of issue questions
    where the branch entered, and the held-out count (``heldout_report``). Invalid rows are excluded from every
    figure and counted."""
    runs: dict[str, Any] = {}
    for label, rows in rows_by_label.items():
        usable = [r for r in rows if _usable(r)]
        summ = summarize(usable)
        jc, pg = join_completeness(rows, cust_to_company), precision_gate(rows)
        run_verdict = ("invalid" if "invalid" in (jc["verdict"], pg["verdict"])
                       else "failed" if "failed" in (jc["verdict"], pg["verdict"]) else "passed")
        runs[label] = {
            "run_label": label, "join_completeness": jc, "precision": pg, "verdict": run_verdict,
            "n_invalid": sum(not _usable(r) for r in rows),
            "issue_entered": sum(1 for r in usable if r.get("kind") == "issue" and r.get("resolution_entered") is True),
            "summary": summ, "issue5_recall": summ["issue5"]["recall"], "issue5_precision": summ["issue5"]["precision"],
            "issue5_f1": summ["issue5"]["f1"], "agg2": summ["agg2"], "passed": run_verdict == "passed"}
    jc_v = _combine([runs[lb]["join_completeness"]["verdict"] if lb in runs else "invalid" for lb in RUN_LABELS])
    pr_v = _combine([runs[lb]["precision"]["verdict"] if lb in runs else "invalid" for lb in RUN_LABELS])
    verdict = _combine([jc_v, pr_v])
    return {"labels": list(RUN_LABELS), "runs": runs, "verdict": verdict, "passed": verdict == "passed",
            "join_completeness_verdict": jc_v, "precision_verdict": pr_v,
            "join_completeness_passed": jc_v == "passed", "precision_passed": pr_v == "passed",
            "script_issue5_f1": script_f1, "heldout": heldout_report(heldout_rows, runs, n_asked)}


def run_mvv(questions: list[dict[str, Any]], universe: list[str], cust_to_company: Mapping[str, str],
            heldout: list[dict[str, Any]], cap_usd: float, plan_library: Any, *, run_id: str | None = None,
            sink: MvvRows | None = None,
            prior: Mapping[tuple[str, str], Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Both labelled runs, then the held-out set, under one cap; return rows, held-out rows, meta, verdict.

    Refuses (``NxError``) before any sweep, call or ledger row unless the resolution switch is on in this
    process's environment (``mvv_switch_violations``). With *run_id* the cap is cumulative over every
    invocation of that run (the cost ledger is read back); with *sink* every row is appended as it lands; with
    *prior* (``sink.load()``) valid rows are kept and not re-run."""
    bad = mvv_switch_violations(os.environ) + mvv_llm_violations(os.environ)
    if bad:
        raise NxError("the MVV refuses to spend: " + "; ".join(bad))
    budget = Budget(cap_usd, spent=ledger_spent(run_id))
    start = budget.spent
    rows, meta = run_mvv_nx_answer(questions, universe, cap_usd, plan_library, budget=budget, run_id=run_id,
                                   sink=sink, prior=prior)
    held_rows, held_meta = run_heldout_rows(heldout, cap_usd, plan_library, budget=budget,
                                            ledger_label=MVV_LEDGER_LABEL, run_id=run_id, sink=sink, prior=prior)
    return {"rows": rows, "heldout_rows": held_rows,
            "score": score_mvv(rows, cust_to_company, held_rows, n_asked=len(heldout)),
            "meta": {"cap_usd": cap_usd, "total_cost_usd": round(budget.spent - start, 6),
                     "cumulative_cost_usd": round(budget.spent, 6), "nx_answer": meta, "heldout": held_meta,
                     "complete": meta["complete"] and held_meta["complete"]}}


def copy_scorecard(dest_dir: Path) -> Path:
    """Copy ``results/herb/herb-scorecard.json`` byte for byte into *dest_dir* (the MVV run copies it
    into nexus at ``docs/rdr/evidence/rdr-147``). ``NxError``, and nothing created, when it is missing."""
    import shutil

    if not _SCORECARD.exists():
        raise NxError(f"{_SCORECARD} does not exist: run the harness first")
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / _SCORECARD.name
    shutil.copyfile(_SCORECARD, out)
    return out


# The plan library lives in nexus, which the harness interpreter does not import: like the P2 resolve
# step, it is reached through the conexus interpreter, behind the same sandbox guard (re-asserted in the
# driver, from its own environment, before a client exists) and a loopback-only service URL.

_PLAN_DRIVER = textwrap.dedent(
    """
    from __future__ import annotations
    import json, os, pwd, sys
    from urllib.parse import urlparse

    @@GUARD@@

    bad = p2_sandbox_violations(os.environ, sys.executable, pwd.getpwuid(os.getuid()).pw_dir)
    if bad:
        raise SystemExit("plan driver refuses to run: " + "; ".join(bad))
    op, args = sys.argv[1], json.loads(sys.argv[2])
    if op not in ("list_plans", "get_plan", "set_plan_disabled"):
        raise SystemExit("plan driver refuses operation " + repr(op))
    from nexus.db.t2.http_plan_library import HttpPlanLibrary
    lib = HttpPlanLibrary()
    url = str(lib._base_url)
    print("plan service url: " + url, file=sys.stderr, flush=True)
    if urlparse(url).hostname not in ("127.0.0.1", "localhost", "::1"):
        raise SystemExit("refusing non-loopback service url " + url)
    print("@@SENTINEL@@" + json.dumps({"result": getattr(lib, op)(**args), "service_url": url}, default=str))
    """
).replace("@@GUARD@@", textwrap.dedent(inspect.getsource(p2_sandbox_violations))).replace("@@SENTINEL@@", _NXA_SENTINEL)


class SandboxPlanLibrary:
    """The three ``HttpPlanLibrary`` calls the MVV uses (``list_plans``, ``get_plan``,
    ``set_plan_disabled``), each one spawn of the conexus interpreter against the sandbox service.
    Refuses (``NxError``) before any spawn unless ``p2_sandbox_violations`` is empty; a failed spawn or a
    missing result line raises, never returns an empty answer. ``service_url`` is the URL the driver reported
    on its latest call (None before the first): this driver WRITES (it disables plans), so the run records it."""

    def __init__(self) -> None:
        self.service_url: str | None = None

    def _call(self, op: str, **args: Any) -> Any:
        bad = p2_sandbox_violations(os.environ, CONEXUS_PYTHON, _real_home())
        if bad:
            raise NxError("the plan library refuses to reach a nexus service: " + "; ".join(bad)
                          + ". Run it through benchmark/herb_sandbox.sh with NX_PYTHON/NX_BIN set to a sandbox "
                            "venv and HERB_NEXUS_REV set")
        if not CONEXUS_PYTHON.exists():
            raise NxError(f"nx interpreter not found at {CONEXUS_PYTHON}; set NX_PYTHON")
        try:
            proc = subprocess.run([str(CONEXUS_PYTHON), "-c", _PLAN_DRIVER, op, json.dumps(args)],
                                  capture_output=True, timeout=SEARCH_TIMEOUT_S, check=True)
        except subprocess.CalledProcessError as e:
            raise NxError(f"plan driver {op} failed (exit {e.returncode}): "
                          f"{(e.stderr or b'').decode(errors='replace')[-600:]}") from e
        except subprocess.TimeoutExpired as e:
            raise NxError(f"plan driver {op} timed out after {SEARCH_TIMEOUT_S}s") from e
        for line in proc.stdout.decode(errors="replace").splitlines():
            if line.startswith(_NXA_SENTINEL):
                payload = json.loads(line[len(_NXA_SENTINEL):])
                url = payload.get("service_url")
                if url and url != self.service_url:
                    self.service_url = url
                    err_console.print(f"plan service url: {url}")
                return payload["result"]
        raise NxError(f"plan driver {op} printed no result line")

    def list_plans(self, limit: int = 20, project: str = "", *, include_disabled: bool = False) -> list[dict[str, Any]]:
        return self._call("list_plans", limit=limit, project=project, include_disabled=include_disabled)

    def get_plan(self, plan_id: Any) -> dict[str, Any] | None:
        return self._call("get_plan", plan_id=plan_id)

    def set_plan_disabled(self, plan_id: Any, *, reason: str = "") -> bool:
        return bool(self._call("set_plan_disabled", plan_id=plan_id, reason=reason))


# --------------------------------------------------------------------------- #
# Scorecard assembly
# --------------------------------------------------------------------------- #


@dataclass
class Rung:
    phase: str
    metric: str
    value: float | None
    target: str  # human-readable target, e.g. "==1.0" or ">=script F1"
    status: str  # "RED" | "GREEN" | "BASELINE"


def _status(value: float | None, predicate: Callable[[float], bool]) -> str:
    return "GREEN" if value is not None and predicate(value) else "RED"


def _compare(measured: dict[str, float]) -> list[dict[str, Any]]:
    """Each prediction beside its measurement. ``asserted_exactly`` is True only for the deterministic
    table; an nx_answer row is recorded and never asserted exactly."""
    out = []
    for table, exact in ((PREDICTIONS, True), (NX_ANSWER_PREDICTIONS, False)):
        for name, pred in table.items():
            out.append({"metric": name, "predicted": pred["value"], "op": pred["op"],
                        "measured": measured.get(name), "asserted_exactly": exact})
    return out


_VERDICT_STATUS = {"passed": "GREEN", "failed": "RED", "invalid": "INVALID"}


def mvv_rungs(score: dict[str, Any]) -> list[dict[str, Any]]:
    """The scorecard rows of an MVV verdict (``score_mvv``): the two gates (both runs; GREEN, RED, or INVALID
    when the run was not a measurement), then what is reported only (the held-out count as entered of valid
    of asked, branch-entered counts, issue5 and agg2 per run label beside the script's F1)."""
    runs = score["runs"]
    means = [runs[lb]["precision"]["mean"] for lb in RUN_LABELS if lb in runs]
    held = score["heldout"]
    jc_v, pr_v = score["join_completeness_verdict"], score["precision_verdict"]
    read = "interpretable" if held["interpretable"] else "not interpretable: " + "; ".join(held["reasons"])
    rungs = [
        Rung("MVV", "join_completeness_both_runs", {"passed": 1.0, "failed": 0.0}.get(jc_v),
             f">={DOMAIN_FLOOR} of {ISSUE_QUESTIONS} issue questions in the domain, at most {MAX_DOMAIN_MISSES} "
             f"failing, both runs",
             _VERDICT_STATUS[jc_v]),
        Rung("MVV", "precision_mean_both_runs", min(means) if len(means) == len(RUN_LABELS) else None,
             f">={PRECISION_GATE} in both runs", _VERDICT_STATUS[pr_v]),
        Rung("MVV", "heldout_branch_entered", held["entered"] / held["n_valid"] if held["n_valid"] else None,
             f"report ({held['entered']} entered of {held['n_valid']} valid of {held['asked']} asked; {read})",
             "BASELINE"),
    ]
    for lb in RUN_LABELS:
        if lb not in runs:
            continue
        r = runs[lb]
        rungs += [
            Rung("MVV", f"issue_branch_entered[{lb}]", r["issue_entered"] / ISSUE_QUESTIONS, "report", "BASELINE"),
            Rung("MVV", f"issue5_recall[{lb}]", r["issue5_recall"], "report", "BASELINE"),
            Rung("MVV", f"issue5_f1[{lb}]", r["issue5_f1"], "report", "BASELINE"),
            Rung("MVV", f"agg2_recall[{lb}]", r["agg2"]["recall"], "report", "BASELINE"),
            Rung("MVV", f"agg2_precision[{lb}]", r["agg2"]["precision"], "report", "BASELINE"),
            Rung("MVV", f"agg2_f1[{lb}]", r["agg2"]["f1"], "report", "BASELINE"),
        ]
    return [asdict(x) for x in rungs]


def patch_scorecard_mvv(card: dict[str, Any], score: dict[str, Any], meta: dict[str, Any], file: str,
                        timestamp_utc: str) -> dict[str, Any]:
    """Replace the MVV rungs and the ``mvv`` section of *card* in place (adding them on the first run) and mark
    the manifest. Every other rung, the card's own manifest versions and ``manifest.patched_rungs`` of other
    rungs stay as they were measured. The per-row detail is in the dated file *file*, not in the scorecard."""
    card["rungs"] = [r for r in card["rungs"] if r.get("phase") != "MVV"] + mvv_rungs(score)
    card["mvv"] = {**score, "run": meta, "file": file}
    card["manifest"].setdefault("patched_rungs", {})["MVV"] = {
        "by": "--mvv", "file": file, "timestamp_utc": timestamp_utc, "verdict": score["verdict"],
        "nexus_revision": meta.get("nexus_revision"), "nx_version": meta.get("nx_version"),
        "engine_release_version": meta.get("engine_release_version"),
        "plan_service_url": meta.get("plan_service_url")}
    return card


def build(k: int = 20, nx_answer: bool = False, cap_usd: float = DEFAULT_CAP_USD,
          label: str = "run") -> tuple[dict, dict]:
    """Run every rung; return (scorecard, baselines). The MVV is not a rung of this build: ``emit_mvv`` runs
    it and patches only its own rungs into the scorecard, leaving the Step 0 baselines as measured."""
    gold = _load_gold()
    product = _load_product()
    questions = company_questions(product)
    universe = sorted(set(gold["cust_to_company"].values()))
    versions = service_versions()

    p1_recall, p1_prec = score_p1_link_recall_precision(gold["mentions"])
    p2_detail = score_p2_resolution(gold["members"])
    p2 = p2_detail["accuracy"]
    p3_rows = score_p3_detail(gold["q_to_custs"], k)
    p3 = sum(r["precision"] for r in p3_rows) / len(p3_rows) if p3_rows else 0.0

    script_rows = run_script_rows(questions, k)
    oracle_rows = run_oracle_rows(questions, product)
    oracle_retr_rows = run_oracle_retrieval_rows(questions)
    script = summarize(script_rows)
    oracle = summarize(oracle_rows)
    oracle_retr = summarize(oracle_retr_rows)

    nxa_rows: list[dict[str, Any]] | None = None
    nxa_meta: dict[str, Any] | None = None
    nxa: dict[str, dict[str, float | int]] | None = None
    if nx_answer:
        nxa_rows, nxa_meta = run_nx_answer_rows(questions, universe, cap_usd, label=label)
        nxa = summarize(nxa_rows)

    measured: dict[str, float] = {
        "script_fixed.recall.all7": script["all7"]["recall"],
        "script_fixed.recall.issue5": script["issue5"]["recall"],
        "script_fixed.recall.agg2": script["agg2"]["recall"],
        "p3_hop1_precision": p3,
        "p2_resolution_accuracy_29": p2,
        "oracle_cited.recall.issue5": oracle["issue5"]["recall"],
        "oracle_cited.precision.issue5": oracle["issue5"]["precision"],
        "oracle_cited.precision.agg2": oracle["agg2"]["precision"],
    }
    if nxa is not None:
        measured["nx_answer.recall.all7"] = nxa["all7"]["recall"]
        measured["nx_answer.precision.all7"] = nxa["all7"]["precision"]

    script_f1 = script["issue5"]["f1"]
    rungs = [
        Rung("P1", "mentions_link_recall", p1_recall, "==1.0", _status(p1_recall, lambda v: v == 1.0)),
        Rung("P1", "mentions_link_precision", p1_prec, ">=0.95", _status(p1_prec, lambda v: v >= 0.95)),
        p2_rung(p2_detail),
        Rung("P3", "hop1_precision", p3, ">=0.80", _status(p3, lambda v: v >= 0.80)),
        Rung("E2E-script", "company_recall_all7", script["all7"]["recall"], "baseline", "BASELINE"),
        Rung("E2E-script", "company_precision_all7", script["all7"]["precision"], "baseline", "BASELINE"),
        Rung("E2E-script", "company_f1_issue5", script_f1, "baseline", "BASELINE"),
    ]
    if nxa is not None:
        f1 = nxa["issue5"]["f1"]
        rungs += [
            Rung("E2E-nx_answer", "company_recall_all7", nxa["all7"]["recall"], "report", "BASELINE"),
            Rung("E2E-nx_answer", "company_precision_all7", nxa["all7"]["precision"], "report", "BASELINE"),
            Rung("E2E-nx_answer", "company_f1_issue5", f1, ">=script F1 (issue5)",
                 _status(f1, lambda v: v >= script_f1)),
        ]
    else:
        rungs.append(Rung("E2E-nx_answer", "company_f1_issue5", None, ">=script F1 (issue5)", "RED"))

    stamp = datetime.now(timezone.utc).isoformat()
    card = {
        "rdr": "RDR-147",
        "step": "0",
        "baseline_e2e_company_recall_june": _E2E_BASELINE,
        "k": k,
        "manifest": {"timestamp_utc": stamp, **versions,
                     "product": PRODUCT, "collection": CORPUS,
                     "sandbox": os.environ.get("HERB_SANDBOX") == "1"},
        "rungs": [asdict(r) for r in rungs],
        "p2_resolve": p2_resolve_entry(p2_detail, versions, stamp, None),
    }
    baselines = {
        "manifest": card["manifest"],
        "k": k,
        "comparison_predicted_vs_measured": _compare(measured),
        "baselines": {
            "script_fixed": {"summary": script, "rows": script_rows},
            "nx_answer": ({"summary": nxa, "run": nxa_meta, "rows": nxa_rows} if nxa is not None
                          else {"summary": None, "note": "not run (--nx-answer not given)"}),
            "oracle_cited_artifacts": {"summary": oracle, "rows": oracle_rows,
                                       "note": "hop 1 = the question's cited artifacts from the HERB source"},
            "oracle_retrieval_k300_filtered_to_cited": {"summary": oracle_retr, "rows": oracle_retr_rows},
        },
        "p2": {"n_members": len(gold["members"]), "accuracy": p2, "resolve": p2_detail},
        "p3": {"mean_precision": p3, "by_kind": p3_by_kind(p3_rows), "rows": p3_rows},
    }
    return card, baselines


def build_scorecard(k: int = 20, nx_answer: bool = False, cap_usd: float = DEFAULT_CAP_USD) -> dict:
    return build(k, nx_answer, cap_usd)[0]


def emit_scorecard(k: int = 20, nx_answer: bool = False, cap_usd: float = DEFAULT_CAP_USD,
                   label: str = "run") -> dict:
    card, baselines = build(k=k, nx_answer=nx_answer, cap_usd=cap_usd, label=label)
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _SCORECARD.write_text(json.dumps(card, indent=2))
    _BASELINES.write_text(json.dumps(baselines, indent=2))
    for r in card["rungs"]:
        colour = {"GREEN": "green", "RED": "red"}.get(r["status"], "yellow")
        val = "—" if r["value"] is None else f"{r['value']:.3f}"
        console.print(
            f"[{colour}]{r['status']:8}[/{colour}] {r['phase']:14} "
            f"{r['metric']:32} {val:>6}  (target {r['target']})"
        )
    for c in baselines["comparison_predicted_vs_measured"]:
        m = "—" if c["measured"] is None else f"{c['measured']:.3f}"
        console.print(f"  predicted {c['op']} {c['predicted']:.2f}  measured {m:>6}  {c['metric']}")
    console.print(f"\nscorecard written: {_SCORECARD}\nbaselines written: {_BASELINES}")
    return card


def valid_run_date(date: str) -> str:
    """*date* as a run id: ``YYYY-MM-DD`` and a real date, or ``NxError``. It names ``mvv-<date>.json(l)`` and keys
    the cost ledger, so nothing else (a path, a short date) may through."""
    if _RUN_DATE_RE.fullmatch(date) is None:
        raise NxError(f"--resume {date!r} is not a date: use YYYY-MM-DD")
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError as e:
        raise NxError(f"--resume {date!r} is not a date: use YYYY-MM-DD") from e
    return date


def emit_mvv(cap_usd: float = MVV_CAP_USD, date: str | None = None, resume: bool = False,
             plan_library: Any | None = None) -> dict[str, Any]:
    """The RDR-147 MVV, recorded on its own (nexus-bwct4.1 fix round 1).

    Refuses, before any spend, unless the resolution switch is on in the environment, the scorecard exists
    (this patches it), and the row file is consistent with *resume*. Rows land in
    ``results/herb/mvv-<date>.jsonl`` as they are made; with *resume* the valid rows of that file are kept
    and the cap is cumulative (the cost ledger is read back). At the end writes
    ``results/herb/mvv-<date>.json`` (rows, held-out rows, verdict, meta) and patches ONLY the MVV rungs and
    ``mvv`` section into ``herb-scorecard.json``; ``step0-baselines.json`` and every other rung are untouched
    (``tests/test_herb_ladder.py`` reads the Step 0 nx_answer record from it). No other rung is re-measured:
    the spend is the MVV's nx_answer calls only."""
    if date:
        valid_run_date(date)
    bad = mvv_switch_violations(os.environ) + mvv_llm_violations(os.environ)
    if bad:
        raise NxError("the MVV refuses to spend: " + "; ".join(bad))
    stamp = datetime.now(timezone.utc)
    date = date or stamp.strftime("%Y-%m-%d")
    sink = MvvRows(_RESULTS_DIR / f"mvv-{date}.jsonl")
    out_path = _RESULTS_DIR / f"mvv-{date}.json"
    if resume and not sink.exists():
        raise NxError(f"nothing to resume: {sink.path} does not exist")
    if not resume and sink.exists():
        raise NxError(f"{sink.path} exists: pass --resume to continue it (valid rows are kept and the cap is "
                      "cumulative), or move it aside to start a new run")
    if not _SCORECARD.exists():
        raise NxError(f"{_SCORECARD} does not exist: run the harness first (the MVV patches its rungs into it)")
    card = json.loads(_SCORECARD.read_text())
    gold, product = _load_gold(), _load_product()
    questions = company_questions(product)
    universe = sorted(set(gold["cust_to_company"].values()))
    heldout = load_heldout()
    versions = service_versions()
    library = plan_library if plan_library is not None else SandboxPlanLibrary()
    result = run_mvv(questions, universe, gold["cust_to_company"], heldout, cap_usd, library, run_id=date,
                     sink=sink, prior=sink.load() if resume else None)
    meta = {**result["meta"], "run_id": date, "timestamp_utc": stamp.isoformat(), "resumed": resume,
            "nexus_revision": os.environ.get("HERB_NEXUS_REV"), **versions,
            "plan_service_url": getattr(library, "service_url", None), "product": PRODUCT, "collection": CORPUS,
            "sandbox": os.environ.get("HERB_SANDBOX") == "1"}
    score = result["score"]
    rec = {"rdr": "RDR-147", "bead": "nexus-bwct4.1", "step": "mvv", "verdict": score["verdict"], "score": score,
           "rows": result["rows"], "heldout_rows": result["heldout_rows"], "meta": meta}
    # Patch first: a card that cannot take the section raises before either file is written.
    card = patch_scorecard_mvv(card, score, meta, out_path.name, meta["timestamp_utc"])
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rec, indent=2))
    _SCORECARD.write_text(json.dumps(card, indent=2))
    for r in card["rungs"]:
        if r["phase"] != "MVV":
            continue
        colour = {"GREEN": "green", "RED": "red", "INVALID": "magenta"}.get(r["status"], "yellow")
        val = "—" if r["value"] is None else f"{r['value']:.3f}"
        console.print(f"[{colour}]{r['status']:8}[/{colour}] {r['metric']:32} {val:>6}  (target {r['target']})")
    console.print(f"\nMVV verdict: {score['verdict']}  (spent {meta['total_cost_usd']} this invocation, "
                  f"{meta['cumulative_cost_usd']} of {cap_usd} cumulative)\nwritten: {out_path}\n"
                  f"rows: {sink.path}\nscorecard updated: {_SCORECARD}")
    return rec


def emit_p2_resolve(date: str | None = None,
                    resolver: Callable[[list[str]], dict[str, Any]] | None = None) -> dict[str, Any]:
    """Score ONLY the P2 rung through the ``resolve`` step and record it (no other rung runs).

    Writes ``results/herb/p2-resolve-<date>.json`` and patches the P2 rung, ``p2_resolve`` and
    ``manifest.patched_rungs.P2`` of the existing ``herb-scorecard.json`` (``patch_scorecard_p2``:
    exactly one P2 rung must exist). ``step0-baselines.json`` is not touched: it stays exactly as the
    full run measured it. Spend is 0: no embedding, no model call."""
    gold = _load_gold()
    detail = score_p2_resolution(gold["members"], resolver)
    versions = service_versions()
    stamp = datetime.now(timezone.utc)
    date = date or stamp.strftime("%Y-%m-%d")
    row = {"rdr": "RDR-147", "bead": "nexus-v9v5r", "step": "3", "rung": "P2", "via": "resolve_keys",
           "timestamp_utc": stamp.isoformat(), "product": PRODUCT, "collection": CORPUS,
           "sandbox": os.environ.get("HERB_SANDBOX") == "1", "spend_usd": 0.0, **versions, **detail}
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = _RESULTS_DIR / f"p2-resolve-{date}.json"
    # Patch first: a scorecard that does not hold exactly one P2 rung raises before either file is written.
    card = patch_scorecard_p2(json.loads(_SCORECARD.read_text()), detail, versions, row["timestamp_utc"], out.name)
    out.write_text(json.dumps(row, indent=2))
    _SCORECARD.write_text(json.dumps(card, indent=2))
    status = next(r["status"] for r in card["rungs"] if r["phase"] == "P2")
    console.print(f"P2 via resolve: accuracy {detail['accuracy']:.3f} ({detail['hits']}/{detail['n_tokens']}), "
                  f"missing {detail['missing']}, ambiguous {detail['ambiguous']}, {status}\n"
                  f"nexus {detail['nexus_revision']} (wheel {detail['nexus_dist_version']})\n"
                  f"written: {out}\nscorecard updated: {_SCORECARD}")
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="RDR-147 HERB benchmark ladder")
    ap.add_argument("--build-gold", action="store_true", help="rebuild gold fixtures and exit")
    ap.add_argument("-k", type=int, default=20, help="hop-1 retrieval depth")
    ap.add_argument("--nx-answer", action="store_true",
                    help="also run the nx_answer product rung (spends money; needs the claude CLI)")
    ap.add_argument("--cap-usd", type=float, default=None,
                    help=f"hard cap on total nx_answer spend (default {DEFAULT_CAP_USD}; {MVV_CAP_USD} with --mvv)")
    ap.add_argument("--mvv", action="store_true",
                    help="the RDR-147 MVV: every issue question twice (a, b) with grown plans disabled before each "
                         "run, then the frozen held-out set once; needs HERB_NEXUS_REV, the sandbox, the claude CLI "
                         "and NX_ANSWER_RESOLUTION_REPAIR=1 (spends money). Writes results/herb/mvv-<date>.json(l) and "
                         "patches only the MVV rungs into herb-scorecard.json")
    ap.add_argument("--resume", nargs="?", const="", default=None, metavar="DATE",
                    help="with --mvv: continue the run of DATE (default today): valid rows are kept, the rest rerun, "
                         "and the cap counts what the cost ledger already charged to that run")
    ap.add_argument("--freeze-heldout", action="store_true",
                    help="write benchmark/fixtures/herb/single_hop_heldout.json from ~/git/HERB and exit "
                         "(refuses when it exists: the set is frozen)")
    ap.add_argument("--copy-scorecard-to", metavar="DIR",
                    help="copy results/herb/herb-scorecard.json into DIR (nexus docs/rdr/evidence/rdr-147) and exit")
    ap.add_argument("--p2-resolve", action="store_true",
                    help="score only the P2 rung through the resolve step; write results/herb/p2-resolve-<date>.json "
                         "and update the P2 row of herb-scorecard.json (spend 0)")
    ap.add_argument("--label", default="run", help="name of this run in the nx_answer cost ledger")
    args = ap.parse_args()
    if args.build_gold:
        build_gold_fixtures()
        return
    if args.p2_resolve:
        emit_p2_resolve()
        return
    if args.freeze_heldout:
        freeze_heldout()
        return
    if args.copy_scorecard_to:
        console.print(f"copied: {copy_scorecard(Path(args.copy_scorecard_to))}")
        return
    if args.resume is not None and not args.mvv:
        ap.error("--resume continues an MVV run: give it with --mvv")
    if args.mvv and args.nx_answer:
        ap.error("--mvv is its own run (two labelled runs, the held-out set): do not combine it with --nx-answer")
    cap = args.cap_usd if args.cap_usd is not None else (MVV_CAP_USD if args.mvv else DEFAULT_CAP_USD)
    if args.resume:
        try:
            valid_run_date(args.resume)
        except NxError as e:
            ap.error(str(e))
    if args.mvv:
        emit_mvv(cap_usd=cap, date=args.resume or None, resume=args.resume is not None)
        return
    emit_scorecard(k=args.k, nx_answer=args.nx_answer, cap_usd=cap, label=args.label)


if __name__ == "__main__":
    try:
        main()
    except NxError as e:
        raise SystemExit(f"nx error: {e}")
