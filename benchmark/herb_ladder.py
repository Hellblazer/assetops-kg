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
every grown plan soft-disabled before each run, then scores join completeness and precision, and
counts how often the resolution branch fires on a frozen single-hop held-out set. The rules are in
``benchmark/README-herb.md`` (section "The RDR-147 MVV").

Run inside the isolated sandbox (see herb_sandbox.sh):
    ./benchmark/herb_sandbox.sh run python3 -m benchmark.herb_ladder --build-gold
    ./benchmark/herb_sandbox.sh run python3 -m benchmark.herb_ladder
    # with the nx_answer product rung (spends money; needs the claude CLI token):
    HERB_SANDBOX_LLM=1 python3 <nexus>/tests/e2e/lib/claude_credentials.py run -- \\
        ./benchmark/herb_sandbox.sh run python3 -m benchmark.herb_ladder --nx-answer --cap-usd 2.0
    # the MVV: two labelled runs per question + the held-out set (spends, cap 4 USD):
    HERB_NEXUS_REV=<sha> HERB_SANDBOX_LLM=1 python3 <nexus>/tests/e2e/lib/claude_credentials.py run -- \\
        ./benchmark/herb_sandbox.sh run python3 -m benchmark.herb_ladder --mvv
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import pwd
import re
import subprocess
import textwrap
import time
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
PRECISION_GATE = 0.95            # per-run mean of per-question precision, in both runs
HELDOUT_LIMIT = 20               # the single-hop held-out set: the first 20 non-company SearchFlow questions
MVV_CAP_USD = 4.0                # spend cap of the MVV run
# The script's issue5 F1 (step0-baselines.json): the diagnostic baseline the nx_answer F1 is reported beside.
# A deterministic rung, asserted exactly by the paired pytest.
SCRIPT_ISSUE5_F1 = 0.7642857142857142
GROWN_PLAN_REASON = "herb-mvv: a plan grown from one question captures later ones"
PLAN_SWEEP_LIMIT = 1000          # plans listed per sweep; a list that fills this is refused as possibly partial
#: The RepairReport fields copied into every nx_answer row (nexus-qpimy adds them to the report).
REPAIR_FIELDS = ("resolution_entered", "anchor_key_ids", "resolved", "missing", "ambiguous", "joined_dropped")

DEFAULT_CAP_USD = 2.0


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


def _append_cost_ledger(label: str, question: str, cost: float | None) -> None:
    """Append-only record of every nx_answer call's recorded spend."""
    _COST_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with _COST_LEDGER.open("a") as f:
        f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "run": label,
                            "cost_usd": cost, "question": question[:90]}) + "\n")


# --------------------------------------------------------------------------- #
# The RDR-147 MVV: grown plans, the RepairReport columns, two runs per question
# --------------------------------------------------------------------------- #
#
# A plan grown from one question captures later questions (RDR-147 research 9: 2 of 10 spike runs ran
# on grown plan 14). Before EVERY nx_answer run every grown plan in the sandbox is soft-disabled
# (HttpPlanLibrary.set_plan_disabled), and a run whose reported plan is a grown plan is invalid: it is
# recorded and never counted.


def plan_is_grown(plan: Mapping[str, Any] | None) -> bool:
    """True unless the plan row is a curated template. Fails closed, as nexus's own eligibility does
    (``nexus.plans.answer_kind``): a plan is a template only when its tags carry ``builtin-template`` and
    not ``grown``; a plan with no tags, no row, or the ``grown`` tag (the nx_answer grow site writes
    ``ad-hoc,grown``) is grown."""
    if plan is None:
        return True
    tags = {t.strip() for t in str(plan.get("tags") or "").split(",")}
    return "grown" in tags or "builtin-template" not in tags


def sweep_grown_plans(library: Any, reason: str = GROWN_PLAN_REASON) -> list[int]:
    """Soft-disable every active grown plan, once each; return their ids.

    *library* is an ``HttpPlanLibrary`` (or the sandbox adapter): ``list_plans`` lists the ACTIVE plans
    (an already-disabled plan is not listed, so it is not disabled again). ``NxError`` when a plan will
    not disable, or when the list filled ``PLAN_SWEEP_LIMIT`` (it may be partial, and a grown plan past
    the cut would still match): a run never starts on a sweep that may have missed one."""
    rows = library.list_plans(limit=PLAN_SWEEP_LIMIT, include_disabled=False)
    if len(rows) >= PLAN_SWEEP_LIMIT:
        raise NxError(f"plan list reached its limit of {PLAN_SWEEP_LIMIT}: a grown plan may have been missed")
    grown = [int(r["id"]) for r in rows if plan_is_grown(r)]
    for pid in grown:
        if not library.set_plan_disabled(pid, reason=reason):
            raise NxError(f"plan {pid} did not disable: a run on it would not be a measurement")
    return grown


def plan_validity(library: Any, plan_id: Any) -> tuple[bool, str]:
    """(valid, reason) for a run that reports *plan_id*. No plan (the inline planner answered) is valid;
    a grown plan, or a plan row that cannot be read, is not."""
    if plan_id is None:
        return True, ""
    row = library.get_plan(plan_id)
    if row is None:
        return False, f"plan row {plan_id} not found: the run cannot be shown to be off a grown plan"
    if plan_is_grown(row):
        return False, f"run on grown plan {plan_id}"
    return True, ""


def repair_columns(report: Any) -> dict[str, Any]:
    """The RepairReport fields an nx_answer row carries, read tolerantly: nexus's Step 4 adds them to the
    report, and a report without them (an older nexus, or no repair) is a branch that did not enter.
    ``resolution_entered`` is True only for a boolean True; the others are copied when present."""
    rep = report if isinstance(report, Mapping) else {}

    def listy(v: Any) -> list[Any]:
        return list(v) if isinstance(v, (list, tuple, set, frozenset)) else []

    def keep(v: Any) -> Any:
        if isinstance(v, Mapping):
            return dict(v)
        return listy(v) if isinstance(v, (list, tuple, set, frozenset)) else []

    dropped = rep.get("joined_dropped")
    if isinstance(dropped, (list, tuple, set, frozenset)):
        dropped = list(dropped)             # a count, or the dropped items themselves: copied as given
    elif not isinstance(dropped, int) or isinstance(dropped, bool):
        dropped = 0
    return {
        "resolution_entered": rep.get("resolution_entered") is True,
        "anchor_key_ids": [str(x) for x in listy(rep.get("anchor_key_ids"))],
        "resolved": keep(rep.get("resolved")),
        "missing": keep(rep.get("missing")),
        "ambiguous": keep(rep.get("ambiguous")),
        "joined_dropped": dropped,
    }


def _guarded_nx_answer(question: str, plan_library: Any | None, budget: float, max_steps: int,
                       timeout: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """One nx_answer run behind the grown-plan sweep. Returns (envelope, the columns every row carries:
    plan, validity, plans disabled before the run, the RepairReport fields). Without a plan library the
    run is recorded but never valid: nothing showed it was off a grown plan."""
    disabled = sweep_grown_plans(plan_library) if plan_library is not None else []
    env = nx_answer_call(question, corpus_arg(PRODUCT), max_steps, timeout, budget_usd=budget)
    if plan_library is not None:
        valid, why = plan_validity(plan_library, env.get("plan_id"))
    else:
        valid, why = False, "plan not checked: no plan library, grown plans were not disabled"
    cols = {"plan_id": env.get("plan_id"), "valid": valid, "invalid_reason": why,
            "plans_disabled_before": disabled, **repair_columns(env.get("repair"))}
    return env, cols


def run_nx_answer_rows(
    questions: list[dict[str, Any]],
    universe: list[str],
    cap_usd: float,
    max_steps: int = 6,
    timeout: int = 420,
    label: str = "run",
    plan_library: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Baseline 2 (the product): nx_answer over the corpus, scored on the companies named in its
    answer, recall AND precision. Spend is summed from each run's recorded cost_usd; once the cap
    is reached the remaining questions are not run and the baseline is marked incomplete.

    With *plan_library* every grown plan is disabled before each run and each row is marked valid or
    invalid (``valid``, ``invalid_reason``); without one no row is valid. Every row carries its run
    ``label`` and the RepairReport columns (``REPAIR_FIELDS``)."""
    rows: list[dict[str, Any]] = []
    spent = 0.0
    unknown_cost = 0
    for q in questions:
        remaining = cap_usd - spent
        if remaining <= 0.02:
            rows.append(_row(q, set(), skipped="cost cap reached", cost_usd=0.0, run_label=label,
                             valid=False, invalid_reason="skipped: cost cap reached"))
            continue
        env, cols = _guarded_nx_answer(q["question"], plan_library, min(0.5, remaining), max_steps, timeout)
        cost = env.get("cost_usd")
        if cost is None:
            unknown_cost += 1
        else:
            spent += float(cost)
        pred = extract_companies(env.get("final_text") or "", universe)
        ev_ids = sorted({cid for t in env.get("evidence_texts") or [] for cid in _RE_CUST.findall(t or "")})
        rows.append(_row(q, pred, final_text=env.get("final_text"), run_label=label,
                         step_count=env.get("step_count"), cost_usd=cost,
                         coverage_state=env.get("coverage_state"), coverage_reason=env.get("coverage_reason"),
                         n_evidence_chunks=len(env.get("chunks") or []),
                         evidence_cust_ids=ev_ids,
                         evidence_companies=sorted(companies_for_ids(ev_ids)), **cols))
        _append_cost_ledger(label, q["question"], cost)
        console.print(f"  nx_answer[{label}] cost {cost} (total {spent:.4f}) recall={rows[-1]['recall']:.2f} "
                      f"precision={rows[-1]['precision']:.2f} valid={rows[-1]['valid']} "
                      f"entered={rows[-1]['resolution_entered']}  {q['question'][:50]}")
    meta = {
        "total_cost_usd": round(spent, 6),
        "cap_usd": cap_usd,
        "runs_with_unknown_cost": unknown_cost,
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
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Every question twice, run ``a`` then run ``b``, under ONE spend cap: ``b`` gets what ``a`` left."""
    rows: dict[str, list[dict[str, Any]]] = {}
    by_run: dict[str, dict[str, Any]] = {}
    spent = 0.0
    for label in RUN_LABELS:
        rows[label], by_run[label] = run_nx_answer_rows(
            questions, universe, cap_usd - spent, max_steps, timeout, label=label, plan_library=plan_library)
        spent += by_run[label]["total_cost_usd"]
    meta = {"total_cost_usd": round(spent, 6), "cap_usd": cap_usd,
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
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The single-hop held-out questions, each run ONCE with the resolution switch on (the sandbox's
    config), behind the same grown-plan sweep. Rows record the plan, validity and RepairReport columns;
    they are not scored against a gold (the count is of runs where the branch entered)."""
    rows: list[dict[str, Any]] = []
    spent = 0.0
    unknown_cost = 0
    for q in heldout:
        base = {"herb_index": q["herb_index"], "question": q["question"], "type": q.get("type"), "run_label": label}
        remaining = cap_usd - spent
        if remaining <= 0.02:
            rows.append({**base, "skipped": "cost cap reached", "cost_usd": 0.0, "valid": False,
                         "invalid_reason": "skipped: cost cap reached", **repair_columns(None)})
            continue
        env, cols = _guarded_nx_answer(q["question"], plan_library, min(0.5, remaining), max_steps, timeout)
        cost = env.get("cost_usd")
        if cost is None:
            unknown_cost += 1
        else:
            spent += float(cost)
        rows.append({**base, "final_text": env.get("final_text"), "step_count": env.get("step_count"),
                     "cost_usd": cost, "coverage_state": env.get("coverage_state"),
                     "coverage_reason": env.get("coverage_reason"), **cols})
        _append_cost_ledger(label, q["question"], cost)
        console.print(f"  heldout cost {cost} (total {spent:.4f}) valid={cols['valid']} "
                      f"entered={cols['resolution_entered']}  {q['question'][:50]}")
    meta = {"total_cost_usd": round(spent, 6), "cap_usd": cap_usd, "runs_with_unknown_cost": unknown_cost,
            "complete": all("skipped" not in r for r in rows)}
    return rows, meta


def _usable(row: Mapping[str, Any]) -> bool:
    """A row that counts: it ran, and its plan was shown not to be a grown one."""
    return row.get("valid") is True and not row.get("skipped")


def join_completeness(rows: list[dict[str, Any]], cust_to_company: Mapping[str, str], *,
                      floor: int = DOMAIN_FLOOR, n_issue: int = ISSUE_QUESTIONS) -> dict[str, Any]:
    """Join completeness of ONE run, gated (RDR-147 MVV).

    The domain is the issue questions whose ``anchor_key_ids`` is non-empty in this run: the branch scans
    the boundary's anchors, not all the evidence. A question passes when it is in the domain, its answer
    names every company its anchor ids give (through the registry), and ``resolution_entered`` is true
    (a pass through the semantic repair alone does not count). The run passes when ``n_issue`` issue rows
    all ran validly, at least ``floor`` are in the domain (fewer is a vacuous run, a fail), and every
    in-domain question passes. Completeness over ALL hydrated evidence ids is reported, never gated."""
    issue = [r for r in rows if r.get("kind") == "issue"]
    questions = []
    for r in issue:
        anchors = [a for a in (r.get("anchor_key_ids") or [])]
        in_domain = _usable(r) and bool(anchors)
        companies = sorted({cust_to_company[a] for a in anchors if a in cust_to_company})
        named = {c.lower() for c in r.get("pred", [])}
        missing = sorted(c for c in companies if c.lower() not in named)
        complete = bool(companies) and not missing
        entered = r.get("resolution_entered") is True
        if not _usable(r):
            reason = r.get("invalid_reason") or "row did not run"
        elif not anchors:
            reason = "no anchor ids: outside the domain"
        elif not companies:
            reason = "anchor ids not in the registry: cannot be graded"
        elif missing:
            reason = f"answer does not name {missing}"
        elif not entered:
            reason = "resolution branch did not enter"
        else:
            reason = ""
        questions.append({"question": r.get("question"), "in_domain": in_domain, "valid": _usable(r),
                          "resolution_entered": entered, "anchor_key_ids": anchors,
                          "anchor_companies": companies, "named": sorted(named), "missing_companies": missing,
                          "complete": complete, "passed": in_domain and complete and entered, "reason": reason})
    n_in = sum(q["in_domain"] for q in questions)
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
    passed = (len(issue) == n_issue and n_invalid == 0 and not vacuous
              and all(q["passed"] for q in questions if q["in_domain"]))
    if len(issue) != n_issue:
        reason = f"{len(issue)} issue rows, expected {n_issue}"
    elif n_invalid:
        reason = f"{n_invalid} row(s) invalid or skipped: the run is not a measurement"
    elif vacuous:
        reason = f"vacuous: {n_in} of {len(issue)} issue questions in the domain, floor {floor}"
    else:
        reason = "" if passed else "an in-domain question failed"
    return {"n_issue": len(issue), "in_domain": n_in, "floor": floor, "vacuous": vacuous, "n_invalid": n_invalid,
            "passed": passed, "reason": reason, "questions": questions,
            "evidence_completeness": {"n": len(ev), "n_complete": sum(e["complete"] for e in ev),
                                      "n_with_evidence": sum(1 for e in ev if e["n_evidence_ids"]),
                                      "questions": ev, "gated": False}}


def precision_gate(rows: list[dict[str, Any]], threshold: float = PRECISION_GATE,
                   n_issue: int = ISSUE_QUESTIONS) -> dict[str, Any]:
    """Precision of ONE run, gated: the MEAN over the issue questions of per-question precision
    (``score_set``), at least *threshold*. This is the aggregation that gave 0.9667 in 147-research-9.
    The pooled ratio and the per-question values are reported beside it. The run passes only with
    ``n_issue`` issue rows that all ran validly."""
    issue = [r for r in rows if r.get("kind") == "issue"]
    valid = [r for r in issue if _usable(r)]
    per = [float(r["precision"]) for r in valid]
    mean = sum(per) / len(per) if per else 0.0
    n_pred = sum(r["n_pred"] for r in valid)
    pooled = sum(r["hit"] for r in valid) / n_pred if n_pred else 0.0
    return {"mean": mean, "pooled": pooled, "per_question": per, "n_issue": len(issue), "n_valid": len(valid),
            "threshold": threshold,
            "passed": len(issue) == n_issue and len(valid) == n_issue and mean >= threshold - 1e-9}


def score_mvv(rows_by_label: Mapping[str, list[dict[str, Any]]], cust_to_company: Mapping[str, str],
              heldout_rows: list[dict[str, Any]], script_f1: float = SCRIPT_ISSUE5_F1) -> dict[str, Any]:
    """The MVV verdict from the two labelled runs and the held-out runs.

    Gated: join completeness and precision, each passing in BOTH runs (a missing run fails). Reported
    only: issue5 recall/precision/F1 beside the script's F1, agg2 recall/precision/F1, per-run counts of
    issue questions where the branch entered, and how many held-out runs entered it. Invalid rows (a run
    on a grown plan) are excluded from every figure and counted."""
    runs: dict[str, Any] = {}
    for label, rows in rows_by_label.items():
        usable = [r for r in rows if _usable(r)]
        summ = summarize(usable)
        jc, pg = join_completeness(rows, cust_to_company), precision_gate(rows)
        runs[label] = {
            "run_label": label, "join_completeness": jc, "precision": pg,
            "n_invalid": sum(not _usable(r) for r in rows),
            "issue_entered": sum(1 for r in usable if r.get("kind") == "issue" and r.get("resolution_entered") is True),
            "summary": summ, "issue5_recall": summ["issue5"]["recall"], "issue5_precision": summ["issue5"]["precision"],
            "issue5_f1": summ["issue5"]["f1"], "agg2": summ["agg2"], "passed": jc["passed"] and pg["passed"]}
    jc_ok = all(l in runs and runs[l]["join_completeness"]["passed"] for l in RUN_LABELS)
    pr_ok = all(l in runs and runs[l]["precision"]["passed"] for l in RUN_LABELS)
    held_valid = [r for r in heldout_rows if _usable(r)]
    return {"labels": list(RUN_LABELS), "runs": runs, "join_completeness_passed": jc_ok,
            "precision_passed": pr_ok, "passed": jc_ok and pr_ok, "script_issue5_f1": script_f1,
            "heldout": {"n": len(heldout_rows), "n_valid": len(held_valid),
                        "n_invalid": len(heldout_rows) - len(held_valid),
                        "entered": sum(1 for r in held_valid if r.get("resolution_entered") is True)}}


def run_mvv(questions: list[dict[str, Any]], universe: list[str], cust_to_company: Mapping[str, str],
            heldout: list[dict[str, Any]], cap_usd: float, plan_library: Any) -> dict[str, Any]:
    """Both labelled runs, then the held-out set, under one cap; return rows, held-out rows, meta, verdict."""
    rows, meta = run_mvv_nx_answer(questions, universe, cap_usd, plan_library)
    held_rows, held_meta = run_heldout_rows(heldout, cap_usd - meta["total_cost_usd"], plan_library)
    total = round(meta["total_cost_usd"] + held_meta["total_cost_usd"], 6)
    return {"rows": rows, "heldout_rows": held_rows, "score": score_mvv(rows, cust_to_company, held_rows),
            "meta": {"cap_usd": cap_usd, "total_cost_usd": total, "nx_answer": meta, "heldout": held_meta,
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
    if urlparse(url).hostname not in ("127.0.0.1", "localhost", "::1"):
        raise SystemExit("refusing non-loopback service url " + url)
    print("@@SENTINEL@@" + json.dumps({"result": getattr(lib, op)(**args)}, default=str))
    """
).replace("@@GUARD@@", textwrap.dedent(inspect.getsource(p2_sandbox_violations))).replace("@@SENTINEL@@", _NXA_SENTINEL)


class SandboxPlanLibrary:
    """The three ``HttpPlanLibrary`` calls the MVV uses (``list_plans``, ``get_plan``,
    ``set_plan_disabled``), each one spawn of the conexus interpreter against the sandbox service.
    Refuses (``NxError``) before any spawn unless ``p2_sandbox_violations`` is empty; a failed spawn or a
    missing result line raises, never returns an empty answer."""

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
                return json.loads(line[len(_NXA_SENTINEL):])["result"]
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


def mvv_rungs(score: dict[str, Any]) -> list[dict[str, Any]]:
    """The scorecard rows of an MVV verdict (``score_mvv``): the two gates (both runs), then what is
    reported only (branch-entered counts, issue5 and agg2 per run label beside the script's F1)."""
    runs = score["runs"]
    means = [runs[lb]["precision"]["mean"] for lb in RUN_LABELS if lb in runs]
    held = score["heldout"]
    rungs = [
        Rung("MVV", "join_completeness_both_runs", 1.0 if score["join_completeness_passed"] else 0.0,
             f">={DOMAIN_FLOOR} of {ISSUE_QUESTIONS} issue questions in the domain and every one passes, both runs",
             "GREEN" if score["join_completeness_passed"] else "RED"),
        Rung("MVV", "precision_mean_both_runs", min(means) if len(means) == len(RUN_LABELS) else None,
             f">={PRECISION_GATE} in both runs", "GREEN" if score["precision_passed"] else "RED"),
        Rung("MVV", "heldout_branch_entered", held["entered"] / held["n_valid"] if held["n_valid"] else None,
             "report", "BASELINE"),
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


def build(k: int = 20, nx_answer: bool = False, cap_usd: float = DEFAULT_CAP_USD,
          label: str = "run", mvv: bool = False, plan_library: Any | None = None) -> tuple[dict, dict]:
    """Run every rung; return (scorecard, baselines). With *mvv* the nx_answer rung is the MVV: two
    labelled runs per question and the held-out set, behind the grown-plan sweep of *plan_library*
    (the sandbox adapter by default); ``nx_answer`` is then not run a third time."""
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
    mvv_out: dict[str, Any] | None = None
    if mvv:
        mvv_out = run_mvv(questions, universe, gold["cust_to_company"], load_heldout(), cap_usd,
                          plan_library if plan_library is not None else SandboxPlanLibrary())
    elif nx_answer:
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
    elif mvv_out is None:
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
    if mvv_out is not None:
        card["rungs"] += mvv_rungs(mvv_out["score"])
        card["mvv"] = {**mvv_out["score"], "run": mvv_out["meta"], "rows": mvv_out["rows"],
                       "heldout_rows": mvv_out["heldout_rows"]}
    baselines = {
        "manifest": card["manifest"],
        "k": k,
        "comparison_predicted_vs_measured": _compare(measured),
        "baselines": {
            "script_fixed": {"summary": script, "rows": script_rows},
            "nx_answer": ({"summary": nxa, "run": nxa_meta, "rows": nxa_rows} if nxa is not None
                          else {"summary": None,
                                "note": "see nx_answer_mvv (two labelled runs)" if mvv_out is not None
                                else "not run (--nx-answer not given)"}),
            "oracle_cited_artifacts": {"summary": oracle, "rows": oracle_rows,
                                       "note": "hop 1 = the question's cited artifacts from the HERB source"},
            "oracle_retrieval_k300_filtered_to_cited": {"summary": oracle_retr, "rows": oracle_retr_rows},
        },
        "p2": {"n_members": len(gold["members"]), "accuracy": p2, "resolve": p2_detail},
        "p3": {"mean_precision": p3, "by_kind": p3_by_kind(p3_rows), "rows": p3_rows},
    }
    if mvv_out is not None:
        baselines["baselines"]["nx_answer_mvv"] = {
            "score": mvv_out["score"], "run": mvv_out["meta"], "rows": mvv_out["rows"],
            "heldout_rows": mvv_out["heldout_rows"],
            "note": "nx_answer values vary per run: recorded with their run label, never asserted exactly"}
    return card, baselines


def build_scorecard(k: int = 20, nx_answer: bool = False, cap_usd: float = DEFAULT_CAP_USD) -> dict:
    return build(k, nx_answer, cap_usd)[0]


def emit_scorecard(k: int = 20, nx_answer: bool = False, cap_usd: float = DEFAULT_CAP_USD,
                   label: str = "run", mvv: bool = False) -> dict:
    card, baselines = build(k=k, nx_answer=nx_answer, cap_usd=cap_usd, label=label, mvv=mvv)
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
                         "run, then the frozen held-out set once; needs HERB_NEXUS_REV, the sandbox and the claude CLI "
                         "(spends money)")
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
    cap = args.cap_usd if args.cap_usd is not None else (MVV_CAP_USD if args.mvv else DEFAULT_CAP_USD)
    emit_scorecard(k=args.k, nx_answer=args.nx_answer, cap_usd=cap, label=args.label, mvv=args.mvv)


if __name__ == "__main__":
    try:
        main()
    except NxError as e:
        raise SystemExit(f"nx error: {e}")
