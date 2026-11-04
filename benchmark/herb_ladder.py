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

Every metric is asserted EXACT by the paired pytest. Until the corresponding nexus
feature exists the metric computes to its RED baseline; the harness is the oracle, not
the implementation.

Run inside the isolated sandbox (see herb_sandbox.sh):
    ./benchmark/herb_sandbox.sh run python3 -m benchmark.herb_ladder --build-gold
    ./benchmark/herb_sandbox.sh run python3 -m benchmark.herb_ladder
    # with the nx_answer product rung (spends money; needs the claude CLI token):
    HERB_SANDBOX_LLM=1 python3 <nexus>/tests/e2e/lib/claude_credentials.py run -- \\
        ./benchmark/herb_sandbox.sh run python3 -m benchmark.herb_ladder --nx-answer --cap-usd 2.0
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
_RESULTS_DIR = Path(__file__).parent.parent / "results" / "herb"
_SCORECARD = _RESULTS_DIR / "herb-scorecard.json"
_BASELINES = _RESULTS_DIR / "step0-baselines.json"

PRODUCT = "SearchFlow"

# The E2E baseline the Problem Statement declared broken (single-hop nx_answer, June).
_E2E_BASELINE = 0.000

# Predictions written down BEFORE the first measured run (RDR-147 re-plan memo
# plan-rdr147-replan-2026-11-02 section 4, k=20, bge-768). op "==" is a point estimate,
# "<" an upper bound. Embedded beside the measured value in step0-baselines.json.
PREDICTIONS: dict[str, dict[str, Any]] = {
    "script_fixed.recall.all7": {"value": 0.40, "op": "=="},
    "script_fixed.recall.issue5": {"value": 0.52, "op": "=="},
    "script_fixed.recall.agg2": {"value": 0.10, "op": "=="},
    "p3_hop1_precision": {"value": 0.55, "op": "=="},
    "p2_resolution_accuracy_29": {"value": 1.0, "op": "=="},
    "nx_answer.recall.all7": {"value": 0.10, "op": "=="},
    "nx_answer.precision.all7": {"value": 0.30, "op": "<"},
    "oracle_cited.recall.issue5": {"value": 1.0, "op": "=="},
    "oracle_cited.precision.issue5": {"value": 0.61, "op": "=="},
    "oracle_cited.precision.agg2": {"value": 0.10, "op": "=="},
}

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
    own service config), and HERB_NEXUS_REV must name the nexus revision being measured."""
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


def run_nx_answer_rows(
    questions: list[dict[str, Any]],
    universe: list[str],
    cap_usd: float,
    max_steps: int = 6,
    timeout: int = 420,
    label: str = "run",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Baseline 2 (the product): nx_answer over the corpus, scored on the companies named in its
    answer, recall AND precision. Spend is summed from each run's recorded cost_usd; once the cap
    is reached the remaining questions are not run and the baseline is marked incomplete."""
    rows: list[dict[str, Any]] = []
    spent = 0.0
    unknown_cost = 0
    for q in questions:
        remaining = cap_usd - spent
        if remaining <= 0.02:
            rows.append(_row(q, set(), skipped="cost cap reached", cost_usd=0.0))
            continue
        env = nx_answer_call(q["question"], corpus_arg(PRODUCT), max_steps, timeout,
                             budget_usd=min(0.5, remaining))
        cost = env.get("cost_usd")
        if cost is None:
            unknown_cost += 1
        else:
            spent += float(cost)
        pred = extract_companies(env.get("final_text") or "", universe)
        ev_ids = sorted({cid for t in env.get("evidence_texts") or [] for cid in _RE_CUST.findall(t or "")})
        rows.append(_row(q, pred, final_text=env.get("final_text"), plan_id=env.get("plan_id"),
                         step_count=env.get("step_count"), cost_usd=cost,
                         coverage_state=env.get("coverage_state"), coverage_reason=env.get("coverage_reason"),
                         n_evidence_chunks=len(env.get("chunks") or []),
                         evidence_cust_ids=ev_ids,
                         evidence_companies=sorted(companies_for_ids(ev_ids))))
        _append_cost_ledger(label, q["question"], cost)
        console.print(f"  nx_answer cost {cost} (total {spent:.4f}) recall={rows[-1]['recall']:.2f} "
                      f"precision={rows[-1]['precision']:.2f}  {q['question'][:50]}")
    meta = {
        "total_cost_usd": round(spent, 6),
        "cap_usd": cap_usd,
        "runs_with_unknown_cost": unknown_cost,
        "complete": all("skipped" not in r for r in rows),
    }
    return rows, meta


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
    out = []
    for name, pred in PREDICTIONS.items():
        m = measured.get(name)
        out.append({"metric": name, "predicted": pred["value"], "op": pred["op"], "measured": m})
    return out


def build(k: int = 20, nx_answer: bool = False, cap_usd: float = DEFAULT_CAP_USD,
          label: str = "run") -> tuple[dict, dict]:
    """Run every rung; return (scorecard, baselines)."""
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
    out.write_text(json.dumps(row, indent=2))
    card = patch_scorecard_p2(json.loads(_SCORECARD.read_text()), detail, versions, row["timestamp_utc"], out.name)
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
    ap.add_argument("--cap-usd", type=float, default=DEFAULT_CAP_USD, help="hard cap on total nx_answer spend")
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
    emit_scorecard(k=args.k, nx_answer=args.nx_answer, cap_usd=args.cap_usd, label=args.label)


if __name__ == "__main__":
    try:
        main()
    except NxError as e:
        raise SystemExit(f"nx error: {e}")
