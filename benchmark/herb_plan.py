"""Part 1 of the HERB plan experiment: a hand-written, generic aggregation plan for nx_answer.

Design of record: T2 ``nexus_rdr/herb-plan-experiment-prereg-2026-11-19`` and
``herb-plan-experiment-prereg-addendum-1-2026-11-19`` (binding). One plan template, built only from operators
the plan runner dispatches today (search, store_get_many, groupby, aggregate, generate), saved to the sandbox's
plan library only. Retrieval is a lexical, record-shaped query assembled from the question's scope value and the
entity-key prefix of the ``entity.collections`` config; the records are hydrated, grouped by the customer key,
counted, and the generator maps keys to companies from the entity records and names the maximum.

``nx_answer`` has no plan-id or plan-body argument at the pinned rev (dc0244253). The plan is pinned through
two documented arguments instead: ``dimensions={"verb": PLAN_VERB}`` (a hard filter in the plan matcher, and
PLAN_VERB belongs to this plan alone) and ``min_confidence=0`` (the cosine floor admits it). The driver also sets
``NX_ANSWER_KIND_ELIGIBILITY=0`` for these calls (the answer-kind gate runs in shadow), so a request the extractor
labels another kind cannot drop the pinned plan. Every routed row records whether the envelope's ``plan_id`` is the
pinned plan; a row where it is not is a pin miss and is counted.

Nothing here reads HERB gold; ``run_plan_answers`` receives the question rows from ``run_herb_pool`` and sends
only the question text.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark import run_herb_pool as hp

#: The prereg's split: plan authoring and iteration look at these products only.
DEV_PRODUCTS: frozenset[str] = frozenset(
    {"SearchFlow", "ActionGenie", "AnomalyForce", "AutoTuneForce", "CoachForce"})

#: Belongs to this plan alone; the verb is the pin (see the module docstring).
PLAN_VERB = "herbagg"
#: The plan row's ``query`` column: a generic description, never a HERB question.
PLAN_QUERY = ("Aggregate the records of one scope by their entity key, map each key to its company, "
              "count per company and name the maximum")
PLAN_TAGS = "herb-plan-experiment,part1,aggregate"

# --------------------------------------------------------------------------- routing rule

#: The shape rule, from the question template wording only (addendum 1): aggregate = "maximum number of" /
#: "most"; previous-release = "previous release". Aggregate wins when both appear.
_AGG_RE = re.compile(r"\b(?:maximum number of|most)\b", re.IGNORECASE)
_PREV_RE = re.compile(r"\bprevious release\b", re.IGNORECASE)
_FACET_RE = re.compile(r"\b(?:maximum number of|most)\s+(\w+)\s+(?:bugs|issues)\b", re.IGNORECASE)
#: An identifier-shaped term (CamelCase), the class ``nexus.plans.coverage_repair.key_terms`` returns for these
#: questions: the scope value. The last one wins.
_IDENT_RE = re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+\b")


def route_question(question: str) -> str | None:
    """``"aggregate"``, ``"prev-release"`` or ``None`` (the stock path), from the question wording alone."""
    if _AGG_RE.search(question):
        return "aggregate"
    if _PREV_RE.search(question):
        return "prev-release"
    return None


def scope_value(question: str) -> str | None:
    found = _IDENT_RE.findall(question)
    return found[-1] if found else None


def facet_word(question: str) -> str | None:
    m = _FACET_RE.search(question)
    return m.group(1).lower() if m else None


# --------------------------------------------------------------------------- profile and bindings


@dataclass(frozen=True)
class Profile:
    """Profile-level facts the bindings are assembled from: the entity-key prefix (from the
    ``entity.collections`` pattern), the corpus, and the record-shape words (DEV-derived: the words a
    key-bearing record of this corpus carries around its key)."""

    key_prefix: str
    corpus: str
    record_words: str = "bug impacting customer"
    report_words: str = "reported issues customer"
    entity_word: str = "company"
    #: status word of a question -> the status-record word that retrieves it (DEV-derived). A status with no
    #: entry of its own (unresolved, active) is read as the complement of "resolved".
    status_words: tuple[tuple[str, str], ...] = (("reopened", "reopened"),)
    status_default: str = "resolved"


def key_prefix_from_pattern(pattern: str) -> str | None:
    """The literal prefix of an entity-key pattern: ``CUST-\\d{4}`` -> ``CUST``."""
    m = re.match(r"([A-Za-z][A-Za-z_]*)", pattern)
    return m.group(1) if m else None


def _answer_form(question: str, route: str, facet: str | None) -> str:
    if route == "aggregate":
        status = (f" Count only the bugs that are {facet}, using the status messages; a bug that has no "
                  f"matching status message is not {facet}." if facet else "")
        method = (f"Use the per-customer counts, map each customer key to its company with the Customer "
                  f"records, add the counts per company, and name the company with the highest total "
                  f"(every company on a tie).{status}")
    else:
        method = ("Map each customer key in the records to its company with the Customer records, and name "
                  "every company the records show answering the question.")
    return f"final answer to this question: {question} {method} State the company names plainly."


def plan_bindings(question: str, profile: Profile) -> dict[str, str]:
    """The free-text bindings the plan's ``$`` slots take for one routed question."""
    route = route_question(question)
    if route is None:
        raise ValueError(f"question is not routed to the plan: {question[:60]!r}")
    scope = scope_value(question)
    if scope is None:
        raise ValueError(f"no scope value in the question: {question[:60]!r}")
    facet = facet_word(question) if route == "aggregate" else None
    if route == "aggregate":
        record = f"{scope} {profile.record_words} {profile.key_prefix}"
        word = dict(profile.status_words).get(facet or "", profile.status_default)
        status = f"{scope} bug {word}"
    else:
        record = f"{scope} {profile.report_words} {profile.key_prefix}"
        status = f"{scope} previous release"
    return {
        "record_query": record,
        "status_query": status,
        "entity_query": f"{profile.key_prefix} {profile.entity_word}",
        "corpus": profile.corpus,
        "answer_form": _answer_form(question, route, facet),
    }


# --------------------------------------------------------------------------- the plan

_GROUP_KEY = ("the customer key (an identifier such as the entity-key prefix followed by digits) that the "
              "record names as affected; a record that names none is unassigned")

#: Steps 1-2: the key-bearing records of the scope; 3: partition them by customer key (cheap tier). 4-5: the status
#: or qualifier records. 6: count per key (the runner's deterministic count reducer). 7-8: the entity records
#: (every one of them: the hydrate cap is the entity count of the corpus). 9: the answer, the question arriving as
#: ``$answer_form``. Retrieval steps sit between the operators so that the runner does not fuse them into one
#: opus-tier bundle call (measured $0.32 of a $0.34 run); the generate step names its own model tier.
PLAN: dict[str, Any] = {
    "steps": [
        {"tool": "search", "args": {"query": "$record_query", "corpus": "$corpus", "limit": 60,
                                    "lexical": True, "threshold": 2.0}},
        {"tool": "store_get_many", "args": {"ids": "$step1.ids", "collections": "$step1.collections",
                                            "limit_per_source": 60, "max_chars_per_doc": 700}},
        {"tool": "groupby", "args": {"items": "$step2.contents", "key": _GROUP_KEY}},
        {"tool": "search", "args": {"query": "$status_query", "corpus": "$corpus", "limit": 40,
                                    "lexical": True, "threshold": 2.0}},
        {"tool": "store_get_many", "args": {"ids": "$step4.ids", "collections": "$step4.collections",
                                            "limit_per_source": 40, "max_chars_per_doc": 700}},
        {"tool": "aggregate", "args": {"groups": "$step3.groups", "reducer": "count"}},
        {"tool": "search", "args": {"query": "$entity_query", "corpus": "$corpus", "limit": 120,
                                    "lexical": True, "threshold": 2.0}},
        {"tool": "store_get_many", "args": {"ids": "$step7.ids", "collections": "$step7.collections",
                                            "limit_per_source": 120}},
        {"tool": "generate", "args": {"template": "$answer_form", "model": "sonnet",
                                      "context": ["$step6.aggregates", "$step5.contents", "$step8.contents"]}},
    ],
    "tools_used": ["search", "store_get_many", "groupby", "aggregate", "generate"],
    "outcome_notes": "hand-written generic aggregation plan",
}


def plan_json() -> str:
    return json.dumps(PLAN)


def pin_kwargs() -> dict[str, Any]:
    return {"dimensions": {"verb": PLAN_VERB}, "min_confidence": 0.0}


def pinned_plan_ran(envelope: dict[str, Any], plan_id: int) -> bool:
    return envelope.get("plan_id") == plan_id


# --------------------------------------------------------------------------- readout


def shape_of(question: str) -> str:
    """The prereg's three shapes, by template wording: affected / max-count / prev-release."""
    if "maximum number" in question:
        return "max-count"
    if "previous release" in question:
        return "prev-release"
    return "affected"


def is_decline(text: str | None) -> bool:
    """An answer row that withholds: a coverage non-answer, an insufficient-evidence notice, or nothing."""
    t = (text or "").strip()
    return (not t) or t.startswith("[non-answer:") or t.startswith("[insufficient-evidence:")


def summarize_readout(questions: list[dict[str, Any]], plan_f1: dict[str, float], stock_f1: dict[str, float],
                      rows: dict[tuple[str, int], dict[str, Any]],
                      stock_rows: dict[tuple[str, int], dict[str, Any]] | None = None) -> dict[str, Any]:
    """Cust (mean F1 x100) for the plan run and the stock run over the same questions, overall and per shape,
    with the plan run's decline counts. ``plan_f1``/``stock_f1`` map ``product#index`` to F1."""
    def mean(vals: list[float]) -> float:
        return 100 * sum(vals) / len(vals) if vals else 0.0

    keys = [f"{q['product']}#{q['index']}" for q in questions]
    out: dict[str, Any] = {"overall": {"n": len(keys), "plan": mean([plan_f1.get(k, 0.0) for k in keys]),
                                       "stock": mean([stock_f1.get(k, 0.0) for k in keys])}, "shapes": {}}
    def declines(qlist: list[dict[str, Any]], r: dict[tuple[str, int], dict[str, Any]]) -> int:
        return sum(is_decline((r.get((q["product"], q["index"])) or {}).get("final_text")) for q in qlist)

    out["overall"]["plan_declines"] = declines(questions, rows)
    if stock_rows is not None:
        out["overall"]["stock_declines"] = declines(questions, stock_rows)
    for shape in ("affected", "max-count", "prev-release"):
        sq = [q for q in questions if shape_of(q["question"]) == shape]
        sk = [f"{q['product']}#{q['index']}" for q in sq]
        out["shapes"][shape] = {
            "n": len(sq), "plan": mean([plan_f1.get(k, 0.0) for k in sk]),
            "stock": mean([stock_f1.get(k, 0.0) for k in sk]),
            "plan_declines": declines(sq, rows)}
        if stock_rows is not None:
            out["shapes"][shape]["stock_declines"] = declines(sq, stock_rows)
    return out


# --------------------------------------------------------------------------- paired statistics

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261119


def win_tie_loss(diffs: list[float]) -> tuple[int, int, int]:
    return (sum(d > 0 for d in diffs), sum(d == 0 for d in diffs), sum(d < 0 for d in diffs))


def sign_test(wins: int, losses: int) -> float:
    """Exact two-sided sign test over the untied pairs (p = 1.0 with none)."""
    from math import comb

    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired_bootstrap(diffs: list[float], *, n_resamples: int = BOOTSTRAP_RESAMPLES,
                     seed: int = BOOTSTRAP_SEED) -> tuple[float, float, float]:
    """Mean paired difference with a percentile 95% CI over ``n_resamples`` resamples of the pairs."""
    import random

    if not diffs:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_resamples))
    return (sum(diffs) / n, means[int(0.025 * n_resamples)], means[int(0.975 * n_resamples) - 1])


def required_pairs(*, mean: float, sd: float) -> int | None:
    """Pairs for 80% power at two-sided alpha 0.05 (normal approximation); None when the effect is zero."""
    from math import ceil

    if mean == 0:
        return None
    return ceil(((1.959964 + 0.841621) * sd / abs(mean)) ** 2)


def seeded_subset(items: list[Any], n: int, *, seed: int) -> list[Any]:
    """A reproducible random subset of n items, kept in the input order (all of them when n >= len)."""
    import random

    if n >= len(items):
        return list(items)
    keep = set(random.Random(seed).sample(range(len(items)), n))
    return [x for i, x in enumerate(items) if i in keep]


def _paired_block(diffs: list[float], n_resamples: int, seed: int) -> dict[str, Any]:
    mean, lo, hi = paired_bootstrap(diffs, n_resamples=n_resamples, seed=seed)
    w, t, l = win_tie_loss(diffs)
    verdict = "plan better" if lo > 0 else "stock better" if hi < 0 else "inconclusive"
    return {"n": len(diffs), "mean_diff": mean, "ci95": [lo, hi], "wtl": [w, t, l],
            "sign_p": sign_test(w, l), "verdict": verdict}


def paired_report(questions: list[dict[str, Any]], plan_f1: dict[str, float], stock_f1: dict[str, float], *,
                  n_resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    """Per-question plan-minus-stock F1 on ``questions``: overall and per shape. A CI that crosses 0 is called
    inconclusive, whatever the mean."""
    def d(q: dict[str, Any]) -> float:
        k = f"{q['product']}#{q['index']}"
        return plan_f1.get(k, 0.0) - stock_f1.get(k, 0.0)

    out: dict[str, Any] = {"overall": _paired_block([d(q) for q in questions], n_resamples, seed), "shapes": {}}
    for shape in ("affected", "max-count", "prev-release"):
        sq = [q for q in questions if shape_of(q["question"]) == shape]
        if sq:
            out["shapes"][shape] = _paired_block([d(q) for q in sq], n_resamples, seed)
    return out


# --------------------------------------------------------------------------- driver


def select_products(questions: list[dict[str, Any]], products: frozenset[str] | set[str],
                    *, invert: bool = False) -> list[dict[str, Any]]:
    return [q for q in questions if (q["product"] in products) != invert]


StockFn = Callable[[str, str], dict[str, Any]]
PlanFn = Callable[[str, str, dict[str, str]], dict[str, Any]]


def run_plan_answers(
    questions: list[dict[str, Any]], out_dir: Path, *, stock_fn: StockFn, plan_fn: PlanFn, plan_id: int,
    profile: Profile, cap: float, held_back: float = hp.HELD_BACK_USD, only_routed: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Answer ``questions`` in order: rule-routed ones through the pinned plan, the rest through the stock path
    (or skipped under ``only_routed``). The cap is checked from the ledger before every call; done rows are
    skipped; an error is a recorded zero charged at the unknown-cost charge."""
    out_dir.mkdir(parents=True, exist_ok=True)
    answers = out_dir / "answers.jsonl"
    ledger = hp.Ledger(out_dir / "ledger.jsonl")
    done = hp.done_keys(answers)
    stats: dict[str, Any] = {"attempted": 0, "skipped_done": 0, "errors": 0, "stopped": None, "pin_misses": 0,
                             "routed": 0, "stock": 0}
    for q in questions:
        route = route_question(q["question"])
        if only_routed and route is None:
            continue
        key = (q["product"], q["index"])
        if key in done:
            stats["skipped_done"] += 1
            continue
        if limit is not None and stats["attempted"] >= limit:
            break
        total = ledger.total()
        if hp.would_exceed(total, cap, held_back):
            stats["stopped"] = (f"cap: charged {total:.4f} + {hp.UNKNOWN_COST_USD} > {cap} - {held_back}; "
                                "result is PARTIAL")
            break
        rid = uuid.uuid4().hex
        ledger.append({"reserve_id": rid, "charged": hp.UNKNOWN_COST_USD, "settled": False,
                       "product": q["product"], "index": q["index"]})
        t0 = time.time()
        env: dict[str, Any] = {}
        err: str | None = None
        try:
            if route:
                env = plan_fn(q["question"], hp.POOL_CORPUS, plan_bindings(q["question"], profile))
            else:
                env = stock_fn(q["question"], hp.POOL_CORPUS)
        except Exception as e:  # noqa: BLE001  one attempt; an error is a recorded zero
            err = f"{type(e).__name__}: {e}"
        cost = env.get("cost_usd") if env else None
        charged = hp.charge_for(cost)
        ledger.append({"reserve_id": rid, "charged": charged, "settled": True, "cost_recorded": cost is not None,
                       "product": q["product"], "index": q["index"]})
        pinned = pinned_plan_ran(env, plan_id) if route else None
        row = {"product": q["product"], "index": q["index"], "question": q["question"], "route": route,
               "final_text": (env or {}).get("final_text"), "cost": cost, "duration": (env or {}).get("duration_ms"),
               "wall_s": round(time.time() - t0, 1), "charged": charged, "error": err,
               "plan_id": (env or {}).get("plan_id"), "pinned_plan_ran": pinned,
               "coverage_state": (env or {}).get("coverage_state"),
               "answer_shape": (env or {}).get("answer_shape"), "step_count": (env or {}).get("step_count")}
        with answers.open("a") as f:
            f.write(json.dumps(row) + "\n")
        done.add(key)
        stats["attempted"] += 1
        stats["errors"] += err is not None
        stats["routed" if route else "stock"] += 1
        if route and not pinned:
            stats["pin_misses"] += 1
        hp.console.print(f"[{q['product']}#{q['index']}] route={route} cost={cost} charged={charged:.4f} "
                         f"total={ledger.total():.4f} pinned={pinned} err={err}")
    stats["charged_total"] = ledger.total()
    return stats


# --------------------------------------------------------------------------- subprocess drivers and commands

_INSTALL_DRIVER = """
import json, os, sys
from pathlib import Path
from nexus.mcp import core
from nexus.plans.answer_kind import PlanAnswerKind, saved_plan_json
plan_json, query, verb, tags, prev = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
if prev:
    print("DELETE", core.plan_delete(int(prev)))
# plan_save() stores only the verb COLUMN; the matcher's dimension filter reads the dimensions MAP, which that
# tool never writes. So the same library call plan_save makes is made directly, with the map.
rid = core._t2_index_write(lambda db: db.plans.save_plan(
    query=query, plan_json=saved_plan_json(plan_json, PlanAnswerKind.LOOKUP), outcome="success", tags=tags,
    project=Path.cwd().name, verb=verb, dimensions=json.dumps({"verb": verb})))
print("@@PLANSAVE@@Saved plan: [%s]" % rid)
"""

_PLAN_DRIVER = """
import asyncio, json, os, sys
os.environ.pop("NX_ANSWER_POOL_KEY_SCAN", None)
os.environ["NX_ASPECT_WORKER_AUTOSTART"] = "0"
os.environ["NX_ANSWER_KIND_ELIGIBILITY"] = "0"   # the pinned plan must not be dropped by the answer-kind gate
if len(sys.argv) > 5 and sys.argv[5] == "nocov":
    os.environ["NX_ANSWER_COVERAGE"] = "0"
from nexus.mcp.core import nx_answer
question, scope, bindings, kw = sys.argv[1], sys.argv[2], json.loads(sys.argv[3]), json.loads(sys.argv[4])
async def go():
    r = await nx_answer(question=question, scope=scope, structured=True, bindings=bindings, **kw)
    if isinstance(r, dict):
        out = {k: r.get(k) for k in ("final_text", "cost_usd", "duration_ms", "plan_id", "step_count", "answer_shape")}
        out["coverage_state"] = (r.get("coverage") or {}).get("state")
        out["run_summary"] = r.get("run_summary")
        out["eligibility"] = r.get("eligibility") or r.get("answer_kind")
    else:
        out = {"final_text": r, "cost_usd": None, "duration_ms": None}
    print("@@HERBPLAN@@" + json.dumps(out, default=str))
asyncio.run(go())
"""


def nx_answer_pinned_subprocess(question: str, scope: str, bindings: dict[str, str],
                                *, coverage: bool = True) -> dict[str, Any]:
    """One nx_answer call pinned to the plan, out-of-process in the pinned nexus interpreter."""
    import subprocess

    from benchmark import run_herb as rh
    from benchmark.multihop_resolve import NxError

    try:
        p = subprocess.run([str(rh.CONEXUS_PYTHON), "-c", _PLAN_DRIVER, question, scope, json.dumps(bindings),
                            json.dumps(pin_kwargs()), "cov" if coverage else "nocov"], capture_output=True, timeout=hp.ANSWER_TIMEOUT_S, check=True)
    except subprocess.CalledProcessError as e:
        raise NxError(f"exit {e.returncode}: {(e.stderr or b'').decode(errors='replace')[-500:]}") from e
    except subprocess.TimeoutExpired as e:
        raise NxError(f"timeout after {hp.ANSWER_TIMEOUT_S}s") from e
    for line in p.stdout.decode().splitlines():
        if line.startswith("@@HERBPLAN@@"):
            return json.loads(line[len("@@HERBPLAN@@"):])
    raise NxError("no envelope on stdout")


def cmd_plan_install(args: Any) -> int:
    import subprocess

    from benchmark import run_herb as rh

    hp.sandbox_guard(allow_llm_token=False)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    idf = out / "plan-id.txt"
    prev = idf.read_text().strip() if idf.exists() else ""
    p = subprocess.run([str(rh.CONEXUS_PYTHON), "-c", _INSTALL_DRIVER, plan_json(), PLAN_QUERY, PLAN_VERB,
                        PLAN_TAGS, prev], capture_output=True, text=True, check=False)
    m = re.search(r"@@PLANSAVE@@Saved plan: \[(\d+)\]", p.stdout)
    if not m:
        hp.console.print(f"[red]plan_save failed: {p.stdout[-400:]} {p.stderr[-400:]}[/red]")
        return 2
    idf.write_text(m.group(1))
    (out / f"plan-{m.group(1)}.json").write_text(json.dumps(PLAN, indent=1))
    hp.console.print(p.stdout.strip())
    return 0


def cmd_plan_answer(args: Any) -> int:
    import os
    import sys
    from datetime import datetime, timezone

    from benchmark import run_herb as rh

    hp.sandbox_guard(allow_llm_token=True)
    rev = os.environ.get("HERB_NEXUS_REV", "")
    if not rev:
        hp.console.print("[red]HERB_NEXUS_REV is not set[/red]")
        return 2
    out = Path(args.out_dir)
    idf = Path(args.plan_dir or args.out_dir) / "plan-id.txt"
    if not idf.exists():
        hp.console.print(f"[red]no {idf}; run plan-install first[/red]")
        return 2
    plan_id = int(idf.read_text().strip())
    herb_dir = Path(args.herb_dir).expanduser()
    qs = hp.company_questions(hp.load_products(herb_dir))
    products = DEV_PRODUCTS
    qs = select_products(qs, products, invert=(args.split == "test"))
    expect = {"dev": 27, "test": 103}[args.split]
    if len(qs) != expect:
        hp.console.print(f"[red]split {args.split}: expected {expect} questions, found {len(qs)}[/red]")
        return 2
    subset_f = out / "subset.json"
    if args.subset_n is not None:
        routed = [q for q in qs if route_question(q["question"])]
        keep = seeded_subset([f"{q['product']}#{q['index']}" for q in routed], args.subset_n, seed=args.subset_seed)
        if subset_f.exists() and json.loads(subset_f.read_text())["keys"] != keep:
            hp.console.print(f"[red]{subset_f} differs from the subset these arguments give[/red]")
            return 2
        out.mkdir(parents=True, exist_ok=True)
        subset_f.write_text(json.dumps({"seed": args.subset_seed, "n": args.subset_n, "routed_total": len(routed),
                                        "keys": keep}))
        qs = [q for q in qs if not route_question(q["question"]) or f"{q['product']}#{q['index']}" in set(keep)]
    cfg_dir = Path(os.environ["NEXUS_CONFIG_DIR"])
    pattern = ""
    cfgtxt = (cfg_dir / "config.yml").read_text() if (cfg_dir / "config.yml").exists() else ""
    m = re.search(r"knowledge__herb-\*:\s*(\S+)", cfgtxt)
    pattern = m.group(1) if m else hp.ENTITY_PATTERN
    prefix = key_prefix_from_pattern(pattern)
    if not prefix:
        hp.console.print(f"[red]no key prefix in entity pattern {pattern!r}[/red]")
        return 2
    profile = Profile(key_prefix=prefix, corpus=hp.POOL_CORPUS)
    out.mkdir(parents=True, exist_ok=True)
    mf = out / "manifest.json"
    if not mf.exists():
        mf.write_text(json.dumps({
            "started_at": datetime.now(timezone.utc).isoformat(), "nexus_rev": rev, "split": args.split,
            "plan_id": plan_id, "plan": PLAN, "plan_verb": PLAN_VERB, "profile": profile.__dict__,
            "cap_usd": args.cap_usd, "only_routed": args.only_routed, "plan_coverage": not args.no_coverage, "label": args.label,
            "assetops_sha": rh._sh(["git", "rev-parse", "--short", "HEAD"]),
            "nx_python": str(rh.CONEXUS_PYTHON)}, indent=1, default=str))
    stats = run_plan_answers(
        qs, out, stock_fn=lambda q, s: hp.nx_answer_subprocess(q, s), plan_fn=lambda q, sc, b: nx_answer_pinned_subprocess(q, sc, b, coverage=not args.no_coverage),
        plan_id=plan_id, profile=profile, cap=args.cap_usd, held_back=0.0, only_routed=args.only_routed,
        limit=args.limit)
    hp.console.print(stats)
    (out / f"stats-{datetime.now(timezone.utc):%H%M%S}.json").write_text(json.dumps(stats, indent=1))
    return 0 if stats["stopped"] is None else 3


def cmd_plan_score(args: Any) -> int:
    """Score a plan-answer out dir with HERB's company protocol (gpt-4o extractor) over the split's questions and
    read it out against the stock run's scores for the same questions."""
    herb_dir = Path(args.herb_dir).expanduser()
    out = Path(args.out_dir)
    extract = hp.openai_extractor()
    qs = select_products(hp.company_questions(hp.load_products(herb_dir)), DEV_PRODUCTS, invert=(args.split == "test"))
    rows = hp.load_answers(out / "answers.jsonl")
    if args.only_answered:
        qs = [q for q in qs if (q["product"], q["index"]) in rows]
    cache_f = out / "extractions-openai.jsonl"
    cache: dict[str, list[str]] = {}
    if cache_f.exists():
        cache = {r["key"]: r["names"] for r in map(json.loads, filter(str.strip, cache_f.read_text().splitlines()))}
    res = hp.score_answers(qs, rows, extract, cache, None)
    with cache_f.open("w") as f:
        for k, v in cache.items():
            f.write(json.dumps({"key": k, "names": v}) + "\n")
    plan_f1 = {r["key"]: r["f1"] for r in res["per_question"]}
    stock_f1 = {r["key"]: r["f1"] for r in json.loads((Path(args.stock_dir) / "score-openai.json").read_text())["per_question"]}
    subset_f = out / "subset.json"
    subset = set(json.loads(subset_f.read_text())["keys"]) if subset_f.exists() else None
    # Questions the rule leaves on the stock path (and routed ones outside a declared subset) were not rerun:
    # their rows are the stock run's.
    rerun = [q for q in qs if route_question(q["question"]) and (q["product"], q["index"]) in rows
             and (subset is None or f"{q['product']}#{q['index']}" in subset)]
    rerun_keys = {f"{q['product']}#{q['index']}" for q in rerun}
    for q in qs:
        k = f"{q['product']}#{q['index']}"
        if k not in rerun_keys:
            plan_f1[k] = stock_f1.get(k, 0.0)
    stock_rows = hp.load_answers(Path(args.stock_dir) / "answers.jsonl")
    eff_rows = {**{(q["product"], q["index"]): stock_rows.get((q["product"], q["index"]), {}) for q in qs},
                **{(q["product"], q["index"]): rows[(q["product"], q["index"])] for q in rerun}}
    summary = summarize_readout(qs, plan_f1, stock_f1, eff_rows, stock_rows)
    summary["rerun_declines"] = {"plan": sum(is_decline(rows[(q["product"], q["index"])].get("final_text")) for q in rerun),
                                 "stock": sum(is_decline(stock_rows.get((q["product"], q["index"]), {}).get("final_text")) for q in rerun)}
    summary["merged_note"] = ("non-routed questions carry the stock run's row (identical code, not rerun)"
                              if not args.only_answered else "routed-only DEV run")
    summary["rerun"] = {"n": len(rerun), "routed_in_split": sum(1 for q in qs if route_question(q["question"])),
                        "subset": None if subset is None else len(subset)}
    summary["paired_on_rerun"] = paired_report(rerun, plan_f1, stock_f1)
    summary["status_counts"] = res["status_counts"]
    summary["spend_usd"] = hp.Ledger(out / "ledger.jsonl").total()
    summary["pin_misses"] = sum(1 for r in rows.values() if r.get("route") and not r.get("pinned_plan_ran"))
    summary["per_question"] = [{**r, "plan_f1_used": plan_f1.get(r["key"]), "stock_f1": stock_f1.get(r["key"]),
                                "shape": shape_of(next(q["question"] for q in qs
                                                       if f"{q['product']}#{q['index']}" == r["key"]))}
                               for r in res["per_question"]]
    (out / "score-plan.json").write_text(json.dumps(summary, indent=1))
    hp.console.print({k: v for k, v in summary.items() if k != "per_question"})
    return 0


def add_subcommands(sub: Any) -> None:
    pi = sub.add_parser("plan-install", help="save the part-1 plan into the sandbox plan library (free)")
    pi.set_defaults(fn=cmd_plan_install)
    pa = sub.add_parser("plan-answer", help="rule-routed questions on the pinned plan, the rest stock (PAID)")
    pa.add_argument("--split", choices=["dev", "test"], required=True)
    pa.add_argument("--cap-usd", type=float, required=True)
    pa.add_argument("--only-routed", action="store_true")
    pa.add_argument("--limit", type=int, default=None)
    pa.add_argument("--plan-dir", default=None, help="dir holding plan-id.txt (default: --out-dir)")
    pa.add_argument("--label", default="")
    pa.add_argument("--subset-n", type=int, default=None, help="run a seeded random subset of the routed questions")
    pa.add_argument("--subset-seed", type=int, default=BOOTSTRAP_SEED)
    pa.add_argument("--no-coverage", action="store_true",
                    help="NX_ANSWER_COVERAGE=0 for the pinned-plan calls only (the stock path keeps it on)")
    pa.set_defaults(fn=cmd_plan_answer)
    ps = sub.add_parser("plan-score", help="score a plan-answer dir and read it out against the stock run (extractor PAID)")
    ps.add_argument("--split", choices=["dev", "test"], required=True)
    ps.add_argument("--stock-dir", required=True)
    ps.add_argument("--only-answered", action="store_true", help="restrict to questions with a row (DEV routed-only runs)")
    ps.set_defaults(fn=cmd_plan_score)
