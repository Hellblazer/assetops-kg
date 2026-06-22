"""RDR-147 benchmark ladder — per-phase observable metrics over HERB SearchFlow.

The single end-to-end metric (company recall) only moves when all three RDR-147
phases land, so it gives no intermediate signal. This module decomposes it into a
ladder of isolated sub-metrics — each phase turns its own metric off zero even
while company recall is still 0 — and emits a ``herb-scorecard.json`` that a copy
checked into the nexus tree regresses against.

    Phase             metric                         graded against
    ----------------  -----------------------------  ----------------------------
    P1 (linker)       mentions-link precision+recall gold CUST->company edges
    P2 (resolve)      resolution accuracy            gold ids fed in (hop-1 bypassed)
    P3 (filter)       hop-1 precision                gold question->affected-customers
    E2E (MVV)         company recall                 HERB ground_truth

Every metric is asserted EXACT (``== target``, never ``>=``) by the paired pytest
so a silent-corruption regression cannot pass. Until the corresponding nexus
feature exists the metric computes to its RED baseline; the harness is the oracle,
not the implementation.

Run inside the isolated sandbox (same as multihop_resolve):
    ./benchmark/herb_sandbox.sh .venv/bin/python -m benchmark.herb_ladder
    ./benchmark/herb_sandbox.sh .venv/bin/python -m benchmark.herb_ladder --build-gold
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from rich.console import Console

# Reuse the existing, proven harness primitives rather than re-deriving them.
from benchmark.multihop_resolve import (  # type: ignore
    CORPUS,
    HERB_DIR,
    _RE_CUST,
    _search,
    answer_company,
    resolve_company,
)

console = Console()

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "herb"
_GOLD_CUST_COMPANY = _FIXTURE_DIR / "cust_to_company.json"
_GOLD_Q_CUSTOMERS = _FIXTURE_DIR / "question_to_affected_customers.json"
_SCORECARD = Path(__file__).parent.parent / "results" / "herb" / "herb-scorecard.json"

# The E2E baseline the Problem Statement declared broken (single-hop nx_answer).
_E2E_BASELINE = 0.000


# --------------------------------------------------------------------------- #
# Gold-truth fixtures — built once, deterministically, from the HERB source.
# --------------------------------------------------------------------------- #


def build_gold_fixtures() -> tuple[dict[str, str], dict[str, list[str]]]:
    """Derive the two gold fixtures from the HERB source.

    cust_to_company: from the GLOBAL customer registry
    ``data/metadata/customers_data.json`` — a 120-row list of
    ``{"id": "CUST-xxxx", "company": ..., "name": ..., "role": ...}``. This is the
    authoritative CUST->company truth: HERB products only carry a membership list of
    bare CUST-id strings (``SearchFlow.json["customers"]``), so the registry, not
    product document text, is the source. (The earlier draft scanned product
    ``documents`` for an "... at company C." shape that HERB does not use — it
    yielded 0 pairs against the real schema.)

    question_to_affected_customers: each SearchFlow company-type question's
    ``ground_truth`` is a list of company NAMES; invert it against cust_to_company to
    get every CUST-id whose company is named. Inversion is GLOBAL (against all 120
    registry customers), not restricted to SearchFlow's 29-entry membership list:
    the registry is the customer universe and a company question asks which customers
    (by company) are affected. Product-scoped inversion was considered and rejected —
    it produces empty gold sets for some questions, which would make the P3 oracle
    degenerate (and violate the non-empty-affected-set well-formedness assertion).
    Verified 2026-06-21: all 7 SearchFlow company questions map fully (no missing
    companies; affected sets 13..81).
    """
    registry = json.loads(
        (HERB_DIR / "data/metadata/customers_data.json").read_text()
    )
    cust_to_company: dict[str, str] = {
        c["id"]: c["company"]
        for c in registry
        if c.get("id") and c.get("company")
    }

    product = json.loads((HERB_DIR / "data/products/SearchFlow.json").read_text())

    # question -> affected CUST-ids: invert cust_to_company against gold companies.
    company_to_custs: dict[str, list[str]] = {}
    for cid, comp in cust_to_company.items():
        company_to_custs.setdefault(comp.lower(), []).append(cid)

    q_to_custs: dict[str, list[str]] = {}
    for q in product["answerable_questions"]:
        if q.get("type") != "company":
            continue
        gold_companies = {g.lower() for g in q["ground_truth"]}
        affected = sorted(
            {cid for comp in gold_companies for cid in company_to_custs.get(comp, [])}
        )
        q_to_custs[q["question"]] = affected

    _FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    _GOLD_CUST_COMPANY.write_text(json.dumps(cust_to_company, indent=2, sort_keys=True))
    _GOLD_Q_CUSTOMERS.write_text(json.dumps(q_to_custs, indent=2, sort_keys=True))
    console.print(
        f"[green]gold fixtures written[/green]: "
        f"{len(cust_to_company)} cust->company, {len(q_to_custs)} questions"
    )
    return cust_to_company, q_to_custs


def _load_gold() -> tuple[dict[str, str], dict[str, list[str]]]:
    if not _GOLD_CUST_COMPANY.exists() or not _GOLD_Q_CUSTOMERS.exists():
        return build_gold_fixtures()
    return (
        json.loads(_GOLD_CUST_COMPANY.read_text()),
        json.loads(_GOLD_Q_CUSTOMERS.read_text()),
    )


# --------------------------------------------------------------------------- #
# Per-phase metrics. Each returns an exact float in [0, 1] (or None if the
# feature it grades does not exist yet -> RED).
# --------------------------------------------------------------------------- #


def score_p1_link_recall_precision(
    cust_to_company: dict[str, str],
) -> tuple[float | None, float | None]:
    """P1: does the entity-linker create correct artifact--mentions-->entity edges?

    Graded directly on the catalog edge set vs the gold CUST->company truth —
    independent of retrieval. RED (None) until the entity-linker hook (RDR-147
    Step 2) exists and the corpus is re-indexed with it enabled.

    Implementation when the feature lands:
        for each CUST-id in gold: query catalog links of type 'mentions' into the
        entity record; recall = fraction of gold ids with >=1 inbound mention;
        precision = fraction of created mentions edges that are correct.
    """
    # FEATURE-ABSENT STUB: no 'mentions' links exist yet -> RED baseline.
    # Replace with: nx catalog link_query --link-type mentions ... once Step 2 ships.
    return None, None


def score_p2_resolution_accuracy(cust_to_company: dict[str, str]) -> float:
    """P2: given a CORRECT CUST-id, does resolution return the right company?

    Hop-1 is bypassed (gold ids fed directly) so this isolates the resolve
    operator / where=title exact-filter. Reuses the proven resolve_company().
    This metric can run TODAY against the where=title primitive and should already
    be high — it is the one rung that is green before P2's operator is formalised.
    """
    if not cust_to_company:
        return 0.0
    hits = 0
    for cid, gold_company in cust_to_company.items():
        pred = resolve_company(cid)
        if pred and pred.strip().lower() == gold_company.strip().lower():
            hits += 1
    return hits / len(cust_to_company)


def score_p3_hop1_precision(q_to_custs: dict[str, list[str]], k: int = 20) -> float:
    """P3: of the customers surfaced for a question, what fraction are affected?

    This is the recursing-bottleneck number. Graded on hop-1's surfaced CUST set
    vs the gold affected-customer set. Without the operator_filter precision step
    (RDR-147 Step 4) this is low — that is exactly the signal P3 must move.

    precision = |surfaced & gold| / |surfaced|, averaged over questions.
    """
    if not q_to_custs:
        return 0.0
    precisions: list[float] = []
    for question, gold in q_to_custs.items():
        gold_set = set(gold)
        hop1 = _search(question, k=k)
        surfaced = set(_RE_CUST.findall(hop1))
        if not surfaced:
            precisions.append(0.0)
            continue
        precisions.append(len(surfaced & gold_set) / len(surfaced))
    return sum(precisions) / len(precisions)


def score_e2e_company_recall(k: int = 20) -> float:
    """E2E (MVV): company recall on real HERB questions. The headline metric.

    Reuses answer_company() — the full discovered plan. This is the number that
    must move off 0.000 for the RDR to close. Stays ~0 until P1+P2+P3 all land.
    """
    product = json.loads((HERB_DIR / "data/products/SearchFlow.json").read_text())
    qs = [q for q in product["answerable_questions"] if q.get("type") == "company"]
    if not qs:
        return 0.0
    recalls: list[float] = []
    for q in qs:
        gold = {g.lower() for g in q["ground_truth"]}
        pred = {c.lower() for c in answer_company(q["question"], k=k)}
        recalls.append(len(gold & pred) / len(gold) if gold else 0.0)
    return sum(recalls) / len(recalls)


# --------------------------------------------------------------------------- #
# Scorecard assembly.
# --------------------------------------------------------------------------- #


@dataclass
class Rung:
    phase: str
    metric: str
    value: float | None
    target: str  # human-readable target, e.g. "==1.0" or ">0"
    status: str  # "RED" | "GREEN"


def _status(value: float | None, predicate: Callable[[float], bool]) -> str:
    return "GREEN" if value is not None and predicate(value) else "RED"


def build_scorecard(k: int = 20) -> dict:
    cust_to_company, q_to_custs = _load_gold()

    p1_recall, p1_prec = score_p1_link_recall_precision(cust_to_company)
    p2 = score_p2_resolution_accuracy(cust_to_company)
    p3 = score_p3_hop1_precision(q_to_custs, k=k)
    e2e = score_e2e_company_recall(k=k)

    # Targets are the locked RDR-147 ladder bars. P3's threshold is provisional
    # (tune during Phase 4); use 0.80 as the initial precision gate.
    rungs = [
        Rung("P1", "mentions_link_recall", p1_recall, "==1.0", _status(p1_recall, lambda v: v == 1.0)),
        Rung("P1", "mentions_link_precision", p1_prec, ">=0.95", _status(p1_prec, lambda v: v >= 0.95)),
        Rung("P2", "resolution_accuracy", p2, "==1.0", _status(p2, lambda v: v == 1.0)),
        Rung("P3", "hop1_precision", p3, ">=0.80", _status(p3, lambda v: v >= 0.80)),
        Rung("E2E", "company_recall", e2e, ">0", _status(e2e, lambda v: v > 0.0)),
    ]
    return {
        "rdr": "RDR-147",
        "baseline_e2e_company_recall": _E2E_BASELINE,
        "k": k,
        "rungs": [asdict(r) for r in rungs],
    }


def emit_scorecard(k: int = 20) -> dict:
    card = build_scorecard(k=k)
    _SCORECARD.parent.mkdir(parents=True, exist_ok=True)
    _SCORECARD.write_text(json.dumps(card, indent=2))
    for r in card["rungs"]:
        colour = "green" if r["status"] == "GREEN" else "red"
        val = "—" if r["value"] is None else f"{r['value']:.3f}"
        console.print(
            f"[{colour}]{r['status']:5}[/{colour}] {r['phase']:4} "
            f"{r['metric']:24} {val:>6}  (target {r['target']})"
        )
    console.print(f"\nscorecard written: {_SCORECARD}")
    return card


def main() -> None:
    ap = argparse.ArgumentParser(description="RDR-147 HERB benchmark ladder")
    ap.add_argument("--build-gold", action="store_true", help="rebuild gold fixtures and exit")
    ap.add_argument("-k", type=int, default=20, help="hop-1 retrieval depth")
    args = ap.parse_args()
    if args.build_gold:
        build_gold_fixtures()
        return
    emit_scorecard(k=args.k)


if __name__ == "__main__":
    main()
