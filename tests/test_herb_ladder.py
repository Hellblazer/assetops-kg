"""RDR-147 benchmark-ladder RED contract.

These assertions are the TDD oracle for RDR-147. They are written FIRST and start
RED; each implementation phase flips exactly one rung to GREEN while the downstream
rungs stay RED with their recorded number — so a phase cannot close as "progress"
while its successors are unverified (the load-bearing-jointness the RDR-147 gate
hardened).

Assertions are EXACT (``==``), never ``>=`` on the count side, so a silent-
corruption regression cannot pass (feedback_exact_assertions_for_fixture_regression).

Marked ``herb`` + ``integration``: excluded from default CI, run at each phase-
review gate inside the sandbox. They require the HERB source at ~/git/HERB and the
``knowledge__herb-searchflow`` corpus indexed.

    pytest -m herb tests/test_herb_ladder.py
"""

from __future__ import annotations

import pytest

herb = pytest.importorskip("benchmark.herb_ladder", reason="assetops-kg benchmark pkg")

pytestmark = [pytest.mark.herb, pytest.mark.integration]


@pytest.fixture(scope="module")
def gold():
    return herb._load_gold()


@pytest.fixture(scope="module")
def scorecard():
    return herb.build_scorecard()


# --- gold fixtures are well-formed (this rung is GREEN as soon as HERB exists) ---


def test_gold_cust_to_company_is_complete(gold):
    cust_to_company, _ = gold
    # HERB SearchFlow ships exactly 120 customer records; lock the count.
    assert len(cust_to_company) == 120
    assert all(cid.startswith("CUST-") for cid in cust_to_company)
    assert all(company for company in cust_to_company.values())


def test_gold_questions_have_affected_customers(gold):
    _, q_to_custs = gold
    assert len(q_to_custs) > 0
    # every company question must resolve to a non-empty gold affected-customer set,
    # else P3 has nothing to grade against.
    assert all(custs for custs in q_to_custs.values())


# --- P1: entity-linker (RED until Step 2 ships + re-index) ---


@pytest.mark.xfail(reason="RDR-147 Step 2 entity-linker not implemented", strict=True)
def test_p1_mentions_link_recall_is_total(gold):
    cust_to_company, _ = gold
    recall, _ = herb.score_p1_link_recall_precision(cust_to_company)
    assert recall == 1.0  # every gold CUST has >=1 inbound 'mentions' edge


# --- P2: resolve operator (can run TODAY against where=title; should be ~green) ---


def test_p2_resolution_accuracy_is_total(gold):
    cust_to_company, _ = gold
    acc = herb.score_p2_resolution_accuracy(cust_to_company)
    # Exact-key resolution is deterministic; demand perfection. If this is < 1.0
    # the where=title primitive or the record-shape assumption regressed.
    assert acc == 1.0


# --- P3: hop-1 precision (RED until operator_filter is in the template, Step 4) ---


@pytest.mark.xfail(reason="RDR-147 Step 4 operator_filter not in plan template", strict=True)
def test_p3_hop1_precision_meets_threshold(gold):
    _, q_to_custs = gold
    prec = herb.score_p3_hop1_precision(q_to_custs)
    assert prec >= 0.80


# --- E2E: the MVV bar (RED until P1+P2+P3 all land) ---


@pytest.mark.xfail(reason="RDR-147 end-to-end recall stays 0 until all phases land", strict=True)
def test_e2e_company_recall_beats_baseline():
    recall = herb.score_e2e_company_recall()
    # The MVV bar: the headline metric the Problem Statement declared broken must move.
    assert recall > herb._E2E_BASELINE
