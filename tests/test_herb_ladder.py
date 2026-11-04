"""RDR-147 benchmark-ladder contract (Step 0, nexus-3lu23).

Marked ``herb`` + ``integration``: excluded from default CI, run at each phase-review gate
inside the sandbox, which must be up with ``knowledge__herb-searchflow`` indexed:

    ./benchmark/herb_sandbox.sh up
    ./benchmark/herb_sandbox.sh run python3 -m benchmark.run_herb --mode index
    ./benchmark/herb_sandbox.sh run python3 -m pytest tests/test_herb_ladder.py

The tests that need nexus fail loudly when the sandbox is down (NxError); they never skip.
Unit tests of the harness itself are in tests/test_herb_harness.py.

Assertions are EXACT (``==``), never ``>=`` on the count side, so a silent-corruption
regression cannot pass. The recorded Step 0 baselines are in
results/herb/step0-baselines.json; the script baseline is asserted equal to them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

herb = pytest.importorskip("benchmark.herb_ladder", reason="assetops-kg benchmark pkg")
from benchmark.run_herb import flatten_artifacts  # noqa: E402

pytestmark = [pytest.mark.herb, pytest.mark.integration]

_BASELINES = Path(__file__).parent.parent / "results" / "herb" / "step0-baselines.json"


@pytest.fixture(scope="module")
def gold():
    return herb._load_gold()


@pytest.fixture(scope="module")
def product():
    return herb._load_product()


@pytest.fixture(scope="module")
def recorded():
    assert _BASELINES.exists(), f"{_BASELINES} is missing: run the ladder and commit its output"
    return json.loads(_BASELINES.read_text())


# --- gold fixtures are well-formed (GREEN as soon as HERB exists; no nexus needed) ---


def test_gold_registry_is_complete(gold):
    c2c = gold["cust_to_company"]
    assert len(c2c) == 120  # the global registry
    assert all(cid.startswith("CUST-") for cid in c2c)
    assert all(company for company in c2c.values())


def test_gold_p2_is_the_29_members(gold, product):
    members = gold["members"]
    assert len(members) == 29
    assert sorted(members) == sorted(product["customers"])
    # the denominator the June scorecard got wrong: 29 resolutions out of 120 read 0.2417
    assert round(29 / 120, 5) == 0.24167


def test_gold_p1_is_37_slack_artifact_customer_pairs(gold, product):
    mentions = gold["mentions"]
    assert len(mentions) == 37
    assert sum(len(v) for v in mentions.values()) == 37
    slack_ids = {m["id"] for m in product["slack"]}
    assert set(mentions) <= slack_ids
    # no other artifact type names a customer id, in any field of the raw product JSON
    for section in ("documents", "meeting_transcripts", "meeting_chats", "urls", "prs"):
        assert re.findall(r"CUST-\d{4}", json.dumps(product[section])) == [], section


def test_every_gold_company_is_reachable_through_a_member(gold, product):
    member_companies = {c.lower() for c in gold["members"].values()}
    for q in herb.company_questions(product):
        assert {g.lower() for g in q["gold"]} <= member_companies, q["question"]


def test_company_questions_split_5_and_2(product):
    kinds = [q["kind"] for q in herb.company_questions(product)]
    assert kinds.count("issue") == 5 and kinds.count("aggregation") == 2


def test_questions_have_affected_customers(gold):
    assert len(gold["q_to_custs"]) == 7
    assert all(custs for custs in gold["q_to_custs"].values())


def test_the_200_char_cut_would_have_lost_most_ids(product):
    """Why the harness reads --json: in the indexed text, 30 of the 37 messages that name a
    customer put the first id at or beyond char 200, where `nx search -c` cuts."""
    offs = [m.start() for r in flatten_artifacts(product) if (m := re.search(r"CUST-\d{4}", r.text))]
    assert len(offs) == 37
    assert sum(1 for o in offs if o >= 200) == 30


# --- the corpus is the one the baselines were measured on ---


def test_corpus_is_indexed_through_the_catalog_path():
    """A document per artifact, not a bare T3 write: titles resolve to catalog documents."""
    hits = herb.search_hits("customer company contact", 1, where="title=CUST-0089")
    assert len(hits) == 1 and hits[0]["title"] == "CUST-0089"
    assert hits[0].get("catalog_doc_id"), "no catalog document: the corpus was written around the note path"


# --- P1: entity-linker (RED until Step 2 ships + re-index) ---


@pytest.mark.xfail(reason="RDR-147 Step 2 entity-linker not implemented", strict=True)
def test_p1_mentions_link_recall_is_total(gold):
    recall, _ = herb.score_p1_link_recall_precision(gold["mentions"])
    assert recall == 1.0  # every gold artifact--mentions-->customer pair has an edge


# --- P2: the `resolve` step over the 29 members (RDR-147 Step 3, nexus-v9v5r) ---
#
# The first group runs on a fake resolver (no sandbox, no engine); it pins how the harness CALLS
# nexus's resolve_keys and reads its answer. The last test is the live one.


class _FakeResolver:
    """Stands in for the resolve_keys driver: records every call, answers from the gold members."""

    def __init__(self, members, *, missing=(), ambiguous=None, company_for=None):
        self.members, self.calls = members, []
        self.missing, self.ambiguous = list(missing), dict(ambiguous or {})
        self.company_for = company_for or {}

    def __call__(self, tokens):
        self.calls.append(list(tokens))
        by_token = {t: {"tumbler": f"1.1.{i}", "ids": [f"h{i}"], "collections": ["knowledge__herb-searchflow"]}
                    for i, t in enumerate(tokens, 1)}
        texts = {t: f"Customer {t}: contact N, R at company {self.company_for.get(t, self.members[t])}."
                 for t in tokens}
        return {
            "service_url": "http://127.0.0.1:1", "lookups": len(set(tokens)), "resolve_s": 0.0,
            "texts": texts,
            "result": {"by_token": by_token, "missing": self.missing, "ambiguous": self.ambiguous,
                       "tumblers": [], "ids": [], "collections": []},
        }


@pytest.fixture(scope="module")
def members():
    return json.loads((Path(herb.__file__).parent / "fixtures" / "herb" / "p2_member_customers.json").read_text())


def test_p2_sends_all_29_tokens_in_one_call(members):
    fake = _FakeResolver(members)
    detail = herb.score_p2_resolution(members, resolver=fake)
    assert len(members) == 29
    assert fake.calls == [list(members)]  # one call, every token, in fixture order
    assert detail["n_tokens"] == 29 and detail["resolve_calls"] == 1
    assert detail["accuracy"] == 1.0 and detail["passed"] is True
    assert herb.p2_rung(detail).status == "GREEN"


def test_p2_reports_missing_and_ambiguous_from_the_result(members):
    gone, twice = "CUST-0018", "CUST-0027"
    fake = _FakeResolver(members, missing=[gone], ambiguous={twice: ["1.1.5", "1.2.9"]})
    detail = herb.score_p2_resolution(members, resolver=fake)
    assert detail["missing"] == [gone]
    assert detail["ambiguous"] == {twice: ["1.1.5", "1.2.9"]}


@pytest.mark.parametrize("kw", [{"missing": ["CUST-0018"]}, {"ambiguous": {"CUST-0018": ["1.1.1", "1.2.1"]}}])
def test_p2_fails_when_missing_or_ambiguous_is_non_empty(members, kw):
    # Every token still reads as right in by_token/texts, so only the two lists can fail the rung.
    detail = herb.score_p2_resolution(members, resolver=_FakeResolver(members, **kw))
    assert detail["accuracy"] == 1.0
    assert detail["passed"] is False
    assert herb.p2_rung(detail).status == "RED"


def test_p2_a_record_naming_another_company_is_wrong(members):
    cid = "CUST-0018"
    other = "NotTheCompany"
    assert members[cid] != other
    detail = herb.score_p2_resolution(members, resolver=_FakeResolver(members, company_for={cid: other}))
    assert detail["wrong"] == [cid]
    assert detail["accuracy"] == 28 / 29
    assert detail["passed"] is False


def test_p2_accuracy_wrapper_goes_through_the_resolve_call(members, monkeypatch):
    fake = _FakeResolver(members)
    monkeypatch.setattr(herb, "resolve_p2_tokens", fake)
    monkeypatch.setattr(herb, "resolve_company", lambda cid: pytest.fail("P2 must not use the per-token lookup"))
    assert herb.score_p2_resolution_accuracy(members) == 1.0
    assert len(fake.calls) == 1


def test_p2_resolve_driver_refuses_to_run_outside_the_sandbox(members, monkeypatch):
    monkeypatch.delenv("HERB_SANDBOX", raising=False)
    with pytest.raises(herb.NxError, match="herb_sandbox.sh"):
        herb.resolve_p2_tokens(list(members))


def test_p2_resolution_accuracy_is_total(gold):
    """Live: the sandbox catalog, through resolve_keys."""
    detail = herb.score_p2_resolution(gold["members"])
    assert detail["accuracy"] == 1.0
    assert detail["missing"] == [] and detail["ambiguous"] == {}
    assert detail["passed"] is True
    assert detail["n_tokens"] == 29 and detail["resolve_calls"] == 1


# --- P3: hop-1 precision (RED until a precision step exists) ---


@pytest.mark.xfail(reason="RDR-147 P3 bar 0.80 not met by plain hop-1 retrieval", strict=True)
def test_p3_hop1_precision_meets_threshold(gold):
    assert herb.score_p3_hop1_precision(gold["q_to_custs"]) >= 0.80


# --- E2E: the script baseline is reproducible and equals the recorded one ---


def test_script_baseline_equals_the_recorded_one(product, recorded):
    rows = herb.run_script_rows(herb.company_questions(product), k=recorded["k"])
    want = recorded["baselines"]["script_fixed"]["rows"]
    assert len(rows) == len(want) == 7
    for got, rec in zip(rows, want, strict=True):
        assert got["question"] == rec["question"]
        assert got["pred"] == rec["pred"]
        assert (got["recall"], got["precision"]) == (rec["recall"], rec["precision"])
    assert herb.summarize(rows) == recorded["baselines"]["script_fixed"]["summary"]


def test_script_baseline_is_not_vacuous(recorded):
    """A dead sandbox must not read as recall 0: hop 1 returned hits for every question."""
    rows = recorded["baselines"]["script_fixed"]["rows"]
    assert [r["n_hits"] for r in rows] == [20] * 7


def test_recorded_baselines_name_their_versions(recorded):
    m = recorded["manifest"]
    assert m["sandbox"] is True
    assert m["nx_version"].startswith("nx, version ")
    assert re.fullmatch(r"\d+\.\d+\.\d+", m["engine_release_version"])
    assert m["embedding_model"] == "bge-base-en-v15-768"


def test_p3_precision_is_high_on_the_issue_lookups_and_zero_on_the_aggregations(recorded):
    """The overall P3 mean (below the 0.80 bar) comes from the two aggregation questions, which
    surface at most one customer id; on the 5 issue lookups hop-1 precision is above the bar."""
    by = recorded["p3"]["by_kind"]
    assert by["agg2"]["mean_precision"] == 0.0
    assert by["issue5"]["mean_precision"] >= 0.80
    assert herb.p3_by_kind(recorded["p3"]["rows"]) == by


def test_oracle_ceiling_is_recorded_and_total(recorded):
    s = recorded["baselines"]["oracle_cited_artifacts"]["summary"]
    assert s["all7"]["recall"] == 1.0


# --- E2E product rung: nx_answer must reach the script's F1 (the MVV bar) ---


def test_nx_answer_baseline_run_was_complete_and_within_the_cap(recorded):
    run = recorded["baselines"]["nx_answer"]["run"]
    assert run["complete"] is True
    assert run["runs_with_unknown_cost"] == 0
    assert run["total_cost_usd"] <= 2.0


@pytest.mark.xfail(reason="RDR-147 MVV: nx_answer declines these questions today (non-answer: uncovered)", strict=True)
def test_nx_answer_company_f1_reaches_the_script_baseline(recorded):
    nxa = recorded["baselines"]["nx_answer"]["summary"]["issue5"]["f1"]
    script = recorded["baselines"]["script_fixed"]["summary"]["issue5"]["f1"]
    assert nxa >= script
