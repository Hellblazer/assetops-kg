"""Free checks for part 1 of the HERB plan experiment (T2 nexus_rdr/herb-plan-experiment-prereg-2026-11-19 and
its addendum 1): the deterministic routing rule, the scope-value and query assembly, the plan template's shape
(operators the runner dispatches today, no HERB text) and the plan-answer driver's bookkeeping with stubs.
Nothing here calls nx_answer, the claude CLI, an LLM or OpenAI.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from benchmark import herb_plan as hpl
from benchmark import run_herb_pool as hp

HERB_DIR = Path(os.environ.get("HERB_DIR") or "~/git/HERB").expanduser()
herb = pytest.mark.herb

# Question templates as they read in HERB (the wording only; the products differ).
AGG = "Find the name of company that has the maximum number of resolved bugs in ActionGenie"
AGG_Q = "Find the name of company that has the maximum number of unresolved bugs in SearchFlow?"
AGG_DOT = "Find the name of company that has the maximum number of reopened bugs in CoachForce."
PREV = "Find names of companies that reported issues with the previous release of AutoTuneForce?"
PREV_STR = "Find names of companies that highlighted strengths of the previous release of AutoTuneForce?"
LOOKUP = ("Find the names of companies were affected by issues in the NLP Model & Language Processing of "
          "ActionGenie that led to inaccurate action suggestions and workflow disruptions?")


# --------------------------------------------------------------------------- routing rule


@pytest.mark.parametrize("q", [AGG, AGG_Q, AGG_DOT])
def test_rule_marks_aggregate_questions(q):
    assert hpl.route_question(q) == "aggregate"


@pytest.mark.parametrize("q", [PREV, PREV_STR])
def test_rule_marks_previous_release_questions(q):
    assert hpl.route_question(q) == "prev-release"


def test_rule_leaves_lookups_on_the_stock_path():
    assert hpl.route_question(LOOKUP) is None
    assert hpl.route_question("") is None


def test_rule_aggregate_wins_over_previous_release():
    assert hpl.route_question("the company with the most bugs in the previous release of X") == "aggregate"


def test_rule_is_word_bounded_and_case_insensitive():
    assert hpl.route_question("Name the company with the MOST reopened bugs in ConnectForce") == "aggregate"
    assert hpl.route_question("utmost care of ActionGenie") is None  # "most" inside a word is not the rule
    assert hpl.route_question("Companies that saw the Previous Release of FlowForce fail") == "prev-release"


def test_rule_reads_the_question_only():
    # the rule's only input is the string: it takes no gold, no product, no row
    import inspect
    assert list(inspect.signature(hpl.route_question).parameters) == ["question"]


# --------------------------------------------------------------------------- scope value and facet


@pytest.mark.parametrize("q,scope", [
    (AGG, "ActionGenie"), (AGG_Q, "SearchFlow"), (AGG_DOT, "CoachForce"), (PREV, "AutoTuneForce"),
    (LOOKUP, "ActionGenie"),
    ("Find the name of company that has the maximum number of active bugs in WorkFlowGenie.", "WorkFlowGenie"),
])
def test_scope_value_is_the_identifier_shaped_term(q, scope):
    assert hpl.scope_value(q) == scope


def test_scope_value_none_when_the_question_names_no_identifier():
    assert hpl.scope_value("which company has the most bugs") is None


def test_facet_word_of_an_aggregate_question():
    assert hpl.facet_word(AGG) == "resolved"
    assert hpl.facet_word(AGG_Q) == "unresolved"
    assert hpl.facet_word(AGG_DOT) == "reopened"
    assert hpl.facet_word(PREV) is None


# --------------------------------------------------------------------------- bindings


PROFILE = hpl.Profile(key_prefix="CUST", corpus="knowledge__herb-pool")


def test_key_prefix_from_the_entity_pattern():
    assert hpl.key_prefix_from_pattern(r"CUST-\d{4}") == "CUST"
    assert hpl.key_prefix_from_pattern(r"RDR-\d+") == "RDR"
    assert hpl.key_prefix_from_pattern(r"\d+") is None


def test_bindings_for_an_aggregate_question():
    b = hpl.plan_bindings(AGG, PROFILE)
    assert b["record_query"] == "ActionGenie bug impacting customer CUST"
    assert b["entity_query"] == "CUST company"
    assert b["corpus"] == "knowledge__herb-pool"
    assert AGG in b["answer_form"]  # the question travels as a binding, never inside the plan


def test_bindings_for_a_previous_release_question():
    b = hpl.plan_bindings(PREV, PROFILE)
    assert b["record_query"].startswith("AutoTuneForce ")
    assert b["record_query"].endswith("CUST")
    assert PREV in b["answer_form"]


def test_bindings_refuse_an_unrouted_question():
    with pytest.raises(ValueError):
        hpl.plan_bindings(LOOKUP, PROFILE)


def test_bindings_refuse_a_question_with_no_scope_value():
    with pytest.raises(ValueError):
        hpl.plan_bindings("which company has the most bugs", PROFILE)


def test_bindings_are_strings_the_runner_can_substitute():
    for q in (AGG, PREV):
        b = hpl.plan_bindings(q, PROFILE)
        assert all(isinstance(v, str) and v for v in b.values())


# --------------------------------------------------------------------------- the plan template


def _steps(plan):
    return plan["steps"]


def test_plan_uses_only_operators_the_runner_dispatches_today():
    allowed = {"search", "store_get_many", "groupby", "aggregate", "generate", "extract", "filter", "rank",
               "summarize", "compare", "traverse", "resolve"}
    tools = {s["tool"] for s in _steps(hpl.PLAN)}
    assert tools <= allowed
    assert {"search", "store_get_many", "groupby", "aggregate", "generate"} <= tools


def test_plan_hydrates_at_least_forty_records():
    gets = [s for s in _steps(hpl.PLAN) if s["tool"] == "store_get_many"]
    assert gets and gets[0]["args"]["limit_per_source"] >= 40


def test_plan_retrieval_is_lexical_and_scope_free_of_literals():
    searches = [s for s in _steps(hpl.PLAN) if s["tool"] == "search"]
    assert searches
    for s in searches:
        assert s["args"]["lexical"] is True
        assert s["args"]["query"].startswith("$")  # assembled from bindings, never a literal


def test_plan_step_refs_point_backwards_only():
    for i, s in enumerate(_steps(hpl.PLAN), start=1):
        for m in re.finditer(r"\$step(\d+)\.", json.dumps(s)):
            assert int(m.group(1)) < i


def test_plan_has_no_herb_text():
    blob = json.dumps(hpl.PLAN)
    for word in ("HERB", "ActionGenie", "SearchFlow", "AutoTuneForce", "CoachForce", "AnomalyForce",
                 "CUST-", "BlueWave", "TechCorp", "Force", "Genie"):
        assert word not in blob, word


@herb
def test_plan_carries_no_product_name_or_question_text():
    products = hp.load_products(HERB_DIR)
    blob = json.dumps(hpl.PLAN).lower() + hpl.PLAN_QUERY.lower()
    for name, _ in products:
        assert name.lower() not in blob
    for q in hp.company_questions(products):
        assert q["question"].lower() not in blob


def test_plan_validates_with_the_save_rules_it_will_meet():
    # whole-token $refs only: an embedded $stepN.field is refused by plan_save
    for s in _steps(hpl.PLAN):
        for v in s["args"].values():
            for item in v if isinstance(v, list) else [v]:
                if isinstance(item, str) and "$" in item:
                    assert re.fullmatch(r"\$[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?", item), item


def test_plan_yaml_roundtrip_is_json_equal():
    assert json.loads(hpl.plan_json()) == hpl.PLAN


# --------------------------------------------------------------------------- driver bookkeeping


def _qs():
    return [
        {"product": "P", "index": 0, "question": AGG, "ground_truth": ["x"], "citations": []},
        {"product": "P", "index": 1, "question": LOOKUP, "ground_truth": ["x"], "citations": []},
        {"product": "Q", "index": 0, "question": PREV, "ground_truth": ["x"], "citations": []},
    ]


def test_split_questions_by_product_set():
    qs = _qs()
    dev = hpl.select_products(qs, {"P"})
    assert [(q["product"], q["index"]) for q in dev] == [("P", 0), ("P", 1)]
    test = hpl.select_products(qs, {"P"}, invert=True)
    assert [(q["product"], q["index"]) for q in test] == [("Q", 0)]


def test_dev_and_test_sets_partition_the_52_products():
    assert len(hpl.DEV_PRODUCTS) == 5
    assert hpl.DEV_PRODUCTS == frozenset({"SearchFlow", "ActionGenie", "AnomalyForce", "AutoTuneForce", "CoachForce"})


@herb
def test_dev_has_27_questions_and_test_103():
    qs = hp.company_questions(hp.load_products(HERB_DIR))
    dev = hpl.select_products(qs, hpl.DEV_PRODUCTS)
    test = hpl.select_products(qs, hpl.DEV_PRODUCTS, invert=True)
    assert (len(dev), len(test)) == (27, 103)


@herb
def test_rule_shapes_on_dev_match_the_template_wording():
    qs = hp.company_questions(hp.load_products(HERB_DIR))
    dev = hpl.select_products(qs, hpl.DEV_PRODUCTS)
    routed = [(q["product"], q["index"], hpl.route_question(q["question"])) for q in dev
              if hpl.route_question(q["question"])]
    assert sum(1 for *_, r in routed if r == "aggregate") == 10
    assert sum(1 for *_, r in routed if r == "prev-release") == 2
    # no lookup on DEV is caught by the rule
    for q in dev:
        if "affected by" in q["question"]:
            assert hpl.route_question(q["question"]) is None


def test_plan_pin_dimensions_and_floor():
    kw = hpl.pin_kwargs()
    assert kw["dimensions"] == {"verb": hpl.PLAN_VERB}
    assert kw["min_confidence"] == 0.0


def test_plan_driver_asserts_the_pinned_plan_ran():
    assert hpl.pinned_plan_ran({"plan_id": 7}, 7) is True
    assert hpl.pinned_plan_ran({"plan_id": 3}, 7) is False
    assert hpl.pinned_plan_ran({}, 7) is False


def test_run_plan_answers_routes_and_records(tmp_path):
    calls = []

    def stock(q, scope):
        calls.append(("stock", q))
        return {"final_text": "stock", "cost_usd": 0.05, "plan_id": 1}

    def planned(q, scope, bindings):
        calls.append(("plan", q))
        return {"final_text": "planned", "cost_usd": 0.10, "plan_id": 99}

    stats = hpl.run_plan_answers(_qs(), tmp_path, stock_fn=stock, plan_fn=planned, plan_id=99, profile=PROFILE,
                                 cap=5.0, held_back=0.0)
    assert [c[0] for c in calls] == ["plan", "stock", "plan"]
    rows = [json.loads(line) for line in (tmp_path / "answers.jsonl").read_text().splitlines()]
    assert [r["route"] for r in rows] == ["aggregate", None, "prev-release"]
    assert [r["final_text"] for r in rows] == ["planned", "stock", "planned"]
    assert all(r["pinned_plan_ran"] is True for r in rows if r["route"])
    assert stats["charged_total"] == pytest.approx(0.25)


def test_run_plan_answers_flags_a_pinned_plan_that_did_not_run(tmp_path):
    def planned(q, scope, bindings):
        return {"final_text": "inline", "cost_usd": 0.2, "plan_id": 5}  # the matcher missed: a planner ran

    stats = hpl.run_plan_answers(_qs()[:1], tmp_path, stock_fn=lambda q, s: {}, plan_fn=planned, plan_id=99,
                                 profile=PROFILE, cap=5.0, held_back=0.0)
    row = json.loads((tmp_path / "answers.jsonl").read_text().splitlines()[0])
    assert row["pinned_plan_ran"] is False
    assert stats["pin_misses"] == 1


def test_run_plan_answers_only_routed_skips_stock_questions(tmp_path):
    def stock(q, scope):
        raise AssertionError("stock path must not run")

    stats = hpl.run_plan_answers(_qs(), tmp_path, stock_fn=stock,
                                 plan_fn=lambda q, s, b: {"final_text": "p", "cost_usd": 0.1, "plan_id": 99},
                                 plan_id=99, profile=PROFILE, cap=5.0, held_back=0.0, only_routed=True)
    rows = [json.loads(line) for line in (tmp_path / "answers.jsonl").read_text().splitlines()]
    assert [r["route"] for r in rows] == ["aggregate", "prev-release"]
    assert stats["attempted"] == 2


def test_run_plan_answers_stops_at_the_cap_before_a_call(tmp_path):
    n = []

    def planned(q, scope, bindings):
        n.append(q)
        return {"final_text": "p", "cost_usd": 0.4, "plan_id": 99}

    stats = hpl.run_plan_answers(_qs(), tmp_path, stock_fn=lambda q, s: {"final_text": "s", "cost_usd": 0.05},
                                 plan_fn=planned, plan_id=99, profile=PROFILE, cap=0.5, held_back=0.0)
    assert stats["stopped"] and "cap" in stats["stopped"]
    assert len(n) == 1  # the second routed question never ran


# --------------------------------------------------------------------------- readout


def test_shape_of_by_template_wording():
    assert hpl.shape_of(AGG) == "max-count"
    assert hpl.shape_of(PREV) == "prev-release"
    assert hpl.shape_of(LOOKUP) == "affected"


def test_is_decline():
    assert hpl.is_decline("[non-answer: uncovered] The plan retrieved evidence")
    assert hpl.is_decline("[insufficient-evidence: 1 unresolved] The evidence did not settle")
    assert not hpl.is_decline("[exact-key join: 1 record(s)] The company is X")
    assert not hpl.is_decline("TechCorp")
    assert hpl.is_decline("")
    assert hpl.is_decline(None)


def test_summarize_readout_per_shape_and_declines():
    qs = [{"product": "P", "index": 0, "question": AGG}, {"product": "P", "index": 1, "question": LOOKUP},
          {"product": "P", "index": 2, "question": PREV}]
    plan = {"P#0": 1.0, "P#1": 0.5, "P#2": 0.0}
    stock = {"P#0": 0.0, "P#1": 0.5, "P#2": 0.5}
    rows = {("P", 0): {"final_text": "X"}, ("P", 1): {"final_text": "Y"},
            ("P", 2): {"final_text": "[non-answer: uncovered] ..."}}
    out = hpl.summarize_readout(qs, plan, stock, rows)
    assert out["overall"] == {"n": 3, "plan": pytest.approx(50.0), "stock": pytest.approx(33.333333, rel=1e-4),
                              "plan_declines": 1}
    assert out["shapes"]["max-count"] == {"n": 1, "plan": 100.0, "stock": 0.0, "plan_declines": 0}
    assert out["shapes"]["prev-release"]["plan_declines"] == 1
    assert out["shapes"]["affected"]["n"] == 1


# --------------------------------------------------------------------------- paired statistics


def test_win_tie_loss():
    assert hpl.win_tie_loss([0.5, 0.0, -0.2, 0.0, 1.0]) == (2, 2, 1)


def test_sign_test_exact_two_sided():
    assert hpl.sign_test(0, 0) == 1.0
    assert hpl.sign_test(5, 0) == pytest.approx(2 * 0.5 ** 5)
    assert hpl.sign_test(3, 3) == pytest.approx(1.0)
    assert hpl.sign_test(8, 2) == pytest.approx(0.109375)


def test_paired_bootstrap_is_seeded_and_brackets_the_mean():
    d = [0.5, 0.25, -0.25, 1.0, 0.0, 0.5, 0.75, 0.0]
    a = hpl.paired_bootstrap(d, n_resamples=2000, seed=7)
    b = hpl.paired_bootstrap(d, n_resamples=2000, seed=7)
    assert a == b
    mean, lo, hi = a
    assert mean == pytest.approx(sum(d) / len(d))
    assert lo <= mean <= hi


def test_paired_bootstrap_ci_excludes_zero_for_a_clear_effect():
    mean, lo, hi = hpl.paired_bootstrap([1.0] * 12 + [0.5] * 4, n_resamples=2000, seed=1)
    assert lo > 0
    mean, lo, hi = hpl.paired_bootstrap([1.0, -1.0] * 8, n_resamples=2000, seed=1)
    assert lo < 0 < hi


def test_required_pairs_for_80_percent_power():
    # n = ((z_a/2 + z_b) * sd / mean)^2, z = 1.96 and 0.8416
    n = hpl.required_pairs(mean=0.2, sd=0.5)
    assert n == 50
    assert hpl.required_pairs(mean=0.0, sd=0.5) is None


def test_seeded_subset_is_reproducible_and_sorted():
    items = list(range(40))
    a = hpl.seeded_subset(items, 10, seed=20261119)
    assert a == hpl.seeded_subset(items, 10, seed=20261119)
    assert len(a) == 10 and a == sorted(a)
    assert hpl.seeded_subset(items, 99, seed=1) == items


def test_paired_report_per_shape_marks_ci_crossing_zero_inconclusive():
    qs = [{"product": "P", "index": i, "question": AGG} for i in range(6)] + \
         [{"product": "Q", "index": i, "question": PREV} for i in range(4)]
    plan = {f"P#{i}": 1.0 for i in range(6)} | {"Q#0": 1.0, "Q#1": 0.0, "Q#2": 1.0, "Q#3": 0.0}
    stock = {f"P#{i}": 0.0 for i in range(6)} | {"Q#0": 0.0, "Q#1": 1.0, "Q#2": 0.0, "Q#3": 1.0}
    rep = hpl.paired_report(qs, plan, stock, n_resamples=2000, seed=3)
    assert rep["overall"]["n"] == 10
    assert rep["shapes"]["max-count"]["verdict"] == "plan better"
    assert rep["shapes"]["prev-release"]["verdict"] == "inconclusive"
    assert rep["shapes"]["max-count"]["wtl"] == [6, 0, 0]


# --------------------------------------------------------------------------- stock coverage-off control


def test_control_questions_are_the_subset_in_question_order():
    qs = _qs()
    got = hpl.control_questions(qs, ["Q#0", "P#0"])
    assert [(q["product"], q["index"]) for q in got] == [("P", 0), ("Q", 0)]
    assert hpl.control_questions(qs, []) == []


def test_control_answer_fn_logs_plan_id_and_sets_coverage_off(tmp_path, monkeypatch):
    import os
    seen = {}

    def inner(question, scope):
        seen["cov"] = os.environ.get("NX_ANSWER_COVERAGE")
        return {"final_text": "x", "cost_usd": 0.01, "plan_id": 10, "coverage_state": "unassessed"}

    monkeypatch.delenv("NX_ANSWER_COVERAGE", raising=False)
    fn = hpl.control_answer_fn(inner, tmp_path / "meta.jsonl", {AGG: "P#0"})
    env = fn(AGG, "knowledge__herb-pool")
    assert env["plan_id"] == 10 and seen["cov"] == "0"
    assert os.environ.get("NX_ANSWER_COVERAGE") is None  # restored after the call
    meta = json.loads((tmp_path / "meta.jsonl").read_text().splitlines()[0])
    assert meta == {"key": "P#0", "plan_id": 10, "coverage_state": "unassessed"}
