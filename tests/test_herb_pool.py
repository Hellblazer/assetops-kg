"""Free checks for the full-pool HERB company harness: question selection, no gold in the corpus, the
cap arithmetic, resume, and score() with a stub extractor. Nothing here calls nx_answer, the claude CLI,
an LLM or OpenAI.

Tests marked ``herb`` read the HERB checkout (HERB_DIR, default ~/git/HERB) and FAIL, not skip, when it is
absent: a skipped selection test would pass over the 130 it exists to count.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from benchmark import run_herb_pool as hp

HERB_DIR = Path(os.environ.get("HERB_DIR") or "~/git/HERB").expanduser()
herb = pytest.mark.herb


@pytest.fixture(scope="module")
def products() -> list[tuple[str, dict]]:
    return hp.load_products(HERB_DIR)


# --------------------------------------------------------------------------- selection


@herb
def test_exactly_130_company_questions_in_product_file_order(products):
    qs = hp.company_questions(products)
    assert len(products) == 30
    assert len(qs) == hp.EXPECTED_COMPANY_QUESTIONS == 130
    # product-file order, then question order within a product
    names = [n for n, _ in products]
    assert [q["product"] for q in qs] == sorted((q["product"] for q in qs), key=names.index)
    for name, p in products:
        mine = [q for q in qs if q["product"] == name]
        idx = [q["index"] for q in mine]
        assert idx == sorted(idx)
        assert all(p["answerable_questions"][q["index"]]["type"] == "company" for q in mine)
        assert [q["question"] for q in mine] == [
            x["question"] for x in p["answerable_questions"] if x["type"] == "company"]
    assert names == sorted(names)
    assert all(q["ground_truth"] for q in qs)


def test_selection_ignores_other_types_and_unanswerable():
    prods = [("A", {"answerable_questions": [
        {"question": "q0", "type": "person", "ground_truth": ["x"]},
        {"question": "q1", "type": "company", "ground_truth": ["Acme"], "citations": ["c1"]},
        {"question": "q2", "type": "company", "ground_truth": ["Beta"]}],
        "unanswerable_questions": [{"question": "u", "type": "company"}]}),
        ("B", {"answerable_questions": [{"question": "q3", "type": "company", "ground_truth": ["Z"]}]})]
    qs = hp.company_questions(prods)
    assert [(q["product"], q["index"], q["question"]) for q in qs] == [
        ("A", 1, "q1"), ("A", 2, "q2"), ("B", 0, "q3")]


# --------------------------------------------------------------------------- no gold in the corpus


def test_flatten_reads_only_section_keys():
    poison = {k: "POISON-" + k for k in hp.GOLD_KEYS}
    base = {"slack": [{"id": "s1", "Channel": {"name": "c"},
                       "Message": {"User": {"userId": "u", "text": "hello"}}, "ThreadReplies": [
                           {"Message": {"User": {"userId": "v", "text": "a reply"}}}]}],
            "documents": [{"id": "d1", "type": "PRD", "content": "doc body"}],
            "meeting_transcripts": [{"id": "m1", "document_type": "t", "transcript": "talk"}],
            "meeting_chats": [{"id": "mc1", "text": "chat body"}],
            "urls": [{"id": "u1", "description": "d", "link": "http://x"}],
            "prs": [{"id": "p1", "title": "T", "summary": "S", "link": "l", "state": "open",
                     "user": {"login": "z"}, "reviews": []}]}
    clean = hp.flatten_pool_product(base)
    dirty = hp.flatten_pool_product({**base, **poison})
    assert [(r.id, r.source_type, r.text) for r in clean] == [(r.id, r.source_type, r.text) for r in dirty]
    assert {r.source_type for r in clean} == set(hp.SECTIONS)
    assert not any("POISON" in r.text for r in dirty)
    assert any("a reply" in r.text for r in clean if r.source_type == "slack")  # thread replies included


@herb
def test_pool_payload_has_no_questions_or_ground_truth(products):
    payload, rep = hp.build_pool(products, HERB_DIR)
    assert len(payload) > 38000  # non-vacuity: the pool is the real corpus
    texts = "\n".join(r["text"] for r in payload)
    leaked = []
    for name, p in products:
        qs = [q["question"] for q in p["answerable_questions"]] + list(p["unanswerable_questions"])
        leaked += [(name, q[:60]) for q in qs if q in texts]
    assert leaked == []
    # a question's own gold-bearing field names never appear as record structure
    assert not [r for r in payload if r["source_type"] not in (*hp.SECTIONS, "customer", "employee")]
    assert rep["entity_records"] > 0 and rep["missing_entities"] == {"customers": [], "employees": []}


@herb
def test_pool_counts_and_entities(products):
    payload, rep = hp.build_pool(products, HERB_DIR)
    per = rep["per_product"]
    assert "_entities" in per and set(per["_entities"]) == {"customer", "employee"}
    assert rep["artifact_records"] == sum(
        n for k, c in per.items() if k != "_entities" for n in c.values())
    assert rep["artifact_records"] == 38600  # the files; HERB's stated 39,190 is not reproducible from them
    cust_ids = {r["id"] for r in payload if r["source_type"] == "customer"}
    assert all(re.fullmatch(hp.ENTITY_PATTERN, c) for c in cust_ids) and len(cust_ids) == 120
    titles = [r["title"] for r in payload]
    assert len(titles) == len(set(titles))  # every record has its own catalog identity
    assert all(re.sub(r"#dup\d+$", "", r["title"]) == r["id"] for r in payload)  # citation ids preserved


# --------------------------------------------------------------------------- cap arithmetic and resume


def test_charge_for_unknown_cost():
    assert hp.charge_for(0.0123) == 0.0123
    assert hp.charge_for(0.0) == 0.0
    assert hp.charge_for(None) == hp.UNKNOWN_COST_USD == 0.30


def test_would_exceed_boundary():
    # stop before a question when charged + 0.30 > 40 - 1 = 39
    assert not hp.would_exceed(38.70)  # 38.70 + 0.30 = 39.00, not > 39
    assert hp.would_exceed(38.71)
    assert not hp.would_exceed(0.0)
    assert hp.would_exceed(39.0)
    assert hp.would_exceed(5.0, cap=5.0, held_back=0.0)  # parameters are honoured


def _qs(n: int) -> list[dict]:
    return [{"product": "P", "index": i, "question": f"q{i}", "ground_truth": ["Acme"]} for i in range(n)]


def _read(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_answer_loop_cap_stops_before_the_question(tmp_path):
    calls: list[str] = []

    def fn(q, scope):
        calls.append(q)
        return {"final_text": "Acme", "cost_usd": 20.0}

    st = hp.run_answers(_qs(5), tmp_path, fn)
    # 20 -> 40 would pass 39, so the 2nd call is made (20 + 0.30 <= 39) and the 3rd is refused (40 + .3 > 39)
    assert len(calls) == 2
    assert st["stopped"] and "PARTIAL" in st["stopped"]
    assert st["charged_total"] == pytest.approx(40.0)
    assert len(_read(tmp_path / "answers.jsonl")) == 2


def test_answer_loop_charges_unknown_cost_and_records_errors(tmp_path):
    def fn(q, scope):
        if q == "q0":
            return {"final_text": "x", "cost_usd": None}
        raise RuntimeError("boom")

    st = hp.run_answers(_qs(2), tmp_path, fn)
    rows = _read(tmp_path / "answers.jsonl")
    assert [r["charged"] for r in rows] == [0.30, 0.30]
    assert rows[1]["error"].startswith("RuntimeError") and rows[1]["final_text"] is None
    assert st["errors"] == 1 and st["charged_total"] == pytest.approx(0.60)
    for r in rows:
        assert set(r) >= {"product", "index", "question", "final_text", "cost", "duration", "error"}


def test_answer_loop_resume_skips_done_rows(tmp_path):
    calls: list[str] = []

    def fn(q, scope):
        calls.append(q)
        return {"final_text": "ok", "cost_usd": 0.1, "duration_ms": 5}

    hp.run_answers(_qs(5), tmp_path, fn, limit=2)
    assert calls == ["q0", "q1"]
    st = hp.run_answers(_qs(5), tmp_path, fn)  # resume: q0, q1 are skipped, an error row would be too
    assert calls == ["q0", "q1", "q2", "q3", "q4"]
    assert st["skipped_done"] == 2 and st["attempted"] == 3
    assert [r["index"] for r in _read(tmp_path / "answers.jsonl")] == [0, 1, 2, 3, 4]
    assert hp.Ledger(tmp_path / "ledger.jsonl").total() == pytest.approx(0.5)


def test_answer_loop_start_and_limit(tmp_path):
    seen: list[str] = []
    hp.run_answers(_qs(6), tmp_path, lambda q, s: seen.append(q) or {"final_text": "a", "cost_usd": 0.0},
                   start=2, limit=2)
    assert seen == ["q2", "q3"]


def test_ledger_counts_an_unsettled_reserve(tmp_path):
    led = hp.Ledger(tmp_path / "l.jsonl")
    led.append({"reserve_id": "a", "charged": 0.30, "settled": False})
    assert led.total() == pytest.approx(0.30)  # a crashed call is not free
    led.append({"reserve_id": "a", "charged": 0.01, "settled": True})
    led.append({"reserve_id": "b", "charged": 0.30, "settled": False})
    assert led.total() == pytest.approx(0.31)


def test_resumed_run_respects_cap_from_ledger(tmp_path):
    led = hp.Ledger(tmp_path / "ledger.jsonl")
    led.append({"reserve_id": "old", "charged": 38.9, "settled": True})
    calls: list[str] = []
    st = hp.run_answers(_qs(2), tmp_path, lambda q, s: calls.append(q) or {"final_text": "a"})
    assert calls == [] and st["stopped"]


# --------------------------------------------------------------------------- score with a stub extractor


def test_normalize_and_f1_match_herb_examples():
    assert hp.normalize("The Acme, Inc.!") == "acme inc"
    assert hp.f1_score_sets(["Acme", "Beta"], ["acme", "Beta"]) == 1.0
    assert hp.f1_score_sets(["Acme", "Beta"], ["Acme"]) == pytest.approx(2 * 1 * 0.5 / 1.5)
    assert hp.f1_score_sets(["Acme"], []) == 0.0
    assert hp.f1_score_sets(["Acme"], ["Zed"]) == 0.0


def test_parse_names():
    assert hp.parse_names('{"names": ["A", "B"]}') == ["A", "B"]
    assert hp.parse_names("```json\n{'names': ['A']}\n```") == ["A"]
    with pytest.raises(ValueError):
        hp.parse_names("no json here")
    with pytest.raises(ValueError):
        hp.parse_names('{"other": []}')


def test_score_on_hand_made_answers_with_stub_extractor(tmp_path):
    qs = [{"product": "A", "index": 0, "question": "qa0", "ground_truth": ["Acme", "Beta"]},
          {"product": "A", "index": 3, "question": "qa3", "ground_truth": ["Gamma"]},
          {"product": "B", "index": 1, "question": "qb1", "ground_truth": ["Delta"]},
          {"product": "B", "index": 2, "question": "qb2", "ground_truth": ["Eps"]},
          {"product": "C", "index": 0, "question": "qc0", "ground_truth": ["Zeta"]}]
    ans = tmp_path / "answers.jsonl"
    ans.write_text("\n".join(json.dumps(r) for r in [
        {"product": "A", "index": 0, "question": "qa0", "final_text": "Acme and Beta", "error": None},
        {"product": "A", "index": 3, "question": "qa3", "final_text": "Gamma, Zed", "error": None},
        {"product": "B", "index": 1, "question": "qb1", "final_text": None, "error": "timeout"},
        {"product": "B", "index": 2, "question": "qb2", "final_text": "Eps", "error": None}]) + "\n")
    called: list[str] = []

    def stub(question: str, content: str) -> list[str]:
        called.append(question)
        if question == "qb2":
            raise RuntimeError("extractor down")
        return [n.strip() for n in content.replace(" and ", ",").split(",")]

    res = hp.score_answers(qs, hp.load_answers(ans), stub)
    f1 = {r["key"]: r["f1"] for r in res["per_question"]}
    assert f1["A#0"] == 1.0
    assert f1["A#3"] == pytest.approx(2 * 0.5 * 1.0 / 1.5)
    assert f1["B#1"] == 0.0 and f1["B#2"] == 0.0 and f1["C#0"] == 0.0
    assert res["n"] == 5  # failed, errored and missing rows stay in the denominator
    assert res["overall_f1_x100"] == pytest.approx(100 * (1.0 + 2 / 3) / 5)
    assert res["per_product_f1_x100"] == {"A": pytest.approx(100 * (1.0 + 2 / 3) / 2), "B": 0.0, "C": 0.0}
    assert res["status_counts"] == {"ok": 2, "answer_error": 1, "extract_error": 1, "no_answer_row": 1}
    assert "qb1" not in called and "qc0" not in called  # no extractor call for rows with no answer


def test_score_cache_avoids_recalling_the_extractor():
    qs = [{"product": "A", "index": 0, "question": "q", "ground_truth": ["Acme"]}]
    ans = {("A", 0): {"product": "A", "index": 0, "final_text": "Acme", "error": None}}
    cache: dict[str, list[str]] = {}
    n = []
    hp.score_answers(qs, ans, lambda q, c: n.append(1) or ["Acme"], cache)
    res = hp.score_answers(qs, ans, lambda q, c: n.append(1) or ["WRONG"], cache)
    assert len(n) == 1 and res["overall_f1_x100"] == 100.0


def test_openai_extractor_refuses_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit) as e:
        hp.openai_extractor()
    assert "OPENAI_API_KEY" in str(e.value)


def test_regex_secondary_is_optional_and_reported():
    qs = [{"product": "A", "index": 0, "question": "q", "ground_truth": ["Acme"]}]
    ans = {("A", 0): {"product": "A", "index": 0, "final_text": "It was Acme corp.", "error": None}}
    res = hp.score_answers(qs, ans, lambda q, c: ["Acme"], regex_universe=["Acme", "Beta"])
    assert res["regex_overall_f1_x100"] == 100.0


@herb
def test_company_prompt_and_scoring_are_verbatim_copies_of_herb():
    src = (HERB_DIR / "code" / "evaluate.py").read_text()
    ns: dict = {"get_gpt4_response": lambda *a, **k: None}
    # the prompt: exec HERB's answer_to_companies with a capturing stub and compare the exact message
    captured: list[str] = []
    body = src[src.index("def answer_to_companies"):src.index("def answer_to_employee_ids")]
    ns["get_gpt4_response"] = lambda messages, response_format="text": (
        captured.append(messages[0]["content"]) or {"names": []})
    exec(body, ns)  # noqa: S102  HERB's own function, trusted checkout
    ns["answer_to_companies"]("Which company?", "Acme.")
    assert captured == [hp.company_prompt("Which company?", "Acme.")]
    # normalize / f1: same results as HERB's on a set of probes
    ns2: dict = {"re": re, "string": __import__("string")}
    exec(src[src.index("def normalize"):src.index("def turn_sample_lst2dataset")], ns2)  # noqa: S102
    exec(src[src.index("def f1_score_sets"):src.index("def evaluate_answerable")], ns2)  # noqa: S102
    probes = [(["The Acme"], ["acme", "Beta"]), (["A", "B", "C"], ["a"]), ([], []), (["x"], [])]
    for t, p in probes:
        assert hp.f1_score_sets(t, p) == ns2["f1_score_sets"](t, p)


# --------------------------------------------------------------------------- guards


def test_sandbox_guard(monkeypatch, tmp_path):
    for k in ("HERB_SANDBOX", "NEXUS_CONFIG_DIR", "NX_SERVICE_URL", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(SystemExit):
        hp.sandbox_guard(allow_llm_token=False)
    cfg = tmp_path / ".herb-sandbox" / "home" / ".config" / "nexus"
    monkeypatch.setenv("HERB_SANDBOX", "1")
    monkeypatch.setenv("NEXUS_CONFIG_DIR", str(cfg))
    assert hp.sandbox_guard(allow_llm_token=False) == cfg
    with pytest.raises(SystemExit, match="OAUTH"):
        hp.sandbox_guard(allow_llm_token=True)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "t")
    assert hp.sandbox_guard(allow_llm_token=True) == cfg
    with pytest.raises(SystemExit, match="never be able to spend"):
        hp.sandbox_guard(allow_llm_token=False)


def test_entity_pattern_check(tmp_path):
    assert not hp.entity_pattern_configured(tmp_path)
    (tmp_path / "config.yml").write_text("install:\n  mode: local\n")
    assert not hp.entity_pattern_configured(tmp_path)
    (tmp_path / "config.yml").write_text(
        "entity:\n  collections:\n    knowledge__herb-*: CUST-\\d{4}\n")
    assert hp.entity_pattern_configured(tmp_path)
