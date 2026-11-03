"""Unit tests for the HERB harness itself (RDR-147 Step 0, nexus-3lu23).

These need neither the HERB source nor a nexus sandbox: they pin the harness's own
behaviour with a fake ``nx``. The rungs that need the indexed corpus are in
tests/test_herb_ladder.py (``herb`` + ``integration``).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from benchmark import herb_ladder, multihop_resolve as mh, run_herb

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _private_cost_ledger(tmp_path, monkeypatch):
    """No unit test may append to the committed nx_answer cost ledger."""
    monkeypatch.setattr(herb_ladder, "_COST_LEDGER", tmp_path / "costs.jsonl")

# A Slack-shaped artifact whose first CUST id sits past char 200, where `nx search -c` cuts.
LONG_HIT = ("[channel:search-bugs] u1: " + "latency regression in the ranking service " * 6
            + "reported by CUST-0089 on the weekly call.")
assert LONG_HIT.index("CUST-0089") > 200


class FakeNx:
    """Stands in for ``nx search``: with ``-c`` it truncates each hit at 200 characters and prints
    text (the real compact mode, search_cmd.py:779); with ``--json`` it prints full chunk text."""

    def __init__(self, content: str = LONG_HIT, title: str = "20260805-1-abc"):
        self.content, self.title, self.calls = content, title, []

    def __call__(self, cmd, **kw):
        self.calls.append(cmd)
        if "--json" in cmd:
            out = json.dumps([{"title": self.title, "content": self.content}])
        elif "-c" in cmd:
            out = self.content[:200] + "\n"
        else:
            out = self.content
        return subprocess.CompletedProcess(cmd, 0, stdout=out.encode(), stderr=b"")


def test_read_is_not_truncated_at_200_chars(monkeypatch):
    """Fails if any read goes back through `nx search -c`: the CUST id past char 200 is lost."""
    fake = FakeNx()
    monkeypatch.setattr(mh.subprocess, "run", fake)
    text = mh._search("any question", k=20)
    assert "CUST-0089" in text
    assert mh._RE_CUST.findall(text) == ["CUST-0089"]
    assert all("--json" in c and "-c" not in c for c in fake.calls)


def test_truncating_fake_would_lose_the_id():
    """Control: the fake really truncates in compact mode, so the test above can fail."""
    out = FakeNx()(["nx", "search", "q", "-c"]).stdout.decode()
    assert "CUST-0089" not in out


def test_answer_company_finds_id_past_char_200(monkeypatch):
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if "--where" in cmd:  # hop 3: the exact customer record
            rec = "Customer CUST-0089: contact A B, Mgr at company FusionTech."
            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps([{"title": "CUST-0089", "content": rec}]).encode(), stderr=b"")
        return FakeNx()(cmd)

    monkeypatch.setattr(mh.subprocess, "run", fake_run)
    assert mh.answer_company("which companies", k=20) == {"FusionTech"}


def test_oracle_filter_drops_hits_outside_the_set(monkeypatch):
    monkeypatch.setattr(mh.subprocess, "run", FakeNx())
    assert mh.answer_company("q", k=20, only_artifacts={"some-other-id"}) == set()


def test_nonzero_exit_raises(monkeypatch):
    def boom(cmd, **kw):
        raise subprocess.CalledProcessError(2, cmd, output=b"", stderr=b"engine down")
    monkeypatch.setattr(mh.subprocess, "run", boom)
    with pytest.raises(mh.NxError, match="engine down"):
        mh.search_hits("q", 5)


def test_missing_collection_warning_raises(monkeypatch):
    """nx exits 0 with a warning and no JSON when --corpus matches nothing."""
    def warn(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0, stdout=b"no matching collections found\n", stderr=b"")
    monkeypatch.setattr(mh.subprocess, "run", warn)
    with pytest.raises(mh.NxError, match="non-JSON"):
        mh.search_hits("q", 5)


def test_timeout_raises(monkeypatch):
    def slow(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 1)
    monkeypatch.setattr(mh.subprocess, "run", slow)
    with pytest.raises(mh.NxError, match="timed out"):
        mh.search_hits("q", 5)


def test_empty_hit_list_is_not_an_error(monkeypatch):
    monkeypatch.setattr(mh.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout=b"[]", stderr=b""))
    assert mh.search_hits("q", 5) == []


@pytest.mark.parametrize("title,expected", [
    ("20260805-1-abc", "20260805-1-abc"),
    ("20260805-1-abc:chunk-2", "20260805-1-abc"),
    ("nltk-url#dup2", "nltk-url"),
])
def test_artifact_id_from_title(title, expected):
    assert mh.artifact_id_from_title(title) == expected


def test_unique_titles_disambiguates_a_repeated_id():
    recs = [run_herb.ArtifactRecord("u1", "urls", "a"), run_herb.ArtifactRecord("u1", "urls", "b"),
            run_herb.ArtifactRecord("u2", "urls", "c")]
    titles = [t for _, t in run_herb.unique_titles(recs)]
    assert titles == ["u1", "u1#dup2", "u2"]
    assert len(set(titles)) == 3


def test_score_set_conventions():
    s = herb_ladder.score_set({"A", "b"}, {"a", "c"})
    assert (s["hit"], s["recall"], s["precision"]) == (1, 0.5, 0.5)
    empty = herb_ladder.score_set(set(), {"a"})
    assert (empty["recall"], empty["precision"], empty["f1"]) == (0.0, 0.0, 0.0)


def test_summarize_splits_issue_and_aggregation():
    rows = [
        {"kind": "issue", "recall": 1.0, "precision": 0.5, "f1": 0.6},
        {"kind": "issue", "recall": 0.0, "precision": 0.0, "f1": 0.0},
        {"kind": "aggregation", "recall": 1.0, "precision": 0.1, "f1": 0.2},
    ]
    s = herb_ladder.summarize(rows)
    assert s["all7"]["n"] == 3 and s["issue5"]["n"] == 2 and s["agg2"]["n"] == 1
    assert s["issue5"]["recall"] == 0.5 and s["agg2"]["precision"] == 0.1


def test_extract_companies_whole_word_any_case():
    universe = ["TechCorp", "SmartData", "DataSolutions"]
    text = "Affected: techcorp, SmartData. Not SmartDatabase and not MegaTechCorpX."
    assert herb_ladder.extract_companies(text, universe) == {"TechCorp", "SmartData"}


def test_aggregation_questions_are_classified_apart():
    assert herb_ladder._is_aggregation("Find the name of company that has the maximum number of resolved bugs in SearchFlow")
    assert not herb_ladder._is_aggregation("Find the names of companies were affected by issues in the Search API")


def test_script_rung_never_calls_nx_answer(monkeypatch):
    """The script is a separate baseline row; running it must not touch the product."""
    def forbidden(*a, **k):
        raise AssertionError("nx_answer called from the script rung")
    monkeypatch.setattr(herb_ladder, "nx_answer_call", forbidden)
    monkeypatch.setattr(herb_ladder, "search_hits", lambda q, k: [{"title": "x", "content": "CUST-0001"}])
    monkeypatch.setattr(herb_ladder, "companies_for_ids", lambda ids: {"FusionTech"} if ids else set())
    qs = [{"question": "q", "gold": ["FusionTech"], "citations": [], "kind": "issue"}]
    rows = herb_ladder.run_script_rows(qs)
    assert rows[0]["recall"] == 1.0 and rows[0]["cust_ids"] == ["CUST-0001"]


def test_nx_answer_rung_scores_precision_and_stops_at_the_cost_cap(monkeypatch):
    envs = iter([{"final_text": "FusionTech and TechCorp", "cost_usd": 1.0, "plan_id": 1, "step_count": 2},
                 {"final_text": "FusionTech", "cost_usd": 1.0, "plan_id": 1, "step_count": 2}])
    monkeypatch.setattr(herb_ladder, "nx_answer_call", lambda *a, **k: next(envs))
    qs = [{"question": f"q{i}", "gold": ["FusionTech"], "citations": [], "kind": "issue"} for i in range(4)]
    rows, meta = herb_ladder.run_nx_answer_rows(qs, ["FusionTech", "TechCorp"], cap_usd=2.0)
    assert (rows[0]["recall"], rows[0]["precision"]) == (1.0, 0.5)
    assert rows[1]["precision"] == 1.0
    assert [("skipped" in r) for r in rows] == [False, False, True, True]
    ledger = [json.loads(line) for line in herb_ladder._COST_LEDGER.read_text().splitlines()]
    assert [e["cost_usd"] for e in ledger] == [1.0, 1.0]
    assert meta["complete"] is False and meta["total_cost_usd"] == 2.0


def test_manifest_requires_versions(monkeypatch):
    monkeypatch.setattr(run_herb, "_sh", lambda cmd: "")
    with pytest.raises(mh.NxError, match="cannot record versions"):
        run_herb.service_versions()


def test_service_versions_parse(monkeypatch):
    status = ("Storage Service Status\n  host: 127.0.0.1\n  port: 53590\n  service_release_version: 0.1.175\n"
              "  service_app_version: 1.0-SNAPSHOT\n  embedding_mode: onnx-local\n  embedding_models: bge-base-en-v15-768\n")
    monkeypatch.setattr(run_herb, "_sh", lambda cmd: "nx, version 7.94.0" if cmd[:2] == ["nx", "--version"] else status)
    v = run_herb.service_versions()
    assert v["engine_release_version"] == "0.1.175" and v["nx_version"] == "nx, version 7.94.0"
    assert v["service_endpoint"] == "127.0.0.1:53590"


def test_sandbox_script_env_allowlist():
    """The sandbox wrapper's own checks: no ambient key reaches the child, the config dir is the
    sandbox's, the real HOME and live config are refused as sandbox paths."""
    r = subprocess.run(["bash", str(REPO / "benchmark" / "herb_sandbox.sh"), "--self-test"],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "self-test: OK" in r.stdout



def test_p3_by_kind_separates_the_aggregation_questions():
    rows = [
        {"question": "Find the names of companies were affected by issues in the Search API", "precision": 1.0, "n_surfaced": 4},
        {"question": "Find the names of companies were affected by issues in the UI", "precision": 0.5, "n_surfaced": 6},
        {"question": "Find the name of company that has the maximum number of resolved bugs in SearchFlow", "precision": 0.0, "n_surfaced": 0},
    ]
    s = herb_ladder.p3_by_kind(rows)
    assert (s["issue5"]["n"], s["agg2"]["n"], s["all7"]["n"]) == (2, 1, 3)
    assert s["issue5"]["mean_precision"] == 0.75 and s["agg2"]["mean_precision"] == 0.0
    assert s["all7"]["mean_precision"] == 0.5
