"""Unit tests for the HERB harness itself (RDR-147 Step 0, nexus-3lu23).

These need neither the HERB source nor a nexus sandbox: they pin the harness's own
behaviour with a fake ``nx``. No test here executes a real interpreter or nx: the P2 driver
runs in-process against stub ``nexus`` modules, and the spawn tests replace ``subprocess.run``. The rungs that need the indexed corpus are in
tests/test_herb_ladder.py (``herb`` + ``integration``).
"""

from __future__ import annotations

import json
import os
import pwd
import subprocess
import sys
import types
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
    ledger = _settled(_ledger_rows())
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


# --- P2: the `resolve` step over the 29 members (RDR-147 Step 3, nexus-v9v5r) ---
#
# Fake-resolver pins of how the harness CALLS nexus's resolve_keys and reads its answer, the
# sandbox guard (host side and the driver's own re-assertion), and the scorecard patching. The one
# live test is in tests/test_herb_ladder.py.

REV = "3e608f6b2"
DIST = "7.94.1"


class _FakeResolver:
    """Stands in for the resolve_keys driver: records every call, answers from the gold members."""

    def __init__(self, members, *, missing=(), ambiguous=None, company_for=None, text_for=None,
                 omit=(), reported_calls=None):
        self.members, self.calls = members, []
        self.missing, self.ambiguous = list(missing), dict(ambiguous or {})
        self.company_for, self.text_for = company_for or {}, text_for or {}
        self.omit, self.reported_calls = set(omit), reported_calls

    def __call__(self, tokens):
        self.calls.append(list(tokens))
        by_token = {t: {"tumbler": f"1.1.{i}", "ids": [f"h{i}"], "collections": ["knowledge__herb-searchflow"]}
                    for i, t in enumerate(tokens, 1) if t not in self.omit}
        texts = {t: self.text_for.get(
            t, f"Customer {t}: contact N, R at company {self.company_for.get(t, self.members[t])}.")
            for t in tokens}
        return {
            "service_url": "http://127.0.0.1:1", "lookups": len(set(tokens)), "resolve_s": 0.0,
            "resolve_calls": len(self.calls) if self.reported_calls is None else self.reported_calls,
            "nexus_revision": REV, "nexus_dist_version": DIST,
            "texts": texts,
            "result": {"by_token": by_token, "missing": self.missing, "ambiguous": self.ambiguous,
                       "tumblers": [], "ids": [], "collections": []},
        }


@pytest.fixture(scope="module")
def members():
    return json.loads((REPO / "benchmark" / "fixtures" / "herb" / "p2_member_customers.json").read_text())


def test_p2_sends_all_29_tokens_in_one_call(members):
    fake = _FakeResolver(members)
    detail = herb_ladder.score_p2_resolution(members, resolver=fake)
    assert len(members) == 29
    assert fake.calls == [list(members)]  # one call, every token, in fixture order
    assert detail["n_tokens"] == 29 and detail["resolve_calls"] == 1
    assert detail["accuracy"] == 1.0 and detail["passed"] is True
    assert herb_ladder.p2_rung(detail).status == "GREEN"


def test_p2_resolve_calls_is_what_the_resolver_counted(members):
    detail = herb_ladder.score_p2_resolution(members, resolver=_FakeResolver(members, reported_calls=3))
    assert detail["resolve_calls"] == 3


def test_p2_reports_missing_and_ambiguous_from_the_result(members):
    gone, twice = "CUST-0018", "CUST-0027"
    fake = _FakeResolver(members, missing=[gone], ambiguous={twice: ["1.1.5", "1.2.9"]})
    detail = herb_ladder.score_p2_resolution(members, resolver=fake)
    assert detail["missing"] == [gone]
    assert detail["ambiguous"] == {twice: ["1.1.5", "1.2.9"]}


@pytest.mark.parametrize("kw", [{"missing": ["CUST-0018"]}, {"ambiguous": {"CUST-0018": ["1.1.1", "1.2.1"]}}])
def test_p2_fails_when_missing_or_ambiguous_is_non_empty(members, kw):
    # Every token still reads as right in by_token/texts, so only the two lists can fail the rung.
    detail = herb_ladder.score_p2_resolution(members, resolver=_FakeResolver(members, **kw))
    assert detail["accuracy"] == 1.0
    assert detail["passed"] is False
    assert herb_ladder.p2_rung(detail).status == "RED"


def test_p2_rung_target_says_the_lists_must_be_empty(members):
    detail = herb_ladder.score_p2_resolution(members, resolver=_FakeResolver(members))
    assert herb_ladder.p2_rung(detail).target == "==1.0, missing=[] ambiguous={}"


def test_p2_a_record_naming_another_company_is_wrong(members):
    cid = "CUST-0018"
    other = "NotTheCompany"
    assert members[cid] != other
    detail = herb_ladder.score_p2_resolution(members, resolver=_FakeResolver(members, company_for={cid: other}))
    assert detail["wrong"] == [cid]
    assert detail["accuracy"] == 28 / 29
    assert detail["passed"] is False


def test_p2_a_token_absent_from_by_token_is_wrong(members):
    cid = "CUST-0018"
    detail = herb_ladder.score_p2_resolution(members, resolver=_FakeResolver(members, omit=[cid]))
    assert detail["wrong"] == [cid]
    assert detail["accuracy"] == 28 / 29
    assert detail["passed"] is False


@pytest.mark.parametrize("text", [
    "Customer CUST-0099: contact N, R at company {co}.",           # another customer's record
    "See CUST-0018 for details. Customer CUST-0099: contact N, R at company {co}.",  # id not the record's own
    "contact N, R at company {co}.",                                 # no id at all
    "Customer CUST-00180: contact N, R at company {co}.",           # id is only a prefix of the record's
])
def test_p2_a_text_not_opening_with_the_ingest_shape_for_the_id_is_wrong(members, text):
    cid = "CUST-0018"
    text = text.format(co=members[cid])
    detail = herb_ladder.score_p2_resolution(members, resolver=_FakeResolver(members, text_for={cid: text}))
    assert detail["wrong"] == [cid]
    assert detail["passed"] is False


def test_p2_correctness_check_is_scoped_to_the_herb_ingest_format():
    assert "HERB" in herb_ladder.score_p2_resolution.__doc__
    assert "Customer {cid}:" in herb_ladder.score_p2_resolution.__doc__


def test_p2_accuracy_wrapper_goes_through_the_resolve_call(members, monkeypatch):
    fake = _FakeResolver(members)
    monkeypatch.setattr(herb_ladder, "resolve_p2_tokens", fake)
    monkeypatch.setattr(herb_ladder, "resolve_company", lambda cid: pytest.fail("P2 must not use the per-token lookup"))
    assert herb_ladder.score_p2_resolution_accuracy(members) == 1.0
    assert len(fake.calls) == 1


# --- the sandbox guard: pure rule, the host refusal before any spawn, the driver's own re-assertion ---


@pytest.fixture
def world(tmp_path):
    """A sandbox-shaped world: real home, sandbox home, an interpreter outside both."""
    real_home = tmp_path / "real"
    tools = real_home / ".local" / "share" / "nexus" / "tools"
    (tools / "gen1" / "bin").mkdir(parents=True)
    (tools / "current").symlink_to(tools / "gen1")
    sb_home = tmp_path / "repo" / ".herb-sandbox" / "home"
    cfg = sb_home / ".config" / "nexus"
    cfg.mkdir(parents=True)
    wheel_py = tmp_path / "wheelvenv" / "bin" / "python3"
    wheel_py.parent.mkdir(parents=True)
    wheel_py.write_text("not an executable\n")  # a spawn of this fails with PermissionError, runs nothing
    wheel_py.chmod(0o644)
    env = {"HERB_SANDBOX": "1", "HOME": str(sb_home), "NEXUS_CONFIG_DIR": str(cfg), "HERB_NEXUS_REV": REV}
    return types.SimpleNamespace(real_home=real_home, tools=tools, sb_home=sb_home, cfg=cfg,
                                 wheel_py=wheel_py, env=env, tmp=tmp_path)


def _violations(world, env=None, interpreter=None):
    return herb_ladder.p2_sandbox_violations(
        world.env if env is None else env, interpreter or world.wheel_py, world.real_home)


def test_guard_accepts_a_sandbox_env_with_a_wheel_interpreter(world):
    assert _violations(world) == []


def _drop(key):
    return lambda w: {k: v for k, v in w.env.items() if k != key}


def _sandbox_named_link(key, leaf):
    """A path that names a .herb-sandbox directory but is a symlink to the real home: realpath must
    see through it."""
    def make(w):
        link = w.tmp / "decoy" / ".herb-sandbox" / leaf
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to(w.real_home, target_is_directory=True)
        return {**w.env, key: str(link)}
    return make


_BAD_ENVS = {
    "no HERB_SANDBOX": (_drop("HERB_SANDBOX"), "HERB_SANDBOX"),
    "HERB_SANDBOX not 1": (lambda w: {**w.env, "HERB_SANDBOX": "yes"}, "HERB_SANDBOX"),
    "no NEXUS_CONFIG_DIR": (_drop("NEXUS_CONFIG_DIR"), "NEXUS_CONFIG_DIR"),
    "config dir is the real one": (lambda w: {**w.env, "NEXUS_CONFIG_DIR": str(w.real_home / ".config" / "nexus")},
                                   "NEXUS_CONFIG_DIR"),
    "config dir only looks like the sandbox": (
        lambda w: {**w.env, "NEXUS_CONFIG_DIR": str(w.tmp / "not.herb-sandbox" / "cfg")}, "NEXUS_CONFIG_DIR"),
    "config dir climbs out of the sandbox": (
        lambda w: {**w.env, "NEXUS_CONFIG_DIR": str(w.sb_home) + "/../../../real/.config/nexus"}, "NEXUS_CONFIG_DIR"),
    "no HOME": (_drop("HOME"), "HOME"),
    "HOME is the real home": (lambda w: {**w.env, "HOME": str(w.real_home)}, "HOME"),
    "no HERB_NEXUS_REV": (_drop("HERB_NEXUS_REV"), "HERB_NEXUS_REV"),
    "empty HERB_NEXUS_REV": (lambda w: {**w.env, "HERB_NEXUS_REV": ""}, "HERB_NEXUS_REV"),
    "HERB_NEXUS_REV not a revision": (lambda w: {**w.env, "HERB_NEXUS_REV": "develop"}, "HERB_NEXUS_REV"),
    "service URL set": (lambda w: {**w.env, "NX_SERVICE_URL": "http://127.0.0.1:7777"}, "NX_SERVICE_URL"),
    "service token set": (lambda w: {**w.env, "NX_SERVICE_TOKEN": "t"}, "NX_SERVICE_TOKEN"),
    "HOME is a sandbox-named symlink to the real home": (_sandbox_named_link("HOME", "home"), "HOME"),
    "config dir is a sandbox-named symlink to the real one": (_sandbox_named_link("NEXUS_CONFIG_DIR", "cfg"),
                                                              "NEXUS_CONFIG_DIR"),
}


@pytest.mark.parametrize("name", list(_BAD_ENVS))
def test_guard_refuses_a_bad_env(world, name):
    make, key = _BAD_ENVS[name]
    out = _violations(world, env=make(world))
    assert out and any(key in v for v in out), out


def test_guard_refuses_the_default_installed_interpreter(world):
    assert any("interpreter" in v for v in _violations(world, interpreter=world.tools / "current" / "bin" / "python3"))


def test_guard_refuses_the_installed_interpreter_through_its_generation_path(world):
    # the same interpreter named by the generation directory `current` points at
    assert any("interpreter" in v for v in _violations(world, interpreter=world.tools / "gen1" / "bin" / "python3"))


def test_guard_resolves_a_symlinked_route_to_the_installed_interpreter(world):
    alias = world.tmp / "alias"
    alias.symlink_to(world.tools / "gen1")  # the installed bin dir under another name
    assert any("interpreter" in v for v in _violations(world, interpreter=alias / "bin" / "python3"))


@pytest.mark.parametrize("name", list(_BAD_ENVS) + ["default interpreter"])
def test_resolve_p2_tokens_refuses_before_any_spawn(world, monkeypatch, name):
    for k in ("HERB_SANDBOX", "HOME", "NEXUS_CONFIG_DIR", "HERB_NEXUS_REV"):
        monkeypatch.delenv(k, raising=False)
    if name == "default interpreter":
        env, interp = world.env, world.tools / "current" / "bin" / "python3"
        interp.write_text("not an executable\n")
    else:
        env, interp = _BAD_ENVS[name][0](world), world.wheel_py
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(herb_ladder, "CONEXUS_PYTHON", interp)
    monkeypatch.setattr(herb_ladder, "_real_home", lambda: str(world.real_home))
    monkeypatch.setattr(herb_ladder.subprocess, "run",
                        lambda *a, **k: pytest.fail("the guard let resolve_p2_tokens spawn a process"))
    with pytest.raises(herb_ladder.NxError, match="refus"):
        herb_ladder.resolve_p2_tokens(["CUST-0001"])


def _drop_service_env(monkeypatch):
    for k in [k for k in os.environ if k.startswith("NX_SERVICE_")]:
        monkeypatch.delenv(k)


@pytest.fixture(autouse=True)
def _no_ambient_service_env(monkeypatch):
    """An operator's shell may export NX_SERVICE_*; the P2 guard refuses on them, so no test may
    inherit them (a case that wants one sets it itself)."""
    _drop_service_env(monkeypatch)


def test_resolve_p2_tokens_spawns_exactly_once_for_a_sandbox_env(world, monkeypatch):
    for k, v in world.env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(herb_ladder, "CONEXUS_PYTHON", world.wheel_py)
    monkeypatch.setattr(herb_ladder, "_real_home", lambda: str(world.real_home))
    spawned = []

    def fake_run(cmd, **kw):
        spawned.append(cmd)
        line = herb_ladder._NXA_SENTINEL + json.dumps({"result": {}, "texts": {}})
        return subprocess.CompletedProcess(cmd, 0, stdout=line.encode(), stderr=b"")

    monkeypatch.setattr(herb_ladder.subprocess, "run", fake_run)
    assert herb_ladder.resolve_p2_tokens(["CUST-0001"]) == {"result": {}, "texts": {}}
    assert len(spawned) == 1 and spawned[0][0] == str(world.wheel_py)


# The driver runs in-process against stub nexus modules: its own guard and loopback refusal are
# exercised without any interpreter or nx being executed.


class _StubNexus:
    def __init__(self, monkeypatch, base_url, dist="9.9.9"):
        self.constructed, self.resolve_calls, self.lookups_seen = [], [], 0
        stub = self

        class Client:
            def __init__(self):
                stub.constructed.append(1)
                self._base_url = base_url

            def find_by_title_exact(self, *a, **kw):
                return []

        def resolve_keys(catalog, tokens):
            stub.resolve_calls.append(list(tokens))
            for t in tokens:
                catalog.find_by_title_exact(t)
            return {"by_token": {t: {"ids": [f"h-{t}"], "collections": ["c"]} for t in tokens},
                    "missing": [], "ambiguous": {}}

        def store_get_many(ids, colls, **kw):
            return {"contents": [f"Customer {i[2:]}: x at company Acme." for i in ids]}

        mods = {"nexus": types.ModuleType("nexus"),
                "nexus.catalog": types.ModuleType("nexus.catalog"),
                "nexus.catalog.http_catalog_client": types.ModuleType("nexus.catalog.http_catalog_client"),
                "nexus.mcp": types.ModuleType("nexus.mcp"),
                "nexus.mcp.core": types.ModuleType("nexus.mcp.core"),
                "nexus.plans": types.ModuleType("nexus.plans"),
                "nexus.plans.resolve_step": types.ModuleType("nexus.plans.resolve_step")}
        mods["nexus.catalog.http_catalog_client"].HttpCatalogClient = Client
        mods["nexus.mcp.core"].store_get_many = store_get_many
        mods["nexus.plans.resolve_step"].resolve_keys = resolve_keys
        for name, mod in mods.items():
            monkeypatch.setitem(sys.modules, name, mod)
        import importlib.metadata as md
        monkeypatch.setattr(md, "version", lambda name: dist)


def _run_driver(tmp_path, monkeypatch, tokens=("CUST-0001", "CUST-0002")):
    tf = tmp_path / "tokens.json"
    tf.write_text(json.dumps(list(tokens)))
    monkeypatch.setattr(sys, "argv", ["-c", str(tf)])
    # dont_inherit: the driver runs as `python -c` text, without this module's `from __future__` flags
    exec(compile(herb_ladder._RESOLVE_DRIVER, "<resolve-driver>", "exec", dont_inherit=True),
         {"__name__": "__driver__"})


def _driver_world(world, monkeypatch, interpreter=None):
    for k in ("HERB_SANDBOX", "HOME", "NEXUS_CONFIG_DIR", "HERB_NEXUS_REV"):
        monkeypatch.delenv(k, raising=False)
    for k, v in world.env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(sys, "executable", str(interpreter or world.wheel_py))
    # the driver reads the real home from the password database; point it at the test's
    monkeypatch.setattr(pwd, "getpwuid", lambda uid: types.SimpleNamespace(pw_dir=str(world.real_home)))


def test_driver_refuses_a_non_loopback_service_url(world, monkeypatch, tmp_path):
    _driver_world(world, monkeypatch)
    stub = _StubNexus(monkeypatch, "https://api.conexus-nexus.com")
    with pytest.raises(SystemExit, match="non-loopback"):
        _run_driver(tmp_path, monkeypatch)
    assert stub.constructed == [1] and stub.resolve_calls == []  # refused before the resolve call


@pytest.mark.parametrize("name", list(_BAD_ENVS) + ["default interpreter"])
def test_driver_reasserts_the_sandbox_before_building_any_client(world, monkeypatch, tmp_path, name):
    interp = None
    if name == "default interpreter":
        interp = world.tools / "current" / "bin" / "python3"
    _driver_world(world, monkeypatch, interpreter=interp)
    if name != "default interpreter":
        for k in ("HERB_SANDBOX", "HOME", "NEXUS_CONFIG_DIR", "HERB_NEXUS_REV"):
            monkeypatch.delenv(k, raising=False)
        for k, v in _BAD_ENVS[name][0](world).items():
            monkeypatch.setenv(k, v)
    stub = _StubNexus(monkeypatch, "http://127.0.0.1:9")
    with pytest.raises(SystemExit, match="refus"):
        _run_driver(tmp_path, monkeypatch)
    assert stub.constructed == [] and stub.resolve_calls == []


def test_driver_counts_resolve_calls_and_reports_the_nexus_it_ran(world, monkeypatch, tmp_path, capsys):
    _driver_world(world, monkeypatch)
    stub = _StubNexus(monkeypatch, "http://127.0.0.1:9", dist="9.9.9")
    _run_driver(tmp_path, monkeypatch, tokens=("CUST-0001", "CUST-0002", "CUST-0003"))
    line = next(l for l in capsys.readouterr().out.splitlines() if l.startswith(herb_ladder._NXA_SENTINEL))
    out = json.loads(line[len(herb_ladder._NXA_SENTINEL):])
    assert out["resolve_calls"] == len(stub.resolve_calls) == 1
    assert out["lookups"] == 3
    assert out["nexus_revision"] == REV and out["nexus_dist_version"] == "9.9.9"
    assert out["service_url"] == "http://127.0.0.1:9"


# --- provenance and scorecard patching ---


def _card(*phases):
    return {"manifest": {"timestamp_utc": "t0", "nx_version": "nx, version 7.94.0"},
            "rungs": [{"phase": p, "metric": "m", "value": None, "target": "x", "status": "RED"} for p in phases],
            "p2_resolve": None}


_VERSIONS = {"nx_version": "nx, version 7.94.1", "engine_release_version": "0.1.175"}


def test_patch_replaces_exactly_the_one_p2_rung_and_marks_the_manifest(members):
    detail = herb_ladder.score_p2_resolution(members, resolver=_FakeResolver(members))
    card = herb_ladder.patch_scorecard_p2(_card("P1", "P2", "P3"), detail, _VERSIONS, "2026-11-04T00:00:00+00:00",
                                          "p2-resolve-2026-11-04.json")
    assert [r["phase"] for r in card["rungs"]] == ["P1", "P2", "P3"]
    assert card["rungs"][1]["status"] == "GREEN" and card["rungs"][0]["status"] == "RED"
    marker = card["manifest"]["patched_rungs"]["P2"]
    assert marker["file"] == "p2-resolve-2026-11-04.json"
    assert marker["nexus_revision"] == REV and marker["nexus_dist_version"] == DIST
    assert marker["nx_version"] == "nx, version 7.94.1"
    assert card["manifest"]["nx_version"] == "nx, version 7.94.0"  # the other rungs' versions stay as measured


@pytest.mark.parametrize("phases", [("P1", "P3"), ("P1", "P2", "P2")])
def test_patch_refuses_unless_exactly_one_p2_rung(members, phases):
    detail = herb_ladder.score_p2_resolution(members, resolver=_FakeResolver(members))
    with pytest.raises(ValueError, match="exactly one P2 rung"):
        herb_ladder.patch_scorecard_p2(_card(*phases), detail, _VERSIONS, "t", "f")


def test_p2_resolve_entry_has_one_shape_on_both_paths(members):
    detail = herb_ladder.score_p2_resolution(members, resolver=_FakeResolver(members))
    full = herb_ladder.p2_resolve_entry(detail, _VERSIONS, "t", None)       # build(): no per-run file
    only = herb_ladder.p2_resolve_entry(detail, _VERSIONS, "t", "p2-x.json")  # --p2-resolve
    assert set(full) == set(only)
    for key in ("nexus_revision", "nexus_dist_version", "nx_version", "engine_release_version",
                "resolve_calls", "timestamp_utc", "file"):
        assert key in full, key


def test_emit_p2_resolve_records_the_revision_and_leaves_baselines_alone(members, monkeypatch, tmp_path):
    results = tmp_path / "herb"
    results.mkdir()
    (results / "herb-scorecard.json").write_text(json.dumps(_card("P1", "P2")))
    baselines = results / "step0-baselines.json"
    baselines.write_text('{"recorded": true}')
    monkeypatch.setattr(herb_ladder, "_RESULTS_DIR", results)
    monkeypatch.setattr(herb_ladder, "_SCORECARD", results / "herb-scorecard.json")
    monkeypatch.setattr(herb_ladder, "_BASELINES", baselines)
    monkeypatch.setattr(herb_ladder, "service_versions", lambda: dict(_VERSIONS))
    monkeypatch.setattr(herb_ladder, "_load_gold", lambda: {"members": members})
    row = herb_ladder.emit_p2_resolve(date="2026-11-04", resolver=_FakeResolver(members))
    assert baselines.read_text() == '{"recorded": true}'
    dated = json.loads((results / "p2-resolve-2026-11-04.json").read_text())
    assert dated["nexus_revision"] == REV and dated["nexus_dist_version"] == DIST == row["nexus_dist_version"]
    card = json.loads((results / "herb-scorecard.json").read_text())
    assert card["p2_resolve"]["nexus_revision"] == REV and card["p2_resolve"]["file"] == "p2-resolve-2026-11-04.json"
    assert card["rungs"][1]["status"] == "GREEN"


def test_emit_p2_resolve_writes_nothing_when_the_scorecard_has_no_single_p2_rung(members, monkeypatch, tmp_path):
    results = tmp_path / "herb"
    results.mkdir()
    card_path = results / "herb-scorecard.json"
    card_path.write_text(json.dumps(_card("P1", "P3")))
    before = card_path.read_text()
    monkeypatch.setattr(herb_ladder, "_RESULTS_DIR", results)
    monkeypatch.setattr(herb_ladder, "_SCORECARD", card_path)
    monkeypatch.setattr(herb_ladder, "service_versions", lambda: dict(_VERSIONS))
    monkeypatch.setattr(herb_ladder, "_load_gold", lambda: {"members": members})
    with pytest.raises(ValueError):
        herb_ladder.emit_p2_resolve(date="2026-11-04", resolver=_FakeResolver(members))
    assert not (results / "p2-resolve-2026-11-04.json").exists(), "a dated row with no scorecard patch"
    assert card_path.read_text() == before


def test_full_build_gives_p2_resolve_the_same_shape(members, monkeypatch):
    """build() runs every rung; stub the others so only the P2 plumbing is exercised."""
    monkeypatch.setattr(herb_ladder, "_load_gold", lambda: {
        "members": members, "cust_to_company": {"CUST-0001": "Acme"}, "mentions": {}, "q_to_custs": {}})
    monkeypatch.setattr(herb_ladder, "_load_product", lambda: {"answerable_questions": []})
    monkeypatch.setattr(herb_ladder, "service_versions", lambda: dict(_VERSIONS))
    monkeypatch.setattr(herb_ladder, "resolve_p2_tokens", _FakeResolver(members))
    monkeypatch.setattr(herb_ladder, "score_p3_detail", lambda q, k: [])
    monkeypatch.setattr(herb_ladder, "run_script_rows", lambda qs, k: [])
    monkeypatch.setattr(herb_ladder, "run_oracle_rows", lambda qs, p: [])
    monkeypatch.setattr(herb_ladder, "run_oracle_retrieval_rows", lambda qs: [])
    card, _ = herb_ladder.build()
    row = herb_ladder.p2_resolve_entry(
        herb_ladder.score_p2_resolution(members, resolver=_FakeResolver(members)), _VERSIONS, "t", "f")
    assert set(card["p2_resolve"]) == set(row)
    assert card["p2_resolve"]["nexus_revision"] == REV and card["p2_resolve"]["file"] is None


# =========================================================================== #
# RDR-147 MVV harness (nexus-bwct4.1): fixture rows and fakes only. No test here starts a sandbox,
# spawns an interpreter, calls a model or reaches a service: the nx_answer call and the plan library
# are fakes, and the two drivers run in-process against stub nexus modules.
# =========================================================================== #

REGISTRY = {"CUST-0001": "Acme", "CUST-1": "Acme", "CUST-2": "Beta", "CUST-3": "Gamma", "CUST-4": "Delta", "CUST-5": "Eps"}
UNIVERSE = sorted(set(REGISTRY.values()))


@pytest.fixture(autouse=True)
def _no_live_company_lookup(monkeypatch):
    """companies_for_ids runs `nx search` per id; a unit test answers from the fixture registry instead."""
    monkeypatch.setattr(herb_ladder, "companies_for_ids", lambda ids: {REGISTRY[i] for i in ids if i in REGISTRY})


@pytest.fixture(autouse=True)
def _mvv_switches_on(monkeypatch):
    """The MVV refuses to spend unless the resolution switch is on in the harness's own environment
    (nexus reads it from os.environ only); the refusal tests turn it off themselves."""
    monkeypatch.setenv("NX_ANSWER_RESOLUTION_REPAIR", "1")
    monkeypatch.delenv("NX_ANSWER_COVERAGE", raising=False)


@pytest.fixture(autouse=True)
def _mvv_llm_reachable(monkeypatch, tmp_path_factory):
    """The MVV also refuses unless claude is on its PATH and the token is set (mvv_llm_violations): a stub
    claude on a private PATH entry and a dummy token; the refusal tests remove them themselves."""
    d = tmp_path_factory.mktemp("claude-bin")
    stub = d / "claude"
    stub.write_text("#!/bin/sh\nexit 0\n")
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", f"{d}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "test-token")


def _qrow(i=1, *, kind="issue", named=("Acme",), gold=("Acme",), anchors=("CUST-1",), entered=True,
          valid=True, plan_id=None, evidence=(), label="a", **over):
    """One nx_answer row as run_nx_answer_rows writes it, built from fixture values. By default every
    anchor id resolved; *over* replaces any column."""
    row = {"question": f"q{i}", "kind": kind, "gold": list(gold), "pred": sorted(named),
           **herb_ladder.score_set(set(named), set(gold)),
           "run_label": label, "plan_id": plan_id, "valid": valid,
           "invalid_reason": "" if valid else "run on grown plan 14",
           "resolution_entered": entered, "resolution_reason": "", "anchor_key_ids": list(anchors),
           "resolved": list(anchors), "missing": [], "ambiguous": [], "joined_dropped": [], "tokens_dropped": [],
           "evidence_cust_ids": list(evidence)}
    return {**row, **over}


def _issue_rows(**over):
    """Five issue questions that all pass: anchors CUST-1 -> Acme, named Acme."""
    return [_qrow(i, **over) for i in range(1, 6)]


# --- the RepairReport columns, read tolerantly ---


FULL_REPORT = {
    "outcome": "completed", "resolution_entered": True, "resolution_reason": "", "resolution_tokens": 3,
    "anchor_key_ids": ["CUST-1", "CUST-2"], "resolved": ["CUST-1"], "missing": ["CUST-2"], "ambiguous": [],
    "joined_added": 1, "items_merged": 1, "items_found": 1,
    "joined_dropped": ["CUST-1@h9:no_text"], "tokens_dropped": ["CUST-9"],
    "anchors_scanned": 4, "anchors_without_pattern": 1,
    "resolved_truncated": 1, "missing_truncated": 2, "ambiguous_truncated": 3, "joined_dropped_truncated": 4,
    "tokens_dropped_truncated": 5, "anchor_key_ids_truncated": 6,
}


def test_repair_columns_copy_the_report_fields():
    cols = herb_ladder.repair_columns(FULL_REPORT)
    assert cols == {
        "resolution_entered": True, "resolution_reason": "", "resolution_tokens": 3,
        "anchor_key_ids": ["CUST-1", "CUST-2"], "resolved": ["CUST-1"], "missing": ["CUST-2"], "ambiguous": [],
        "joined_added": 1, "items_merged": 1, "repair_outcome": "completed",
        "joined_dropped": ["CUST-1@h9:no_text"], "tokens_dropped": ["CUST-9"],
        "anchors_scanned": 4, "anchors_without_pattern": 1,
        "anchor_key_ids_truncated": 6, "resolved_truncated": 1, "missing_truncated": 2, "ambiguous_truncated": 3,
        "joined_dropped_truncated": 4, "tokens_dropped_truncated": 5}


def test_repair_columns_carry_the_reason_a_considered_branch_did_not_enter():
    cols = herb_ladder.repair_columns({"resolution_entered": False, "resolution_reason": "no_key_tokens",
                                       "outcome": "no_new_evidence"})
    assert cols["resolution_reason"] == "no_key_tokens" and cols["repair_outcome"] == "no_new_evidence"
    assert cols["resolution_entered"] is False


@pytest.mark.parametrize("rep", [None, {}, {"outcome": "completed"}, "not a mapping", [], 7])
def test_repair_columns_degrade_to_not_entered(rep):
    cols = herb_ladder.repair_columns(rep)
    assert cols["resolution_entered"] is False
    assert cols["anchor_key_ids"] == [] and cols["resolved"] == [] and cols["missing"] == []
    assert cols["ambiguous"] == [] and cols["joined_dropped"] == [] and cols["tokens_dropped"] == []
    assert cols["resolution_reason"] == "" and cols["resolution_tokens"] == 0
    assert cols["joined_added"] == cols["items_merged"] == cols["anchors_scanned"] == 0
    assert cols["anchors_without_pattern"] == 0
    assert set(cols) == set(herb_ladder.REPAIR_FIELDS)


def test_repair_columns_read_an_older_nexus_report_without_the_step4_fix_fields():
    old = {"resolution_entered": True, "anchor_key_ids": ["CUST-1"], "resolved": ["CUST-1"], "joined_dropped": 2}
    cols = herb_ladder.repair_columns(old)
    assert cols["tokens_dropped"] == [] and cols["anchors_scanned"] == 0 and cols["joined_dropped"] == 2


@pytest.mark.parametrize("flag", ["yes", 1, "true", None])
def test_only_a_true_boolean_counts_as_entered(flag):
    assert herb_ladder.repair_columns({"resolution_entered": flag})["resolution_entered"] is False


def test_repair_columns_turn_null_lists_into_empty_ones():
    cols = herb_ladder.repair_columns({"anchor_key_ids": None, "missing": None, "joined_dropped": None,
                                       "tokens_dropped": None, "anchors_scanned": None})
    assert cols["anchor_key_ids"] == [] and cols["missing"] == [] and cols["joined_dropped"] == []
    assert cols["tokens_dropped"] == [] and cols["anchors_scanned"] == 0


# --- join completeness: gated, over anchor_key_ids ---


def _jc(rows, max_misses=0):
    """Strict by default (no in-domain miss allowed), so one failing question shows as a failed run and the
    question-level rules are pinned on their own; the one-miss tolerance is tested with max_misses=1 below."""
    return herb_ladder.join_completeness(rows, REGISTRY, max_misses=max_misses)


def test_completeness_passes_when_named_companies_are_a_superset():
    rows = [_qrow(1, named=("Acme", "Beta", "Gamma"), anchors=("CUST-1", "CUST-2")), *_issue_rows()[1:]]
    out = _jc(rows)
    assert out["questions"][0]["complete"] is True and out["questions"][0]["passed"] is True
    assert out["passed"] is True and out["in_domain"] == 5


def test_completeness_fails_when_one_anchor_company_is_not_named():
    rows = [_qrow(1, named=("Acme",), anchors=("CUST-1", "CUST-2")), *_issue_rows()[1:]]
    out = _jc(rows)
    q = out["questions"][0]
    assert q["complete"] is False and q["missing_companies"] == ["Beta"]
    assert out["passed"] is False


def test_a_question_with_no_anchor_ids_is_outside_the_domain():
    rows = [_qrow(1, named=(), anchors=()), *_issue_rows()[1:]]   # names nothing, would fail if in domain
    out = _jc(rows)
    assert out["questions"][0]["in_domain"] is False
    assert out["in_domain"] == 4 and out["vacuous"] is False and out["passed"] is True


def test_three_of_five_in_the_domain_fails_as_vacuous():
    rows = [_qrow(1, anchors=()), _qrow(2, anchors=()), *_issue_rows()[2:]]
    out = _jc(rows)
    assert out["in_domain"] == 3 and out["vacuous"] is True
    assert out["passed"] is False and "vacuous" in out["reason"]


def test_four_of_five_in_the_domain_is_the_floor():
    out = _jc([_qrow(1, anchors=()), *_issue_rows()[1:]])
    assert out["in_domain"] == 4 and herb_ladder.DOMAIN_FLOOR == 4 and out["passed"] is True


def test_aggregation_rows_are_not_in_the_domain():
    agg = [_qrow(6, kind="aggregation", anchors=("CUST-1",)), _qrow(7, kind="aggregation", anchors=("CUST-1",))]
    out = _jc(_issue_rows() + agg)
    assert out["in_domain"] == 5 and len(out["questions"]) == 5


def test_a_question_passes_only_where_resolution_entered():
    rows = [_qrow(1, entered=False), *_issue_rows()[1:]]
    out = _jc(rows)
    q = out["questions"][0]
    assert q["complete"] is True and q["passed"] is False and "resolution" in q["reason"]
    assert out["passed"] is False


def test_an_invalid_row_is_outside_the_domain_and_fails_the_set():
    rows = [_qrow(1, valid=False), *_issue_rows()[1:]]
    out = _jc(rows)
    assert out["questions"][0]["in_domain"] is False and out["in_domain"] == 4
    assert out["passed"] is False and out["n_invalid"] == 1   # four in the domain is not a pass: a row is missing


def test_anchor_ids_the_registry_cannot_map_do_not_pass_vacuously():
    out = _jc([_qrow(1, anchors=("CUST-999",)), *_issue_rows()[1:]])
    q = out["questions"][0]
    assert q["in_domain"] is True and q["anchor_companies"] == [] and q["passed"] is False
    assert out["passed"] is False


def test_fewer_than_five_issue_rows_fails_the_set():
    out = _jc(_issue_rows()[:4])
    assert out["n_issue"] == 4 and out["passed"] is False


def test_a_skipped_row_counts_as_not_run():
    rows = [{"question": "q1", "kind": "issue", "skipped": "cost cap reached", **herb_ladder.score_set(set(), {"Acme"})},
            *_issue_rows()[1:]]
    out = _jc(rows)
    assert out["questions"][0]["in_domain"] is False and out["passed"] is False


def test_evidence_completeness_is_reported_and_not_gated():
    rows = [_qrow(1, named=("Acme",), evidence=("CUST-1", "CUST-2")), *_issue_rows()[1:]]
    out = _jc(rows)
    ev = out["evidence_completeness"]
    assert ev["questions"][0]["missing_companies"] == ["Beta"] and ev["questions"][0]["complete"] is False
    assert ev["n_complete"] == 4 and ev["n"] == 5
    assert out["passed"] is True   # the anchors were all named: the evidence shortfall does not gate


# --- join completeness: what a question outside the domain, or a failure inside it, says about itself ---


def test_an_empty_anchor_list_beside_evidence_ids_is_named_as_boundary_not_reached():
    row = _qrow(1, anchors=(), evidence=("CUST-1", "CUST-2"), plan_id=7, resolution_reason="no_key_tokens",
                entered=False)
    out = _jc([row, *_issue_rows()[1:]])
    q = out["questions"][0]
    assert q["in_domain"] is False
    assert "boundary not reached" in q["reason"] and "branch not considered" in q["reason"]
    assert "plan_id=7" in q["reason"] and "no_key_tokens" in q["reason"]
    assert q["domain_exclusion"] == {"evidence_holds_ids": True, "plan_id": 7, "resolution_reason": "no_key_tokens",
                                     "anchors_scanned": 0, "anchors_without_pattern": 0}
    assert out["excluded_with_evidence_ids"] == 1


def test_an_empty_anchor_list_with_no_evidence_ids_is_plainly_outside_the_domain():
    out = _jc([_qrow(1, anchors=(), evidence=()), *_issue_rows()[1:]])
    q = out["questions"][0]
    assert "boundary not reached" not in q["reason"] and "outside the domain" in q["reason"]
    assert q["domain_exclusion"]["evidence_holds_ids"] is False and out["excluded_with_evidence_ids"] == 0


def test_an_in_domain_failure_carries_a_cause_for_every_anchor_id():
    row = _qrow(1, named=("Acme",), anchors=("CUST-1", "CUST-2", "CUST-3", "CUST-4", "CUST-5"),
                resolved=["CUST-1"], missing=["CUST-2"], ambiguous=["CUST-3"], tokens_dropped=["CUST-4"],
                joined_dropped=["CUST-1@h9:no_text"])
    q = _jc([row, *_issue_rows()[1:]])["questions"][0]
    causes = {i: c["cause"] for i, c in q["id_causes"].items()}
    assert causes == {"CUST-1": "resolved", "CUST-2": "missing", "CUST-3": "ambiguous", "CUST-4": "cap-dropped",
                      "CUST-5": "unaccounted"}
    assert q["id_causes"]["CUST-2"]["company"] == "Beta" and q["id_causes"]["CUST-2"]["named"] is False
    assert q["id_causes"]["CUST-1"]["named"] is True and q["id_causes"]["CUST-1"]["dropped"] == ["no_text"]
    assert q["passed"] is False


def test_a_dropped_pair_is_parsed_from_the_right_so_a_key_may_hold_an_at_sign_or_a_colon():
    key = "a@b:c"
    row = _qrow(1, anchors=(key,), resolved=[key], joined_dropped=[f"{key}@e3:no_text", "other@e4:cap", "garbled"])
    q = herb_ladder.join_completeness([row], {**REGISTRY, key: "Acme"})["questions"][0]
    assert q["id_causes"][key]["dropped"] == ["no_text"]
    assert herb_ladder.parse_dropped_pair(f"{key}@e3:no_text") == (key, "e3", "no_text")
    assert herb_ladder.parse_dropped_pair("garbled") is None and herb_ladder.parse_dropped_pair("x@y") is None


def test_a_cap_dropped_key_outranks_a_stale_resolved_entry_and_ambiguous_may_arrive_as_a_mapping():
    row = _qrow(1, named=("Acme",), anchors=("CUST-1", "CUST-3"), resolved=["CUST-1", "CUST-3"],
                tokens_dropped=["CUST-3"], ambiguous={"CUST-9": ["a", "b"]})
    q = _jc([row, *_issue_rows()[1:]])["questions"][0]
    assert q["id_causes"]["CUST-3"]["cause"] == "cap-dropped"
    row = _qrow(1, anchors=("CUST-1",), resolved=[], ambiguous={"CUST-1": ["a", "b"]})
    assert _jc([row])["questions"][0]["id_causes"]["CUST-1"]["cause"] == "ambiguous"


def test_a_resolved_only_completeness_is_reported_beside_the_gated_one_and_is_not_gated():
    # CUST-2 never resolved (missing): the gated completeness needs Beta named, the resolved-only one does not
    row = _qrow(1, named=("Acme",), anchors=("CUST-1", "CUST-2"), resolved=["CUST-1"], missing=["CUST-2"])
    out = _jc([row, *_issue_rows()[1:]])
    q = out["questions"][0]
    assert q["complete"] is False and q["resolved_only"] == {"companies": ["Acme"], "missing_companies": [],
                                                              "complete": True}
    assert out["resolved_only_completeness"] == {"n_in_domain": 5, "n_complete": 5, "gated": False}
    assert out["passed"] is False                       # the gate is the unfiltered one


def test_the_resolved_only_variant_does_not_pass_a_question_that_resolved_nothing():
    row = _qrow(1, named=("Acme",), anchors=("CUST-1",), resolved=[], missing=["CUST-1"])
    q = _jc([row, *_issue_rows()[1:]])["questions"][0]
    assert q["resolved_only"]["complete"] is False and q["resolved_only"]["companies"] == []


def test_anchor_ids_cut_on_the_wire_cannot_be_graded_and_fail_closed():
    row = _qrow(1, anchor_key_ids_truncated=3)
    q = _jc([row, *_issue_rows()[1:]])["questions"][0]
    assert q["in_domain"] is True and q["passed"] is False and "cut" in q["reason"] and "3" in q["reason"]


# --- three verdicts: passed, failed, invalid (not a measurement) ---


def test_a_run_with_an_invalid_row_is_invalid_not_failed():
    out = _jc([_qrow(1, valid=False), *_issue_rows()[1:]])
    assert out["verdict"] == "invalid" and out["passed"] is False
    pg = herb_ladder.precision_gate([_qrow(1, valid=False), *_issue_rows()[1:]])
    assert pg["verdict"] == "invalid" and pg["passed"] is False


def test_a_valid_run_whose_question_misses_a_company_is_failed():
    out = _jc([_qrow(1, named=("Acme",), anchors=("CUST-1", "CUST-2")), *_issue_rows()[1:]])
    assert out["verdict"] == "failed"
    assert _jc(_issue_rows())["verdict"] == "passed"


def test_one_in_domain_miss_per_run_passes_and_two_fail():
    """Sam 2026-11-05 (T2 nexus/rdr147-mvv-bar-revision-2026-11-05): at most one in-domain question may fail."""
    one = [_qrow(1, named=(), anchors=("CUST-1",)), *_issue_rows()[1:]]
    out = herb_ladder.join_completeness(one, REGISTRY)
    assert out["verdict"] == "passed" and out["n_missed"] == 1 and out["max_misses"] == 1
    assert "1 in-domain miss" in out["reason"]
    two = [_qrow(1, named=(), anchors=("CUST-1",)), _qrow(2, entered=False), *_issue_rows()[2:]]
    out = herb_ladder.join_completeness(two, REGISTRY)
    assert out["verdict"] == "failed" and out["n_missed"] == 2 and "at most 1 allowed" in out["reason"]


def test_the_miss_tolerance_does_not_lower_the_domain_floor():
    rows = [_qrow(1, anchors=()), _qrow(2, anchors=()), *_issue_rows()[2:]]       # 3 in the domain, none failing
    out = herb_ladder.join_completeness(rows, REGISTRY)
    assert out["vacuous"] is True and out["verdict"] == "failed"


def test_precision_leaves_out_a_question_that_named_no_company():
    """Sam 2026-11-05: precision of the named companies is undefined for a non-answer, so it is left out."""
    rows = [_qrow(1, named=()), *_issue_rows()[1:]]
    pg = herb_ladder.precision_gate(rows)
    assert pg["per_question"][0] is None and pg["n_named"] == 4 and pg["n_left_out"] == 1
    assert pg["mean"] == pytest.approx(1.0) and pg["verdict"] == "passed"


def test_precision_still_counts_a_wrong_company_and_fails_a_run_that_named_none():
    wrong = [_qrow(1, named=("Acme", "Beta"), gold=("Acme",)), *_issue_rows()[1:]]   # 0.5 on one question
    assert herb_ladder.precision_gate(wrong)["mean"] == pytest.approx((0.5 + 4) / 5)
    none = [_qrow(i, named=()) for i in range(1, 6)]
    pg = herb_ladder.precision_gate(none)
    assert pg["n_named"] == 0 and pg["verdict"] == "failed"


def test_a_vacuous_run_is_failed_not_invalid():
    out = _jc([_qrow(1, anchors=()), _qrow(2, anchors=()), *_issue_rows()[2:]])
    assert out["vacuous"] is True and out["verdict"] == "failed"


def test_too_few_issue_rows_is_invalid():
    assert _jc(_issue_rows()[:4])["verdict"] == "invalid"
    assert herb_ladder.precision_gate(_issue_rows()[:4])["verdict"] == "invalid"


def test_precision_below_the_bar_on_valid_rows_is_failed():
    out = herb_ladder.precision_gate(_prec_rows([(5, 6), (4, 5), (1, 1), (1, 1), (1, 1)]))
    assert out["verdict"] == "failed"
    assert herb_ladder.precision_gate(_prec_rows([(1, 1)] * 5))["verdict"] == "passed"


def test_mvv_verdict_is_invalid_when_a_run_is_invalid_and_the_other_passes():
    bad = [_qrow(1, valid=False, label="b"), *_issue_rows(label="b")[1:]]
    out = _mvv(_issue_rows(label="a"), bad)
    assert out["verdict"] == "invalid" and out["passed"] is False
    assert out["runs"]["b"]["verdict"] == "invalid" and out["runs"]["a"]["verdict"] == "passed"


def test_mvv_verdict_is_failed_when_one_run_fails_on_valid_rows_even_if_the_other_is_invalid():
    worse = [_qrow(1, named=("Acme", "Beta", "Gamma", "Delta"), label="a"), *_issue_rows(label="a")[1:]]
    bad = [_qrow(1, valid=False, label="b"), *_issue_rows(label="b")[1:]]
    out = _mvv(worse, bad)
    assert out["runs"]["a"]["verdict"] == "failed" and out["verdict"] == "failed"


def test_mvv_verdict_is_invalid_when_a_run_is_missing_and_passed_only_when_all_pass():
    assert _mvv(_issue_rows(label="a"))["verdict"] == "invalid"
    assert _mvv(_issue_rows(label="a"), _issue_rows(label="b"))["verdict"] == "passed"


def test_the_rungs_say_invalid_for_a_run_that_is_not_a_measurement():
    bad = [_qrow(1, valid=False, label="b"), *_issue_rows(label="b")[1:]]
    rungs = {r["metric"]: r for r in herb_ladder.mvv_rungs(_mvv(_issue_rows(label="a"), bad))}
    assert rungs["join_completeness_both_runs"]["status"] == "INVALID"
    assert rungs["precision_mean_both_runs"]["status"] == "INVALID"


# --- precision: per-run mean of per-question precision >= 0.95 in both runs ---


def _prec_rows(values, label="a"):
    """Issue rows whose per-question precision is *values* (a precision p = 1/n_pred, so give a pred set)."""
    rows = []
    for i, (hit, n_pred) in enumerate(values, 1):
        named = {f"N{j}" for j in range(n_pred)}
        gold = {f"N{j}" for j in range(hit)}
        rows.append(_qrow(i, named=named, gold=gold or {"X"}, label=label))
    return rows


def test_precision_mean_of_five_passes_at_0_9667():
    out = herb_ladder.precision_gate(_prec_rows([(5, 6), (1, 1), (1, 1), (1, 1), (1, 1)]))
    assert out["mean"] == pytest.approx(0.9666666666666667) and out["passed"] is True


def test_precision_mean_fails_at_0_9267():
    out = herb_ladder.precision_gate(_prec_rows([(5, 6), (4, 5), (1, 1), (1, 1), (1, 1)]))
    assert out["mean"] == pytest.approx(0.9266666666666667) and out["passed"] is False


def test_precision_is_the_mean_of_questions_not_the_pooled_ratio():
    # four questions of 1/1 and one of 9/20: mean 0.89 (fail); pooled 13/24 would differ
    rows = _prec_rows([(1, 1), (1, 1), (1, 1), (1, 1), (9, 20)])
    out = herb_ladder.precision_gate(rows)
    assert out["mean"] == pytest.approx((4 + 0.45) / 5) and out["passed"] is False
    assert out["pooled"] == pytest.approx(13 / 24)
    assert [round(p, 3) for p in out["per_question"]] == [1.0, 1.0, 1.0, 1.0, 0.45]


def test_precision_gate_is_exactly_at_the_threshold_inclusive():
    out = herb_ladder.precision_gate(_prec_rows([(19, 20), (1, 1), (1, 1), (1, 1), (1, 1)]))   # mean 0.99
    assert out["passed"] is True and herb_ladder.PRECISION_GATE == 0.95
    low = herb_ladder.precision_gate(_prec_rows([(1, 2), (1, 1), (1, 1), (1, 1), (4, 5)]))
    assert low["passed"] is False


def test_precision_gate_needs_five_valid_issue_rows():
    rows = _prec_rows([(1, 1)] * 5)
    rows[0]["valid"] = False
    out = herb_ladder.precision_gate(rows)
    assert out["n_valid"] == 4 and out["mean"] == 1.0 and out["passed"] is False


def test_precision_gate_ignores_aggregation_rows():
    rows = _prec_rows([(1, 1)] * 5) + [_qrow(6, kind="aggregation", named={"A", "B", "C"}, gold={"A"})]
    assert herb_ladder.precision_gate(rows)["passed"] is True


# --- the two-run MVV score ---


def _mvv(a=None, b=None, heldout=()):
    runs = {}
    if a is not None:
        runs["a"] = a
    if b is not None:
        runs["b"] = b
    return herb_ladder.score_mvv(runs, REGISTRY, list(heldout))


def test_mvv_passes_when_both_runs_pass_both_gates():
    out = _mvv(_issue_rows(label="a"), _issue_rows(label="b"))
    assert out["join_completeness_passed"] is True and out["precision_passed"] is True and out["passed"] is True
    assert out["runs"]["a"]["issue_entered"] == 5 and out["runs"]["b"]["issue_entered"] == 5


def test_mvv_fails_when_only_one_run_clears_precision():
    bad = _issue_rows(label="b")
    bad[0] = _qrow(1, named=("Acme", "Beta", "Gamma", "Delta"), gold=("Acme",), label="b")
    out = _mvv(_issue_rows(label="a"), bad)
    assert out["runs"]["a"]["precision"]["passed"] is True and out["runs"]["b"]["precision"]["passed"] is False
    assert out["precision_passed"] is False and out["passed"] is False


def test_mvv_fails_when_a_run_is_missing():
    out = _mvv(_issue_rows(label="a"))
    assert out["passed"] is False and out["join_completeness_passed"] is False and out["precision_passed"] is False


def test_mvv_fails_when_either_run_is_vacuous():
    thin = [_qrow(1, anchors=()), _qrow(2, anchors=()), *_issue_rows(label="b")[2:]]
    out = _mvv(_issue_rows(label="a"), thin)
    assert out["runs"]["b"]["join_completeness"]["vacuous"] is True
    assert out["join_completeness_passed"] is False and out["passed"] is False


def test_mvv_reports_issue5_f1_beside_the_script_value_and_agg2_unasserted():
    a = _issue_rows(label="a") + [_qrow(6, kind="aggregation", named=("Acme", "Beta"), gold=("Acme",), label="a"),
                                  _qrow(7, kind="aggregation", named=("Acme",), gold=("Acme", "Beta"), label="a")]
    out = _mvv(a, _issue_rows(label="b"))
    assert out["script_issue5_f1"] == 0.7642857142857142
    assert out["runs"]["a"]["issue5_f1"] == 1.0 and out["runs"]["a"]["issue5_recall"] == 1.0
    agg = out["runs"]["a"]["agg2"]
    assert agg["n"] == 2 and agg["precision"] == 0.75 and agg["recall"] == 0.75
    assert out["runs"]["a"]["run_label"] == "a"


def _held(n=6, *, entered=(0, 1, 4), invalid=(4,), skipped=()):
    rows = []
    for i in range(n):
        row = {"question": f"h{i}", "run_label": "heldout", "valid": i not in invalid and i not in skipped,
               "resolution_entered": i in entered, "resolution_reason": ""}
        if i in skipped:
            row["skipped"] = "cost cap reached"
        rows.append(row)
    return rows


def test_mvv_counts_heldout_runs_that_entered_the_branch_of_those_asked():
    out = _mvv(_issue_rows(label="a"), _issue_rows(label="b"), _held(6, skipped=(5,)))
    h = out["heldout"]
    assert (h["asked"], h["n_valid"], h["n_skipped"], h["n_invalid"], h["entered"]) == (6, 4, 1, 1, 2)
    assert h["interpretable"] is True and h["reasons"] == []


def test_heldout_asked_can_exceed_the_rows_that_exist():
    h = herb_ladder.score_mvv({"a": _issue_rows(label="a"), "b": _issue_rows(label="b")}, REGISTRY, _held(3),
                              n_asked=20)["heldout"]
    assert h["asked"] == 20 and h["n"] == 3


def test_a_heldout_zero_is_not_interpretable_when_the_branch_entered_no_issue_question_in_either_run():
    a = _issue_rows(label="a", entered=False)
    b = _issue_rows(label="b", entered=False)
    h = _mvv(a, b, _held(6, entered=()))["heldout"]
    assert h["entered"] == 0 and h["interpretable"] is False
    assert any("issue_entered" in r for r in h["reasons"])


def test_a_heldout_zero_stays_interpretable_when_one_run_entered_the_branch():
    a = _issue_rows(label="a", entered=False)
    h = _mvv(a, _issue_rows(label="b"), _held(6, entered=()))["heldout"]
    assert h["interpretable"] is True


def test_a_heldout_set_with_no_valid_run_is_not_interpretable():
    h = _mvv(_issue_rows(label="a"), _issue_rows(label="b"), _held(4, entered=(), invalid=(0, 1, 2, 3)))["heldout"]
    assert h["n_valid"] == 0 and h["interpretable"] is False and any("no valid" in r for r in h["reasons"])


def test_the_heldout_rung_says_entered_of_valid_of_asked_and_whether_it_reads():
    rungs = {r["metric"]: r for r in herb_ladder.mvv_rungs(
        _mvv(_issue_rows(label="a", entered=False), _issue_rows(label="b", entered=False), _held(6, entered=())))}
    r = rungs["heldout_branch_entered"]
    assert "0 entered of 5 valid of 6 asked" in r["target"] and "not interpretable" in r["target"]


# --- grown plans: tolerated shape, the sweep, the validity of a run ---


@pytest.mark.parametrize("tags,grown", [
    ("ad-hoc,grown", True), ("grown", True), (" grown , x", True),
    ("builtin-template", False), ("builtin-template,lookup", False),
    ("", True), (None, True), ("ad-hoc", True), ("builtin-template,grown", True),
])
def test_plan_is_grown_reads_tags_and_fails_closed(tags, grown):
    assert herb_ladder.plan_is_grown({"id": 3, "tags": tags}) is grown


def test_plan_is_grown_treats_a_missing_row_as_unproven():
    assert herb_ladder.plan_is_grown(None) is True


def _declared(provenance, *, tags="builtin-template", as_text=True):
    decl = {"version": 1, "kind": "lookup", "provenance": provenance}
    plan_json = {"steps": [], "answer_kind": decl}
    return {"id": 3, "tags": tags, "plan_json": json.dumps(plan_json) if as_text else plan_json}


@pytest.mark.parametrize("as_text", [True, False])
def test_a_declaration_whose_provenance_is_grown_makes_a_tagged_template_grown(as_text):
    """nexus reads a row as grown when its tags say grown OR its declaration does (answer_kind.py)."""
    assert herb_ladder.plan_is_grown(_declared("grown", as_text=as_text)) is True
    assert herb_ladder.plan_is_grown(_declared("curated", as_text=as_text)) is False


@pytest.mark.parametrize("plan_json", [None, "", "not json", "[1]", '{"answer_kind": "x"}', '{"answer_kind": {}}', 7])
def test_an_unreadable_declaration_does_not_make_a_tagged_template_grown(plan_json):
    assert herb_ladder.plan_is_grown({"id": 3, "tags": "builtin-template", "plan_json": plan_json}) is False


class FakeLibrary:
    """A fake plan library with HttpPlanLibrary's three calls the harness uses; every call is logged."""

    def __init__(self, plans, log=None, *, refuse=()):
        self.plans = {p["id"]: {"tags": "", "disabled_at": None, **p} for p in plans}
        self.log, self.refuse = [] if log is None else log, set(refuse)

    def list_plans(self, limit=20, project="", *, include_disabled=False):
        self.log.append(("list", include_disabled))
        rows = [dict(p) for p in self.plans.values() if include_disabled or not p["disabled_at"]]
        return rows[:limit]

    def get_plan(self, plan_id):
        self.log.append(("get", plan_id))
        return dict(self.plans[plan_id]) if plan_id in self.plans else None

    def set_plan_disabled(self, plan_id, *, reason=""):
        self.log.append(("disable", plan_id, reason))
        if plan_id in self.refuse or plan_id not in self.plans:
            return False
        self.plans[plan_id]["disabled_at"] = "2026-11-04T00:00:00Z"
        return True

    def grow(self, plan_id):
        self.plans[plan_id] = {"id": plan_id, "tags": "ad-hoc,grown", "disabled_at": None}


BUILTIN = {"id": 1, "tags": "builtin-template,lookup"}


def test_sweep_disables_every_grown_plan_once_and_leaves_templates():
    lib = FakeLibrary([BUILTIN, {"id": 7, "tags": "ad-hoc,grown"}, {"id": 9, "tags": "ad-hoc,grown"}])
    assert herb_ladder.sweep_grown_plans(lib).disabled == [7, 9]
    assert [e[1] for e in lib.log if e[0] == "disable"] == [7, 9]
    assert lib.plans[1]["disabled_at"] is None
    assert all(e[2] == herb_ladder.GROWN_PLAN_REASON and e[2] for e in lib.log if e[0] == "disable")
    assert herb_ladder.sweep_grown_plans(lib).disabled == []             # already disabled: not disabled again
    assert [e for e in lib.log if e[0] == "disable"] == [("disable", 7, herb_ladder.GROWN_PLAN_REASON),
                                                          ("disable", 9, herb_ladder.GROWN_PLAN_REASON)]


def test_sweep_snapshots_every_plan_id_disabled_ones_included():
    lib = FakeLibrary([BUILTIN, {"id": 7, "tags": "ad-hoc,grown", "disabled_at": "2026-11-01T00:00:00Z"},
                       {"id": 9, "tags": "ad-hoc,grown"}])
    sweep = herb_ladder.sweep_grown_plans(lib)
    assert sweep.snapshot == frozenset({1, 7, 9})
    assert sweep.disabled == [9]                                          # 7 was already disabled
    assert ("list", True) in lib.log                                      # the snapshot asked for disabled plans too


def test_sweep_records_the_id_and_tags_of_each_plan_it_disabled():
    lib = FakeLibrary([BUILTIN, {"id": 7, "tags": "ad-hoc,grown"}, {"id": 8, "tags": "user-made"}])
    sweep = herb_ladder.sweep_grown_plans(lib)
    assert sweep.records == [{"id": 7, "tags": "ad-hoc,grown"}, {"id": 8, "tags": "user-made"}]


def test_sweep_disables_a_tagged_template_whose_declaration_says_grown():
    lib = FakeLibrary([BUILTIN, {**_declared("grown"), "id": 5}])
    assert herb_ladder.sweep_grown_plans(lib).disabled == [5]


def test_sweep_refuses_a_snapshot_that_may_have_been_cut_short_even_when_the_active_list_fits(monkeypatch):
    monkeypatch.setattr(herb_ladder, "PLAN_SWEEP_LIMIT", 2)
    lib = FakeLibrary([{"id": 7, "tags": "grown"}, {"id": 8, "tags": "grown", "disabled_at": "2026-11-01T00:00:00Z"}])
    with pytest.raises(herb_ladder.NxError, match="limit"):                 # all: 2 rows (the limit); active: 1
        herb_ladder.sweep_grown_plans(lib)
    assert not [e for e in lib.log if e[0] == "disable"]


def test_sweep_refuses_when_a_plan_will_not_disable():
    lib = FakeLibrary([{"id": 7, "tags": "grown"}], refuse=[7])
    with pytest.raises(herb_ladder.NxError, match="7"):
        herb_ladder.sweep_grown_plans(lib)


def test_sweep_refuses_a_list_that_may_have_been_cut_short(monkeypatch):
    monkeypatch.setattr(herb_ladder, "PLAN_SWEEP_LIMIT", 2)
    lib = FakeLibrary([{"id": 7, "tags": "grown"}, {"id": 8, "tags": "grown"}])
    with pytest.raises(herb_ladder.NxError, match="limit"):
        herb_ladder.sweep_grown_plans(lib)
    assert not [e for e in lib.log if e[0] == "disable"]   # nothing is disabled on a list that may be partial


# Sam's rule (2026-11-04): a run answered by the plan it grew itself is VALID, plan_id 0 (nothing saved) is
# valid, and a run is invalid only when the plan it reports existed before the run, is a grown or user plan,
# and so should have been disabled by the sweep.


def test_plan_id_zero_and_none_mean_no_plan_row_and_are_valid():
    lib = FakeLibrary([BUILTIN])
    for pid in (None, 0):
        assert herb_ladder.plan_validity(lib, pid, frozenset({1})) == (True, "", False)
    assert not [e for e in lib.log if e[0] == "get"]          # nothing to read for a plan that does not exist


def test_a_plan_the_run_itself_grew_is_valid_and_marked_grown_by_the_call():
    lib = FakeLibrary([BUILTIN, {"id": 14, "tags": "ad-hoc,grown"}])
    valid, why, grown_by_call = herb_ladder.plan_validity(lib, 14, frozenset({1}))     # 14 is not in the snapshot
    assert (valid, why, grown_by_call) == (True, "", True)


def test_a_plan_that_existed_before_the_run_and_is_grown_is_invalid():
    lib = FakeLibrary([BUILTIN, {"id": 14, "tags": "ad-hoc,grown"}])
    valid, why, grown_by_call = herb_ladder.plan_validity(lib, 14, frozenset({1, 14}))
    assert valid is False and "grown plan 14" in why and grown_by_call is False


def test_a_plan_that_existed_before_the_run_and_is_a_user_plan_is_invalid():
    lib = FakeLibrary([{"id": 20, "tags": "mine"}])
    assert herb_ladder.plan_validity(lib, 20, frozenset({20}))[0] is False


def test_a_curated_template_that_existed_before_the_run_is_valid():
    lib = FakeLibrary([BUILTIN])
    assert herb_ladder.plan_validity(lib, 1, frozenset({1})) == (True, "", False)


def test_a_snapshot_plan_whose_row_cannot_be_read_is_invalid():
    ok, why, _ = herb_ladder.plan_validity(FakeLibrary([BUILTIN]), 99, frozenset({99}))
    assert ok is False and "99" in why and "not found" in why


@pytest.mark.parametrize("pid", ["abc", [], 1.5, -3])
def test_a_plan_id_that_is_not_a_plan_id_is_invalid(pid):
    assert herb_ladder.plan_validity(FakeLibrary([BUILTIN]), pid, frozenset({1}))[0] is False


class FakeAnswer:
    """Stands in for nx_answer_call: logs each run beside the library's own log, can grow a plan after
    a run (as the nx_answer grow site does) and reports a RepairReport per run.

    The envelope's plan_id is what nexus reports: with ``report_grown`` the plan this call grew on a miss
    (nexus core.py ``best.plan_id or _saved_grown_plan_id``), and 0 when it saved nothing; otherwise the
    fixed ``plan_id`` (a curated plan that matched). ``fail_on`` names calls (1-based) that raise NxError."""

    def __init__(self, lib, log, *, plan_id=1, final="Acme", grows=(), repair=None, cost=0.1,
                 report_grown=False, fail_on=()):
        self.lib, self.log, self.plan_id, self.final = lib, log, plan_id, final
        self.grows, self.repair, self.cost, self.n = list(grows), repair, cost, 0
        self.report_grown, self.fail_on = report_grown, set(fail_on)

    def __call__(self, question, scope, max_steps, timeout, budget_usd=0.5):
        self.n += 1
        self.log.append(("run", question))
        if self.n in self.fail_on:
            raise herb_ladder.NxError(f"nx_answer failed for {question!r}: exit 1: boom")
        grown_id = None
        if self.grows:
            grown_id = self.grows.pop(0)
            self.lib.grow(grown_id)
        reported = self.plan_id if not self.report_grown else (grown_id if grown_id is not None else 0)
        env = {"final_text": self.final, "cost_usd": self.cost, "plan_id": reported, "step_count": 2,
               "chunks": [], "evidence_texts": ["Customer CUST-0001: x"], "coverage_state": "covered",
               "coverage_reason": ""}
        if self.repair is not None:
            env["repair"] = self.repair
        return env


def _issue_qs(n=5):
    return [{"question": f"q{i}", "gold": ["Acme"], "citations": [], "kind": "issue"} for i in range(1, n + 1)]


def test_every_nx_answer_run_is_preceded_by_a_sweep_of_the_grown_plans(monkeypatch):
    log: list = []
    lib = FakeLibrary([BUILTIN, {"id": 7, "tags": "ad-hoc,grown"}], log)
    ans = FakeAnswer(lib, log, grows=[8, 9, 10])       # each run grows one more plan
    monkeypatch.setattr(herb_ladder, "nx_answer_call", ans)
    rows, _ = herb_ladder.run_nx_answer_rows(_issue_qs(3), UNIVERSE, 2.0, plan_library=lib, label="a")
    events = [e for e in log if e[0] in ("disable", "run")]
    assert events == [("disable", 7, herb_ladder.GROWN_PLAN_REASON), ("run", "q1"),
                      ("disable", 8, herb_ladder.GROWN_PLAN_REASON), ("run", "q2"),
                      ("disable", 9, herb_ladder.GROWN_PLAN_REASON), ("run", "q3")]
    assert [r["plans_disabled_before"] for r in rows] == [[7], [8], [9]]
    assert lib.plans[10]["disabled_at"] is None            # grown by the last run: nothing runs after it


def test_a_sweep_that_fails_stops_before_the_run(monkeypatch):
    log: list = []
    lib = FakeLibrary([{"id": 7, "tags": "grown"}], log, refuse=[7])
    ans = FakeAnswer(lib, log)
    monkeypatch.setattr(herb_ladder, "nx_answer_call", ans)
    with pytest.raises(herb_ladder.NxError):
        herb_ladder.run_nx_answer_rows(_issue_qs(1), UNIVERSE, 2.0, plan_library=lib)
    assert ans.n == 0


def test_rows_carry_the_run_label_the_plan_and_the_repair_columns(monkeypatch):
    log: list = []
    lib = FakeLibrary([BUILTIN], log)
    rep = {"resolution_entered": True, "anchor_key_ids": ["CUST-1"], "resolved": ["CUST-1"], "missing": [],
           "ambiguous": [], "joined_dropped": 0}
    monkeypatch.setattr(herb_ladder, "nx_answer_call", FakeAnswer(lib, log, repair=rep))
    rows, _ = herb_ladder.run_nx_answer_rows(_issue_qs(1), UNIVERSE, 2.0, plan_library=lib, label="b")
    r = rows[0]
    assert r["run_label"] == "b" and r["plan_id"] == 1 and r["valid"] is True and r["invalid_reason"] == ""
    assert r["resolution_entered"] is True and r["anchor_key_ids"] == ["CUST-1"] and r["resolved"] == ["CUST-1"]
    assert r["evidence_cust_ids"] == ["CUST-0001"] and r["evidence_companies"] == ["Acme"]


def test_a_missing_repair_report_reads_as_not_entered_in_the_row(monkeypatch):
    log: list = []
    lib = FakeLibrary([BUILTIN], log)
    monkeypatch.setattr(herb_ladder, "nx_answer_call", FakeAnswer(lib, log, repair=None))
    rows, _ = herb_ladder.run_nx_answer_rows(_issue_qs(1), UNIVERSE, 2.0, plan_library=lib)
    assert rows[0]["resolution_entered"] is False and rows[0]["anchor_key_ids"] == []


ANCHOR_REPORT = {"resolution_entered": True, "anchor_key_ids": ["CUST-1"], "resolved": ["CUST-1"]}


def test_a_run_answered_by_a_pre_existing_grown_plan_is_marked_invalid_and_does_not_pass(monkeypatch):
    log: list = []
    # plan 14 existed before the run and is already disabled, yet the run reports it: it must not count
    lib = FakeLibrary([BUILTIN, {"id": 14, "tags": "ad-hoc,grown", "disabled_at": "2026-11-01T00:00:00Z"}], log)
    monkeypatch.setattr(herb_ladder, "nx_answer_call", FakeAnswer(lib, log, plan_id=14, repair=ANCHOR_REPORT))
    rows, _ = herb_ladder.run_nx_answer_rows(_issue_qs(5), UNIVERSE, 2.0, plan_library=lib)
    assert all(r["valid"] is False and "grown plan 14" in r["invalid_reason"] for r in rows)
    out = herb_ladder.join_completeness(rows, REGISTRY)
    assert out["in_domain"] == 0 and out["passed"] is False and out["verdict"] == "invalid"


def test_runs_that_miss_every_template_and_answer_by_the_plan_they_grew_are_valid(monkeypatch):
    """The live shape: only q1 matches a curated plan; the rest run inline, grow a plan and report its id."""
    log: list = []
    lib = FakeLibrary([BUILTIN], log)
    monkeypatch.setattr(herb_ladder, "nx_answer_call",
                        FakeAnswer(lib, log, grows=[8, 9, 10, 11, 12], report_grown=True, repair=ANCHOR_REPORT))
    rows, _ = herb_ladder.run_nx_answer_rows(_issue_qs(5), UNIVERSE, 2.0, plan_library=lib, label="a")
    assert [r["plan_id"] for r in rows] == [8, 9, 10, 11, 12]
    assert all(r["valid"] is True and r["plan_grown_by_call"] is True and r["invalid_reason"] == "" for r in rows)
    # each next run swept the plan the previous one grew, so none of them could have matched it
    assert [r["plans_disabled_before"] for r in rows] == [[], [8], [9], [10], [11]]
    assert herb_ladder.join_completeness(rows, REGISTRY)["passed"] is True


def test_runs_that_save_no_plan_report_zero_and_are_valid(monkeypatch):
    log: list = []
    lib = FakeLibrary([BUILTIN], log)
    monkeypatch.setattr(herb_ladder, "nx_answer_call", FakeAnswer(lib, log, report_grown=True, repair=ANCHOR_REPORT))
    rows, _ = herb_ladder.run_nx_answer_rows(_issue_qs(5), UNIVERSE, 2.0, plan_library=lib)
    assert [r["plan_id"] for r in rows] == [0] * 5
    assert all(r["valid"] is True and r["plan_grown_by_call"] is False for r in rows)


def test_a_run_matching_a_curated_template_is_valid_and_not_grown_by_the_call(monkeypatch):
    log: list = []
    lib = FakeLibrary([BUILTIN], log)
    monkeypatch.setattr(herb_ladder, "nx_answer_call", FakeAnswer(lib, log, plan_id=1))
    rows, _ = herb_ladder.run_nx_answer_rows(_issue_qs(1), UNIVERSE, 2.0, plan_library=lib)
    assert rows[0]["valid"] is True and rows[0]["plan_grown_by_call"] is False


def test_rows_record_the_id_and_tags_of_every_plan_disabled_before_the_run(monkeypatch):
    log: list = []
    lib = FakeLibrary([BUILTIN, {"id": 7, "tags": "ad-hoc,grown"}], log)
    monkeypatch.setattr(herb_ladder, "nx_answer_call", FakeAnswer(lib, log))
    rows, _ = herb_ladder.run_nx_answer_rows(_issue_qs(1), UNIVERSE, 2.0, plan_library=lib)
    assert rows[0]["plans_disabled_records"] == [{"id": 7, "tags": "ad-hoc,grown"}]


def test_without_a_plan_library_no_row_claims_to_be_valid(monkeypatch):
    log: list = []
    monkeypatch.setattr(herb_ladder, "nx_answer_call", FakeAnswer(FakeLibrary([]), log))
    rows, _ = herb_ladder.run_nx_answer_rows(_issue_qs(1), UNIVERSE, 2.0)
    assert rows[0]["valid"] is False and "not checked" in rows[0]["invalid_reason"]


def test_mvv_runs_every_question_twice_labelled_a_and_b_under_one_cap(monkeypatch):
    log: list = []
    lib = FakeLibrary([BUILTIN], log)
    ans = FakeAnswer(lib, log, cost=0.4)
    monkeypatch.setattr(herb_ladder, "nx_answer_call", ans)
    rows, meta = herb_ladder.run_mvv_nx_answer(_issue_qs(5), UNIVERSE, 4.0, lib)
    assert sorted(rows) == ["a", "b"]
    assert [r["question"] for r in rows["a"]] == [r["question"] for r in rows["b"]] == [f"q{i}" for i in range(1, 6)]
    assert {r["run_label"] for r in rows["a"]} == {"a"} and {r["run_label"] for r in rows["b"]} == {"b"}
    assert ans.n == 10 and meta["total_cost_usd"] == pytest.approx(4.0) and meta["cap_usd"] == 4.0
    assert [e[1] for e in log if e[0] == "run"] == [f"q{i}" for i in range(1, 6)] * 2   # a, then b


def test_mvv_second_run_stops_at_what_the_first_left_of_the_cap(monkeypatch):
    log: list = []
    lib = FakeLibrary([BUILTIN], log)
    monkeypatch.setattr(herb_ladder, "nx_answer_call", FakeAnswer(lib, log, cost=0.6))
    rows, meta = herb_ladder.run_mvv_nx_answer(_issue_qs(3), UNIVERSE, 2.0, lib)
    ran = [[("skipped" not in r) for r in rows[l]] for l in "ab"]
    assert ran == [[True, True, True], [True, False, False]]
    assert meta["complete"] is False and meta["total_cost_usd"] == pytest.approx(2.4)
    assert all(r["valid"] is False for r in rows["b"][1:])            # a skipped row never counts


# --- the held-out single-hop set ---


def _product(types):
    return {"answerable_questions": [{"question": f"q{i}", "type": t, "ground_truth": f"g{i}",
                                      "citations": [f"c{i}"]} for i, t in enumerate(types)]}


def test_heldout_takes_non_company_questions_in_file_order_up_to_the_limit():
    types = ["company", "content", "company", "pr", "person"] + ["content"] * 30
    out = herb_ladder.held_out_questions(_product(types))
    assert len(out) == 20 and [q["herb_index"] for q in out][:4] == [1, 3, 4, 5]
    assert [q["herb_index"] for q in out] == sorted(q["herb_index"] for q in out)
    assert all(q["type"] != "company" for q in out)
    assert out[0] == {"herb_index": 1, "question": "q1", "type": "content", "ground_truth": "g1", "citations": ["c1"]}


def test_heldout_takes_all_when_there_are_fewer_than_the_limit():
    out = herb_ladder.held_out_questions(_product(["company", "pr", "person"]))
    assert [q["herb_index"] for q in out] == [1, 2]


def test_heldout_fixture_is_frozen_and_clean():
    import hashlib
    fx = json.loads((REPO / "benchmark" / "fixtures" / "herb" / "single_hop_heldout.json").read_text())
    qs = fx["questions"]
    assert len(qs) == 20 and len(qs) <= herb_ladder.HELDOUT_LIMIT and fx["product"] == "SearchFlow"
    assert all(q["type"] != "company" for q in qs)
    idx = [q["herb_index"] for q in qs]
    assert idx == sorted(set(idx)), "file order is the HERB order: indices strictly increase"
    assert len({q["question"] for q in qs}) == len(qs)
    assert all(q["question"].strip() and q["ground_truth"] and q["citations"] for q in qs)
    digest = hashlib.sha256(json.dumps(qs, sort_keys=True).encode()).hexdigest()
    assert digest == fx["sha256"], "the stored digest no longer matches the questions"
    assert digest == HELDOUT_SHA256, "the held-out set changed: it is frozen; change it only by decision"


HELDOUT_SHA256 = "4737e1ae58d0b870f7f90b37dba90ab492e636cf9c307bc8cfa6f6da8567c323"


def test_heldout_loader_returns_the_fixture_questions():
    qs = herb_ladder.load_heldout()
    assert len(qs) == 20 and qs[0]["herb_index"] == 0 and all(q["type"] != "company" for q in qs)


def test_heldout_runs_once_each_with_the_sweep_and_counts_entered(monkeypatch):
    log: list = []
    lib = FakeLibrary([BUILTIN, {"id": 7, "tags": "grown"}], log)
    reps = iter([{"resolution_entered": True}, {}, {"resolution_entered": True}])
    ans = FakeAnswer(lib, log)

    def call(*a, **k):
        env = ans(*a, **k)
        env["repair"] = next(reps)
        return env

    monkeypatch.setattr(herb_ladder, "nx_answer_call", call)
    held = [{"herb_index": i, "question": f"h{i}", "type": "content", "ground_truth": "g", "citations": ["c"]}
            for i in range(3)]
    rows, meta = herb_ladder.run_heldout_rows(held, 2.0, lib)
    assert [r["question"] for r in rows] == ["h0", "h1", "h2"] and ans.n == 3
    assert [r["resolution_entered"] for r in rows] == [True, False, True]
    assert {r["run_label"] for r in rows} == {"heldout"} and all(r["valid"] for r in rows)
    assert [e for e in log if e[0] == "disable"] == [("disable", 7, herb_ladder.GROWN_PLAN_REASON)]
    assert [r["herb_index"] for r in rows] == [0, 1, 2]
    assert meta["total_cost_usd"] == pytest.approx(0.3)


# --- the exact-assert table holds only deterministic rungs ---


def test_the_exact_assert_table_has_no_nx_answer_key():
    assert herb_ladder.PREDICTIONS and not [k for k in herb_ladder.PREDICTIONS if "nx_answer" in k]
    assert {"script_fixed.recall.issue5", "p3_hop1_precision", "p2_resolution_accuracy_29",
            "oracle_cited.recall.issue5"} <= set(herb_ladder.PREDICTIONS)
    assert herb_ladder.NX_ANSWER_PREDICTIONS and all(k.startswith("nx_answer.") for k in herb_ladder.NX_ANSWER_PREDICTIONS)


def test_comparison_marks_nx_answer_rows_as_never_asserted_exactly():
    measured = {"script_fixed.recall.issue5": 0.5, "nx_answer.recall.all7": 0.2}
    rows = {r["metric"]: r for r in herb_ladder._compare(measured)}
    assert rows["script_fixed.recall.issue5"]["asserted_exactly"] is True
    assert rows["nx_answer.recall.all7"]["asserted_exactly"] is False
    assert rows["nx_answer.recall.all7"]["measured"] == 0.2


def test_the_script_issue5_f1_is_the_recorded_baseline_exactly():
    """The one nx_answer-adjacent number asserted exactly: the SCRIPT's issue5 F1, a deterministic rung."""
    recorded = json.loads((REPO / "results" / "herb" / "step0-baselines.json").read_text())
    got = recorded["baselines"]["script_fixed"]["summary"]["issue5"]["f1"]
    assert got == herb_ladder.SCRIPT_ISSUE5_F1 == 0.7642857142857142


# --- the nx_answer driver hands the RepairReport through ---


def _run_nxa_driver(monkeypatch, capsys, result):
    async def nx_answer(**kw):
        return result

    def store_get_many(**kw):
        return {"contents": []}

    core = types.ModuleType("nexus.mcp.core")
    core.nx_answer, core.store_get_many = nx_answer, store_get_many
    for name, mod in (("nexus", types.ModuleType("nexus")), ("nexus.mcp", types.ModuleType("nexus.mcp")),
                      ("nexus.mcp.core", core)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(sys, "argv", ["-c", "q", "knowledge__x", "6", "0.5"])
    exec(compile(run_herb._NXA_DRIVER, "<nxa-driver>", "exec", dont_inherit=True), {"__name__": "__driver__"})
    line = next(l for l in capsys.readouterr().out.splitlines() if l.startswith(run_herb._NXA_SENTINEL))
    return json.loads(line[len(run_herb._NXA_SENTINEL):])


def test_driver_envelope_carries_the_coverage_repair_report(monkeypatch, capsys):
    rep = {"outcome": "completed", "resolution_entered": True, "anchor_key_ids": ["CUST-1"]}
    env = _run_nxa_driver(monkeypatch, capsys, {"final_text": "x", "chunks": [], "plan_id": 3,
                                                "coverage": {"state": "covered", "reason": "", "repair": rep}})
    assert env["repair"] == rep and env["plan_id"] == 3


def test_driver_envelope_without_a_repair_report_has_none(monkeypatch, capsys):
    env = _run_nxa_driver(monkeypatch, capsys, {"final_text": "x", "chunks": [], "plan_id": 3, "coverage": {}})
    assert env["repair"] is None
    env = _run_nxa_driver(monkeypatch, capsys, "plain string answer")
    assert env["repair"] is None


# --- the sandbox plan-library adapter and its driver ---


def _plan_env(world, monkeypatch, interp=None):
    for k in ("HERB_SANDBOX", "HOME", "NEXUS_CONFIG_DIR", "HERB_NEXUS_REV"):
        monkeypatch.delenv(k, raising=False)
    for k, v in world.env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(herb_ladder, "CONEXUS_PYTHON", interp or world.wheel_py)
    monkeypatch.setattr(herb_ladder, "_real_home", lambda: str(world.real_home))


@pytest.mark.parametrize("name", list(_BAD_ENVS))
def test_plan_library_refuses_before_any_spawn(world, monkeypatch, name):
    _plan_env(world, monkeypatch)
    for k in ("HERB_SANDBOX", "HOME", "NEXUS_CONFIG_DIR", "HERB_NEXUS_REV"):
        monkeypatch.delenv(k, raising=False)
    for k, v in _BAD_ENVS[name][0](world).items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(herb_ladder.subprocess, "run",
                        lambda *a, **k: pytest.fail("the guard let the plan library spawn a process"))
    with pytest.raises(herb_ladder.NxError, match="refus"):
        herb_ladder.SandboxPlanLibrary().list_plans()


def test_plan_library_spawns_once_per_call_and_passes_the_operation(world, monkeypatch):
    _plan_env(world, monkeypatch)
    spawned = []

    def fake_run(cmd, **kw):
        spawned.append(cmd)
        line = herb_ladder._NXA_SENTINEL + json.dumps({"result": True})
        return subprocess.CompletedProcess(cmd, 0, stdout=line.encode(), stderr=b"")

    monkeypatch.setattr(herb_ladder.subprocess, "run", fake_run)
    lib = herb_ladder.SandboxPlanLibrary()
    assert lib.set_plan_disabled(7, reason="r") is True
    assert len(spawned) == 1 and spawned[0][0] == str(world.wheel_py)
    assert spawned[0][-2] == "set_plan_disabled" and json.loads(spawned[0][-1]) == {"plan_id": 7, "reason": "r"}


def test_plan_library_failure_raises_not_empty(world, monkeypatch):
    _plan_env(world, monkeypatch)

    def boom(cmd, **kw):
        raise subprocess.CalledProcessError(1, cmd, output=b"", stderr=b"engine down")

    monkeypatch.setattr(herb_ladder.subprocess, "run", boom)
    with pytest.raises(herb_ladder.NxError, match="engine down"):
        herb_ladder.SandboxPlanLibrary().get_plan(1)
    monkeypatch.setattr(herb_ladder.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout=b"no line", stderr=b""))
    with pytest.raises(herb_ladder.NxError, match="no result"):
        herb_ladder.SandboxPlanLibrary().get_plan(1)


class _StubPlans:
    def __init__(self, monkeypatch, base_url):
        self.constructed, self.calls = [], []
        stub = self

        class Lib:
            def __init__(self):
                stub.constructed.append(1)
                self._base_url = base_url

            def set_plan_disabled(self, plan_id, *, reason=""):
                stub.calls.append(("disable", plan_id, reason))
                return True

            def list_plans(self, limit=20, project="", *, include_disabled=False):
                stub.calls.append(("list", limit, include_disabled))
                return [{"id": 1, "tags": "grown"}]

            def get_plan(self, plan_id):
                stub.calls.append(("get", plan_id))
                return None

            def delete_plan(self, plan_id):
                stub.calls.append(("delete", plan_id))

        mods = {"nexus": types.ModuleType("nexus"), "nexus.db": types.ModuleType("nexus.db"),
                "nexus.db.t2": types.ModuleType("nexus.db.t2"),
                "nexus.db.t2.http_plan_library": types.ModuleType("nexus.db.t2.http_plan_library")}
        mods["nexus.db.t2.http_plan_library"].HttpPlanLibrary = Lib
        for name, mod in mods.items():
            monkeypatch.setitem(sys.modules, name, mod)


def _run_plan_driver(monkeypatch, op, args):
    monkeypatch.setattr(sys, "argv", ["-c", op, json.dumps(args)])
    exec(compile(herb_ladder._PLAN_DRIVER, "<plan-driver>", "exec", dont_inherit=True), {"__name__": "__driver__"})


def test_plan_driver_runs_the_named_operation_on_a_loopback_service(world, monkeypatch, capsys):
    _driver_world(world, monkeypatch)
    stub = _StubPlans(monkeypatch, "http://127.0.0.1:9")
    _run_plan_driver(monkeypatch, "set_plan_disabled", {"plan_id": 7, "reason": "r"})
    line = next(l for l in capsys.readouterr().out.splitlines() if l.startswith(herb_ladder._NXA_SENTINEL))
    assert json.loads(line[len(herb_ladder._NXA_SENTINEL):]) == {"result": True, "service_url": "http://127.0.0.1:9"}
    assert stub.calls == [("disable", 7, "r")]


def test_plan_driver_refuses_a_non_loopback_service_and_an_unlisted_operation(world, monkeypatch):
    _driver_world(world, monkeypatch)
    stub = _StubPlans(monkeypatch, "https://api.conexus-nexus.com")
    with pytest.raises(SystemExit, match="non-loopback"):
        _run_plan_driver(monkeypatch, "list_plans", {})
    assert stub.calls == []
    stub = _StubPlans(monkeypatch, "http://127.0.0.1:9")
    with pytest.raises(SystemExit, match="operation"):
        _run_plan_driver(monkeypatch, "delete_plan", {"plan_id": 1})
    assert stub.calls == []


@pytest.mark.parametrize("name", list(_BAD_ENVS) + ["default interpreter"])
def test_plan_driver_reasserts_the_sandbox_before_building_a_client(world, monkeypatch, name):
    interp = world.tools / "current" / "bin" / "python3" if name == "default interpreter" else None
    _driver_world(world, monkeypatch, interpreter=interp)
    if name != "default interpreter":
        for k in ("HERB_SANDBOX", "HOME", "NEXUS_CONFIG_DIR", "HERB_NEXUS_REV"):
            monkeypatch.delenv(k, raising=False)
        for k, v in _BAD_ENVS[name][0](world).items():
            monkeypatch.setenv(k, v)
    stub = _StubPlans(monkeypatch, "http://127.0.0.1:9")
    with pytest.raises(SystemExit, match="refus"):
        _run_plan_driver(monkeypatch, "list_plans", {})
    assert stub.constructed == []


# --- the scorecard: MVV section and the copy ---


def test_the_mvv_section_goes_into_the_scorecard_with_the_run_labels(monkeypatch):
    log: list = []
    lib = FakeLibrary([BUILTIN], log)
    rep = {"resolution_entered": True, "anchor_key_ids": ["CUST-1"]}
    monkeypatch.setattr(herb_ladder, "nx_answer_call", FakeAnswer(lib, log, repair=rep, cost=0.05))
    registry = {"CUST-1": "Acme"}
    qs = [{"question": f"q{i}", "gold": ["Acme"], "citations": [], "kind": "issue"} for i in range(1, 6)]
    held = [{"herb_index": 0, "question": "h0", "type": "content", "ground_truth": "g", "citations": ["c"]}]
    mvv = herb_ladder.run_mvv(qs, ["Acme"], registry, held, 4.0, lib)
    assert mvv["score"]["passed"] is True and sorted(mvv["rows"]) == ["a", "b"]
    assert mvv["heldout_rows"][0]["run_label"] == "heldout" and mvv["score"]["heldout"]["entered"] == 1
    rungs = herb_ladder.mvv_rungs(mvv["score"])
    assert {r["phase"] for r in rungs} == {"MVV"} and all(r["status"] == "GREEN" for r in rungs
                                                          if not r["target"].startswith("report"))
    assert any(r["metric"] == "issue5_f1[a]" and r["status"] == "BASELINE" for r in rungs)


def test_copy_scorecard_copies_the_file_byte_for_byte(monkeypatch, tmp_path):
    src = tmp_path / "herb-scorecard.json"
    src.write_text('{"rdr": "RDR-147"}\n')
    monkeypatch.setattr(herb_ladder, "_SCORECARD", src)
    dest = tmp_path / "evidence" / "rdr-147"
    out = herb_ladder.copy_scorecard(dest)
    assert out == dest / "herb-scorecard.json" and out.read_bytes() == src.read_bytes()


def test_copy_scorecard_refuses_a_missing_scorecard(monkeypatch, tmp_path):
    monkeypatch.setattr(herb_ladder, "_SCORECARD", tmp_path / "absent.json")
    with pytest.raises(herb_ladder.NxError, match="absent.json"):
        herb_ladder.copy_scorecard(tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_heldout_loader_refuses_an_edited_fixture(tmp_path):
    fx = json.loads((REPO / "benchmark" / "fixtures" / "herb" / "single_hop_heldout.json").read_text())
    fx["questions"][0]["question"] += " (edited)"
    path = tmp_path / "h.json"
    path.write_text(json.dumps(fx))
    with pytest.raises(herb_ladder.NxError, match="frozen"):
        herb_ladder.load_heldout(path)
    with pytest.raises(herb_ladder.NxError, match="missing"):
        herb_ladder.load_heldout(tmp_path / "absent.json")


def test_freezing_the_heldout_set_never_overwrites_it(tmp_path):
    path = tmp_path / "h.json"
    fx = herb_ladder.freeze_heldout(_product(["company", "pr", "person"]), path)
    assert [q["herb_index"] for q in fx["questions"]] == [1, 2] and fx["n_non_company_in_source"] == 2
    assert herb_ladder.load_heldout(path) == fx["questions"]
    before = path.read_text()
    with pytest.raises(herb_ladder.NxError, match="frozen"):
        herb_ladder.freeze_heldout(_product(["pr"]), path)
    assert path.read_text() == before


def test_completeness_compares_company_names_case_insensitively():
    out = _jc([_qrow(1, named=("ACME",)), *_issue_rows()[1:]])
    assert out["questions"][0]["complete"] is True and out["passed"] is True


def test_precision_of_exactly_0_95_passes():
    out = herb_ladder.precision_gate(_prec_rows([(3, 4), (1, 1), (1, 1), (1, 1), (1, 1)]))
    assert out["mean"] == pytest.approx(0.95) and out["passed"] is True


def test_invalid_rows_are_left_out_of_the_reported_figures():
    a = _issue_rows(label="a")
    a[0] = _qrow(1, named=(), gold=("Acme",), valid=False)            # would drag issue5 recall to 0.8
    a.append(_qrow(6, kind="aggregation", named=("Acme",), gold=("Acme",), label="a"))
    a.append(_qrow(7, kind="aggregation", named=(), gold=("Acme",), label="a", valid=False))
    out = _mvv(a, _issue_rows(label="b"))["runs"]["a"]
    assert out["n_invalid"] == 2 and out["summary"]["issue5"]["n"] == 4 and out["issue5_recall"] == 1.0
    assert out["agg2"]["n"] == 1 and out["agg2"]["recall"] == 1.0


def test_the_scorecard_rungs_go_red_when_a_gate_fails():
    bad = _issue_rows(label="b")
    bad[0] = _qrow(1, named=("Acme", "Beta", "Gamma", "Delta"), label="b")        # precision 0.25 on one question
    bad[1] = _qrow(2, entered=False, label="b")                                    # a question that did not enter
    bad[2] = _qrow(3, named=(), label="b")                                         # and a non-answer: two misses
    rungs = {r["metric"]: r for r in herb_ladder.mvv_rungs(_mvv(_issue_rows(label="a"), bad))}
    assert rungs["join_completeness_both_runs"]["status"] == "RED" and rungs["join_completeness_both_runs"]["value"] == 0.0
    assert rungs["precision_mean_both_runs"]["status"] == "RED"
    assert rungs["precision_mean_both_runs"]["value"] == pytest.approx((0.25 + 1 + 1 + 1) / 4)   # the non-answer left out
    good = {r["metric"]: r for r in herb_ladder.mvv_rungs(_mvv(_issue_rows(label="a"), _issue_rows(label="b")))}
    assert good["join_completeness_both_runs"]["status"] == good["precision_mean_both_runs"]["status"] == "GREEN"


def test_the_heldout_runs_get_only_what_the_issue_runs_left_of_the_cap(monkeypatch):
    log: list = []
    lib = FakeLibrary([BUILTIN], log)
    ans = FakeAnswer(lib, log, cost=0.39)
    monkeypatch.setattr(herb_ladder, "nx_answer_call", ans)
    held = [{"herb_index": i, "question": f"h{i}", "type": "content", "ground_truth": "g", "citations": ["c"]}
            for i in range(3)]
    out = herb_ladder.run_mvv(_issue_qs(5), UNIVERSE, REGISTRY, held, 4.0, lib)
    # 10 issue runs at 0.39 = 3.9; 0.1 is left: one held-out run (0.39 -> over), the rest are skipped
    assert [("skipped" in r) for r in out["heldout_rows"]] == [False, True, True]
    assert out["meta"]["total_cost_usd"] == pytest.approx(10 * 0.39 + 0.39) and out["meta"]["complete"] is False


def test_repair_columns_copy_a_list_of_dropped_items_and_a_bad_count_reads_as_empty():
    assert herb_ladder.repair_columns({"joined_dropped": ["x", "y"]})["joined_dropped"] == ["x", "y"]
    assert herb_ladder.repair_columns({"joined_dropped": True})["joined_dropped"] == []
    assert herb_ladder.repair_columns({"joined_dropped": "3"})["joined_dropped"] == []
    assert herb_ladder.repair_columns({"ambiguous": {"CUST-1": ["h1", "h2"]}})["ambiguous"] == {"CUST-1": ["h1", "h2"]}


# --- the switch gate: nexus reads NX_ANSWER_RESOLUTION_REPAIR from os.environ only ---


@pytest.mark.parametrize("value", ["1", "true", "YES", " on "])
def test_an_on_value_of_the_resolution_switch_passes_the_gate(value):
    assert herb_ladder.mvv_switch_violations({"NX_ANSWER_RESOLUTION_REPAIR": value}) == []


@pytest.mark.parametrize("env", [{}, {"NX_ANSWER_RESOLUTION_REPAIR": ""}, {"NX_ANSWER_RESOLUTION_REPAIR": "0"},
                                 {"NX_ANSWER_RESOLUTION_REPAIR": "off"},
                                 {"NX_ANSWER_RESOLUTION_REPAIR": "banana"}])
def test_an_off_unset_or_unrecognised_resolution_switch_is_a_violation(env):
    bad = herb_ladder.mvv_switch_violations(env)
    assert bad and "NX_ANSWER_RESOLUTION_REPAIR" in bad[0]


@pytest.mark.parametrize("coverage", ["0", "off", "no", "banana"])
def test_coverage_switched_off_or_unrecognised_is_a_violation(coverage):
    bad = herb_ladder.mvv_switch_violations({"NX_ANSWER_RESOLUTION_REPAIR": "1", "NX_ANSWER_COVERAGE": coverage})
    assert bad and "NX_ANSWER_COVERAGE" in bad[0]


@pytest.mark.parametrize("coverage", [None, "", "1", "true"])
def test_coverage_unset_or_on_passes_the_gate(coverage):
    env = {"NX_ANSWER_RESOLUTION_REPAIR": "1"} | ({} if coverage is None else {"NX_ANSWER_COVERAGE": coverage})
    assert herb_ladder.mvv_switch_violations(env) == []


@pytest.mark.parametrize("env", [{"NX_ANSWER_RESOLUTION_REPAIR": ""}, {"NX_ANSWER_RESOLUTION_REPAIR": "0"},
                                 {"NX_ANSWER_RESOLUTION_REPAIR": "1", "NX_ANSWER_COVERAGE": "off"}, {}])
def test_the_mvv_refuses_before_any_spend_when_the_switches_are_not_on(monkeypatch, env):
    for k in ("NX_ANSWER_RESOLUTION_REPAIR", "NX_ANSWER_COVERAGE"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    log: list = []
    lib = FakeLibrary([BUILTIN], log)
    ans = FakeAnswer(lib, log)
    monkeypatch.setattr(herb_ladder, "nx_answer_call", ans)
    with pytest.raises(herb_ladder.NxError, match="NX_ANSWER"):
        herb_ladder.run_mvv(_issue_qs(5), UNIVERSE, REGISTRY, [], 4.0, lib)
    assert ans.n == 0 and log == [] and not herb_ladder._COST_LEDGER.exists()   # no sweep, no run, no ledger row


def test_llm_preflight_passes_with_claude_on_the_path_and_a_token(tmp_path):
    (tmp_path / "claude").write_text("#!/bin/sh\n")
    (tmp_path / "claude").chmod(0o755)
    assert herb_ladder.mvv_llm_violations({"PATH": str(tmp_path), "CLAUDE_CODE_OAUTH_TOKEN": "t"}) == []


def test_llm_preflight_names_a_missing_claude_and_a_missing_token_without_echoing_the_token(tmp_path):
    bad = herb_ladder.mvv_llm_violations({"PATH": str(tmp_path), "CLAUDE_CODE_OAUTH_TOKEN": " "})
    assert len(bad) == 2 and "claude is not on the PATH" in bad[0] and "CLAUDE_CODE_OAUTH_TOKEN" in bad[1]
    secret = "sk-very-secret"
    assert all(secret not in b for b in herb_ladder.mvv_llm_violations({"PATH": str(tmp_path),
                                                                        "CLAUDE_CODE_OAUTH_TOKEN": secret}))


def test_the_mvv_refuses_before_any_spend_when_claude_is_not_on_its_path(monkeypatch, tmp_path):
    """The 2026-11-05 first MVV run: NX_BIN in a venv without claude, so nx_answer fell back to a single-step
    template at exit 0 and every row read as a valid zero. The gate stops that before a sweep or a ledger row."""
    monkeypatch.setenv("PATH", str(tmp_path))
    log: list = []
    lib = FakeLibrary([BUILTIN], log)
    ans = FakeAnswer(lib, log)
    monkeypatch.setattr(herb_ladder, "nx_answer_call", ans)
    with pytest.raises(herb_ladder.NxError, match="claude is not on the PATH"):
        herb_ladder.run_mvv(_issue_qs(5), UNIVERSE, REGISTRY, [], 4.0, lib)
    assert ans.n == 0 and log == [] and not herb_ladder._COST_LEDGER.exists()


def test_emit_mvv_refuses_a_missing_token_before_it_reads_anything_or_touches_a_file(monkeypatch, tmp_path):
    w = _emit_world(monkeypatch, tmp_path)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN")
    monkeypatch.setattr(herb_ladder, "service_versions", lambda: pytest.fail("read the service before the gate"))
    with pytest.raises(herb_ladder.NxError, match="CLAUDE_CODE_OAUTH_TOKEN"):
        herb_ladder.emit_mvv(cap_usd=4.0, date=RUN_ID, plan_library=w.lib)
    assert w.ans.n == 0 and not (w.results / f"mvv-{RUN_ID}.jsonl").exists()


def test_emit_mvv_refuses_a_switch_that_is_off_before_it_reads_anything_or_touches_a_file(monkeypatch, tmp_path):
    w = _emit_world(monkeypatch, tmp_path)
    monkeypatch.delenv("NX_ANSWER_RESOLUTION_REPAIR")
    monkeypatch.setattr(herb_ladder, "service_versions", lambda: pytest.fail("read the service before the gate"))
    with pytest.raises(herb_ladder.NxError, match="NX_ANSWER_RESOLUTION_REPAIR"):
        herb_ladder.emit_mvv(cap_usd=4.0, date=RUN_ID, plan_library=w.lib)
    assert w.ans.n == 0 and not (w.results / f"mvv-{RUN_ID}.jsonl").exists()


# --- rows carry what the branch reported ---


def test_a_row_carries_the_branch_report_columns_a_reader_needs_to_tell_a_zero_from_a_switch_off(monkeypatch):
    log: list = []
    lib = FakeLibrary([BUILTIN], log)
    monkeypatch.setattr(herb_ladder, "nx_answer_call", FakeAnswer(lib, log, repair=FULL_REPORT))
    r = herb_ladder.run_nx_answer_rows(_issue_qs(1), UNIVERSE, 2.0, plan_library=lib)[0][0]
    for key in ("resolution_reason", "resolution_tokens", "joined_added", "items_merged", "repair_outcome",
                "tokens_dropped", "anchors_scanned", "anchors_without_pattern", "joined_dropped"):
        assert r[key] == herb_ladder.repair_columns(FULL_REPORT)[key], key
    assert r["repair_outcome"] == "completed" and r["tokens_dropped"] == ["CUST-9"]


def test_a_heldout_row_carries_the_same_columns(monkeypatch):
    log: list = []
    lib = FakeLibrary([BUILTIN], log)
    monkeypatch.setattr(herb_ladder, "nx_answer_call",
                        FakeAnswer(lib, log, repair={"resolution_entered": False, "resolution_reason": "no_key_tokens"}))
    held = [{"herb_index": 0, "question": "h0", "type": "content", "ground_truth": "g", "citations": ["c"]}]
    row = herb_ladder.run_heldout_rows(held, 2.0, lib)[0][0]
    assert row["resolution_reason"] == "no_key_tokens" and row["repair_outcome"] == ""
    assert "plans_disabled_records" in row and row["plan_grown_by_call"] is False


# --- a failed call is a row, not the end of the run ---


def _ledger_rows() -> list[dict]:
    path = herb_ladder._COST_LEDGER
    return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []


def _settled(rows: list[dict]) -> list[dict]:
    """The ledger rows a call leaves behind once it returned: its reserve rows left out."""
    return [r for r in rows if r.get("settled") is not False]


def test_an_nx_error_becomes_an_invalid_row_with_the_error_text_and_the_run_goes_on(monkeypatch):
    log: list = []
    lib = FakeLibrary([BUILTIN], log)
    ans = FakeAnswer(lib, log, fail_on=[2])
    monkeypatch.setattr(herb_ladder, "nx_answer_call", ans)
    rows, meta = herb_ladder.run_nx_answer_rows(_issue_qs(5), UNIVERSE, 2.0, plan_library=lib)
    assert ans.n == 5 and len(rows) == 5
    bad = rows[1]
    assert bad["valid"] is False and "boom" in bad["invalid_reason"] and "boom" in bad["error"]
    assert bad["resolution_entered"] is False and bad["pred"] == []
    assert all(r["valid"] for r in rows[:1] + rows[2:])
    assert meta["total_cost_usd"] == pytest.approx(4 * 0.1 + 0.5)     # the failed call is charged its budget
    ledger = _settled(_ledger_rows())
    assert [e["cost_usd"] for e in ledger] == [0.1, 0.5, 0.1, 0.1, 0.1] and ledger[1]["error"] is True


def test_a_heldout_nx_error_is_a_row_too(monkeypatch):
    log: list = []
    lib = FakeLibrary([BUILTIN], log)
    monkeypatch.setattr(herb_ladder, "nx_answer_call", FakeAnswer(lib, log, fail_on=[1]))
    held = [{"herb_index": i, "question": f"h{i}", "type": "content", "ground_truth": "g", "citations": ["c"]}
            for i in range(2)]
    rows, meta = herb_ladder.run_heldout_rows(held, 2.0, lib)
    assert [r["valid"] for r in rows] == [False, True] and "boom" in rows[0]["invalid_reason"]
    assert meta["total_cost_usd"] == pytest.approx(0.5 + 0.1)


def test_a_sweep_failure_still_stops_the_run_rather_than_becoming_a_row(monkeypatch):
    log: list = []
    lib = FakeLibrary([BUILTIN], log)
    ans = FakeAnswer(lib, log, grows=[8])
    lib.refuse = {8}
    monkeypatch.setattr(herb_ladder, "nx_answer_call", ans)
    with pytest.raises(herb_ladder.NxError, match="8"):
        herb_ladder.run_nx_answer_rows(_issue_qs(3), UNIVERSE, 2.0, plan_library=lib)
    assert ans.n == 1                                     # the second run never started on a sweep that failed


def test_an_unknown_cost_is_charged_the_per_call_budget_and_the_cap_still_holds(monkeypatch):
    log: list = []
    lib = FakeLibrary([BUILTIN], log)
    monkeypatch.setattr(herb_ladder, "nx_answer_call", FakeAnswer(lib, log, cost=None))
    rows, meta = herb_ladder.run_nx_answer_rows(_issue_qs(6), UNIVERSE, 2.0, plan_library=lib)
    assert [("skipped" in r) for r in rows] == [False] * 4 + [True] * 2
    assert meta["total_cost_usd"] == pytest.approx(2.0) and meta["runs_with_unknown_cost"] == 4
    assert rows[0]["cost_usd"] is None and rows[0]["cost_charged_usd"] == 0.5


# --- an in-band nx_answer error is an INVALID row, not a valid zero-precision measurement (fix round 2, N1) ---

# Literal notices as nexus 8096a1451 returns them in band (the call exits 0 and prints an envelope).
# core.py ``_error_result`` (:13464-13468) formats f"{NON_ANSWER_NOTICE_PREFIX} {shape}] {text}", and
# ``NON_ANSWER_NOTICE_PREFIX`` is "[non-answer:" (plans/answer_shape.py:257).
IN_BAND_ERRORS = [
    # shape "error": plan execution (core.py:14726 and :15006), plan match (:13797), a refused retrieval step's
    # exception text (:14215, :14268), an input check (:13689)
    "[non-answer: error] Error during plan execution: operator timed out",
    "[non-answer: error] Error during plan match: engine down",
    "[non-answer: error] scope refused by the retrieval step",
    "[non-answer: error] min_confidence must be in [0.0, 1.0], got 2.0",
    # shape "planner_error" (core.py:13853-13857)
    "[non-answer: planner_error] No matching plan found and inline planner failed: boom. Try rephrasing, or use "
    "search/query directly.",
    # the budget-exhausted marker: prefix core.py:12538, built at :13520-13523 (kind time or cost, :13509-13512),
    # followed by the partial answer
    "[budget exhausted (time) after step 2 of 5 — partial answer]\n\nAcme",
    "[budget exhausted (cost) after step 1 of 3 — partial answer]",
]
# Real measurements: the coverage non-answers (plans/answer_shape.py:295-312 renders them; shapes no_evidence and
# uncovered), a partial answer (``INSUFFICIENT_EVIDENCE_NOTICE_PREFIX``, answer_shape.py:250), and prose.
REAL_MEASUREMENTS = [
    "[non-answer: uncovered] The plan (plan_id=3, 4 steps) retrieved evidence, but none of it covered a requested "
    "part of the question, so no answer is given.",
    "[non-answer: no_evidence] The plan (plan_id=3, 4 steps) ran, but no usable evidence reached the reduction. "
    "Nothing was reduced.",
    "[non-answer: retrieval_only] The plan ran and returned a retrieval payload",
    "[insufficient-evidence: 2 unresolved]\n\nAcme",
    "Acme",
]


@pytest.mark.parametrize("notice", IN_BAND_ERRORS)
def test_an_in_band_error_notice_is_an_invalid_row_with_the_notice_as_its_reason(monkeypatch, notice):
    log: list = []
    lib = FakeLibrary([BUILTIN], log)
    ans = FakeAnswer(lib, log, final=notice, cost=0.1)
    monkeypatch.setattr(herb_ladder, "nx_answer_call", ans)
    rows, meta = herb_ladder.run_nx_answer_rows(_issue_qs(5), UNIVERSE, 2.0, plan_library=lib)
    assert ans.n == 5 and len(rows) == 5                            # the run goes on past it
    for r in rows:
        assert r["valid"] is False and notice.strip()[:60] in r["invalid_reason"] and r["error"].startswith("[")
        assert "grown plan" not in r["invalid_reason"]              # the notice is the reason, not the plan
    assert meta["total_cost_usd"] == pytest.approx(5 * 0.1)         # the call was paid for: it is still charged
    assert herb_ladder.join_completeness(rows, REGISTRY)["verdict"] == "invalid"
    assert herb_ladder.precision_gate(rows)["verdict"] == "invalid"


@pytest.mark.parametrize("final", REAL_MEASUREMENTS)
def test_a_coverage_non_answer_a_partial_answer_and_prose_stay_real_valid_rows(monkeypatch, final):
    log: list = []
    lib = FakeLibrary([BUILTIN], log)
    monkeypatch.setattr(herb_ladder, "nx_answer_call", FakeAnswer(lib, log, final=final, repair=ANCHOR_REPORT))
    rows, _ = herb_ladder.run_nx_answer_rows(_issue_qs(5), UNIVERSE, 2.0, plan_library=lib)
    assert all(r["valid"] is True and r["invalid_reason"] == "" and "error" not in r for r in rows)
    assert herb_ladder.precision_gate(rows)["verdict"] in ("passed", "failed")     # a measurement, not invalid


def test_in_band_error_reads_only_the_head_of_a_string():
    assert herb_ladder.in_band_error("  \n[non-answer: error] x") == "[non-answer: error] x"
    assert herb_ladder.in_band_error("The run said [non-answer: error] in passing") is None
    assert herb_ladder.in_band_error("An answer that mentions a [budget exhausted] marker") is None
    assert herb_ladder.in_band_error(None) is None and herb_ladder.in_band_error({"x": 1}) is None
    assert herb_ladder.in_band_error("[non-answer: error] " + "x" * 1000) is not None
    assert len(herb_ladder.in_band_error("[non-answer: error] " + "x" * 1000)) == herb_ladder.IN_BAND_ERROR_NOTICE_CHARS


def test_an_in_band_error_is_an_invalid_heldout_row_too(monkeypatch):
    log: list = []
    lib = FakeLibrary([BUILTIN], log)
    monkeypatch.setattr(herb_ladder, "nx_answer_call", FakeAnswer(lib, log, final=IN_BAND_ERRORS[0]))
    held = [{"herb_index": 0, "question": "h0", "type": "content", "ground_truth": "g", "citations": ["c"]}]
    rows, meta = herb_ladder.run_heldout_rows(held, 2.0, lib)
    assert rows[0]["valid"] is False and "plan execution" in rows[0]["invalid_reason"]
    assert meta["total_cost_usd"] == pytest.approx(0.1)


def test_the_non_dict_driver_path_with_an_error_notice_is_invalid(monkeypatch):
    """run_herb.py's driver maps a non-dict nx_answer return (the text form) to plan_id None and the text as
    final_text: the notice is still at its head."""
    log: list = []
    lib = FakeLibrary([BUILTIN], log)

    def text_form(question, scope, max_steps, timeout, budget_usd=0.5):
        log.append(("run", question))
        return {"final_text": IN_BAND_ERRORS[4], "chunks": [], "evidence_texts": [], "plan_id": None,
                "step_count": None, "cost_usd": None, "duration_ms": None, "repair": None}

    monkeypatch.setattr(herb_ladder, "nx_answer_call", text_form)
    rows, meta = herb_ladder.run_nx_answer_rows(_issue_qs(1), UNIVERSE, 2.0, plan_library=lib)
    assert rows[0]["valid"] is False and "planner_error" in rows[0]["invalid_reason"]
    assert rows[0]["plan_id"] is None and meta["runs_with_unknown_cost"] == 1      # unknown cost: charged the budget


def test_a_run_that_only_failed_in_band_is_invalid_not_failed(monkeypatch):
    """The cost of misreading: five budget-exhausted calls must not read as 'the branch scored zero'."""
    log: list = []
    lib = FakeLibrary([BUILTIN], log)
    monkeypatch.setattr(herb_ladder, "nx_answer_call", FakeAnswer(lib, log, final=IN_BAND_ERRORS[5]))
    held: list = []
    out = herb_ladder.run_mvv(_issue_qs(5), UNIVERSE, REGISTRY, held, 4.0, lib, run_id=RUN_ID)
    assert out["score"]["verdict"] == "invalid"


# --- the cost ledger reserves before a call and settles after it (fix round 2, N2) ---


def test_a_reserve_row_is_on_the_ledger_while_the_call_is_in_flight(monkeypatch):
    log: list = []
    lib = FakeLibrary([BUILTIN], log)
    seen = {}
    inner = FakeAnswer(lib, log, cost=0.1)

    def watching(question, scope, max_steps, timeout, budget_usd=0.5):
        seen["rows"] = _ledger_rows()
        seen["spent"] = herb_ladder.ledger_spent(RUN_ID)
        seen["budget_usd"] = budget_usd
        return inner(question, scope, max_steps, timeout, budget_usd)

    monkeypatch.setattr(herb_ladder, "nx_answer_call", watching)
    herb_ladder.run_nx_answer_rows(_issue_qs(1), UNIVERSE, 2.0, plan_library=lib, ledger_label="mvv", run_id=RUN_ID)
    (reserve,) = seen["rows"]
    assert reserve["settled"] is False and reserve["cost_usd"] == 0.5 and reserve["stage"] == "run"
    assert reserve["run"] == "mvv" and reserve["mvv_run"] == RUN_ID and reserve["reserve_id"]
    assert seen["spent"] == pytest.approx(0.5) and seen["budget_usd"] == 0.5     # charged at its per-call budget


def test_the_reserve_is_the_per_call_budget_which_is_what_is_left_under_a_small_cap(monkeypatch):
    log: list = []
    lib = FakeLibrary([BUILTIN], log)
    seen = {}
    inner = FakeAnswer(lib, log, cost=0.1)

    def watching(question, scope, max_steps, timeout, budget_usd=0.5):
        seen["spent"] = herb_ladder.ledger_spent(RUN_ID)
        return inner(question, scope, max_steps, timeout, budget_usd)

    monkeypatch.setattr(herb_ladder, "nx_answer_call", watching)
    herb_ladder.run_nx_answer_rows(_issue_qs(1), UNIVERSE, 0.3, plan_library=lib, ledger_label="mvv", run_id=RUN_ID)
    assert seen["spent"] == pytest.approx(0.3)


def test_a_returned_call_replaces_its_reserve_with_the_actual_cost_and_counts_once(monkeypatch):
    log: list = []
    lib = FakeLibrary([BUILTIN], log)
    monkeypatch.setattr(herb_ladder, "nx_answer_call", FakeAnswer(lib, log, cost=0.1))
    herb_ladder.run_nx_answer_rows(_issue_qs(2), UNIVERSE, 2.0, plan_library=lib, ledger_label="mvv", run_id=RUN_ID)
    rows = _ledger_rows()
    assert [(r["settled"], r["cost_usd"]) for r in rows] == [(False, 0.5), (True, 0.1), (False, 0.5), (True, 0.1)]
    assert rows[0]["reserve_id"] == rows[1]["reserve_id"] != rows[2]["reserve_id"] == rows[3]["reserve_id"]
    assert herb_ladder.ledger_spent(RUN_ID) == pytest.approx(0.2)         # not 0.5 + 0.1 twice, not 1.0


def test_a_failed_call_settles_at_its_per_call_budget_not_twice(monkeypatch):
    log: list = []
    lib = FakeLibrary([BUILTIN], log)
    monkeypatch.setattr(herb_ladder, "nx_answer_call", FakeAnswer(lib, log, fail_on=[1]))
    herb_ladder.run_nx_answer_rows(_issue_qs(1), UNIVERSE, 2.0, plan_library=lib, ledger_label="mvv", run_id=RUN_ID)
    assert herb_ladder.ledger_spent(RUN_ID) == pytest.approx(0.5)


@pytest.mark.parametrize("exc", [KeyboardInterrupt(), json.JSONDecodeError("Expecting value", "", 0),
                                 RuntimeError("anything")], ids=["interrupt", "decode", "other"])
def test_an_exception_mid_call_leaves_the_reserve_counted(monkeypatch, exc):
    log: list = []
    lib = FakeLibrary([BUILTIN], log)
    ok = FakeAnswer(lib, log, cost=0.1)

    def dies_on_second(question, scope, max_steps, timeout, budget_usd=0.5):
        if ok.n == 1:
            raise exc
        return ok(question, scope, max_steps, timeout, budget_usd)

    monkeypatch.setattr(herb_ladder, "nx_answer_call", dies_on_second)
    budget = herb_ladder.Budget(4.0)
    with pytest.raises(type(exc)):
        herb_ladder.run_nx_answer_rows(_issue_qs(3), UNIVERSE, 4.0, plan_library=lib, budget=budget,
                                       ledger_label="mvv", run_id=RUN_ID)
    assert ok.n == 1
    assert herb_ladder.ledger_spent(RUN_ID) == pytest.approx(0.1 + 0.5)   # the paid first call, and the lost one
    assert budget.spent == pytest.approx(0.1 + 0.5)                        # the in-process budget agrees
    # a resume starts from that and so cannot spend what the interrupted call may have cost
    assert herb_ladder.Budget(4.0, spent=herb_ladder.ledger_spent(RUN_ID)).remaining == pytest.approx(3.4)


def test_a_sweep_that_fails_before_the_call_reserves_nothing(monkeypatch):
    log: list = []
    lib = FakeLibrary([{"id": 7, "tags": "grown"}], log, refuse=[7])
    ans = FakeAnswer(lib, log)
    monkeypatch.setattr(herb_ladder, "nx_answer_call", ans)
    budget = herb_ladder.Budget(4.0)
    with pytest.raises(herb_ladder.NxError):
        herb_ladder.run_nx_answer_rows(_issue_qs(1), UNIVERSE, 4.0, plan_library=lib, budget=budget,
                                       ledger_label="mvv", run_id=RUN_ID)
    assert ans.n == 0 and _ledger_rows() == [] and budget.spent == 0.0


def test_ledger_spent_takes_the_last_row_per_reserve_and_sums_the_rest():
    def row(cost, rid=None, run=RUN_ID, label="mvv"):
        return {"run": label, "mvv_run": run, "cost_usd": cost, **({"reserve_id": rid} if rid else {})}
    herb_ladder._COST_LEDGER.write_text("\n".join(json.dumps(e) for e in [
        row(0.5, "r1"), row(0.1, "r1"),             # settled: counts 0.1
        row(0.5, "r2"),                             # never settled: counts its reserve
        row(0.3),                                   # a row without a reserve id: as it stands
        row(0.5, "r3", run="2026-11-03"),           # another day's reserve
        row(0.5, "r4", label="a"),                  # a Step 0 run's reserve
    ]) + "\n")
    assert herb_ladder.ledger_spent(RUN_ID) == pytest.approx(0.1 + 0.5 + 0.3)


def test_an_interrupted_mvv_resumes_under_a_cap_that_counts_the_lost_call(monkeypatch, tmp_path):
    lib, log, ans, sink, _ = _first_pass(monkeypatch, tmp_path, cost=0.1, cap=1.0, nq=3)   # 6 runs, 0.6 spent

    def interrupted(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(herb_ladder, "nx_answer_call", interrupted)
    with pytest.raises(KeyboardInterrupt):
        herb_ladder.run_mvv_nx_answer(_issue_qs(4), UNIVERSE, 1.0, lib, run_id=RUN_ID, sink=sink, prior=sink.load())
    assert herb_ladder.ledger_spent(RUN_ID) == pytest.approx(0.6 + 0.4)     # q4 in run a: reserve = what the cap had left


# --- rows land as they are made; a resume keeps the valid ones; the cap is cumulative ---


RUN_ID = "2026-11-04"


def _mvv_paths(tmp_path):
    return tmp_path / f"mvv-{RUN_ID}.jsonl"


def test_rows_are_appended_to_the_jsonl_as_they_land_and_survive_a_failed_sweep(monkeypatch, tmp_path):
    log: list = []
    lib = FakeLibrary([BUILTIN], log)
    ans = FakeAnswer(lib, log, grows=[8, 9, 10])
    lib.refuse = {9}                                       # the third sweep cannot disable plan 9 (grown by run 2)
    monkeypatch.setattr(herb_ladder, "nx_answer_call", ans)
    sink = herb_ladder.MvvRows(_mvv_paths(tmp_path))
    with pytest.raises(herb_ladder.NxError):
        herb_ladder.run_mvv_nx_answer(_issue_qs(5), UNIVERSE, 4.0, lib, run_id=RUN_ID, sink=sink)
    lines = [json.loads(l) for l in _mvv_paths(tmp_path).read_text().splitlines()]
    assert [r["question"] for r in lines] == ["q1", "q2"] and {r["run_label"] for r in lines} == {"a"}


def test_the_last_row_per_label_and_question_wins_on_load(tmp_path):
    sink = herb_ladder.MvvRows(_mvv_paths(tmp_path))
    sink.append({"run_label": "a", "question": "q1", "valid": False, "invalid_reason": "x"})
    sink.append({"run_label": "b", "question": "q1", "valid": True})
    sink.append({"run_label": "a", "question": "q1", "valid": True})
    prior = sink.load()
    assert set(prior) == {("a", "q1"), ("b", "q1")} and prior[("a", "q1")]["valid"] is True


def test_load_of_a_missing_file_is_empty_and_of_a_torn_last_line_keeps_the_rest(tmp_path):
    sink = herb_ladder.MvvRows(_mvv_paths(tmp_path))
    assert sink.load() == {}
    sink.append({"run_label": "a", "question": "q1", "valid": True})
    with _mvv_paths(tmp_path).open("a") as f:
        f.write('{"run_label": "a", "quest')            # a write cut short by a crash
    assert list(sink.load()) == [("a", "q1")]


def _first_pass(monkeypatch, tmp_path, *, fail_on=(), cost=0.1, cap=4.0, nq=5):
    log: list = []
    lib = FakeLibrary([BUILTIN], log)
    ans = FakeAnswer(lib, log, cost=cost, fail_on=fail_on, repair=ANCHOR_REPORT)
    monkeypatch.setattr(herb_ladder, "nx_answer_call", ans)
    sink = herb_ladder.MvvRows(_mvv_paths(tmp_path))
    out = herb_ladder.run_mvv_nx_answer(_issue_qs(nq), UNIVERSE, cap, lib, run_id=RUN_ID, sink=sink)
    return lib, log, ans, sink, out


def test_a_resume_reruns_only_the_rows_that_were_not_valid(monkeypatch, tmp_path):
    lib, log, ans, sink, (rows, _) = _first_pass(monkeypatch, tmp_path, fail_on=[2, 7])   # calls 1-5 are a, 6-10 b
    assert [r["valid"] for r in rows["a"]] == [True, False, True, True, True]
    assert [r["valid"] for r in rows["b"]] == [True, False, True, True, True]
    log.clear()
    ans2 = FakeAnswer(lib, log, cost=0.1, repair=ANCHOR_REPORT)
    monkeypatch.setattr(herb_ladder, "nx_answer_call", ans2)
    rows2, meta2 = herb_ladder.run_mvv_nx_answer(_issue_qs(5), UNIVERSE, 4.0, lib, run_id=RUN_ID, sink=sink,
                                                 prior=sink.load())
    assert ans2.n == 2 and [e[1] for e in log if e[0] == "run"] == ["q2", "q2"]
    assert all(r["valid"] for lb in "ab" for r in rows2[lb]) and [r["question"] for r in rows2["a"]] == \
        [f"q{i}" for i in range(1, 6)]
    assert meta2["total_cost_usd"] == pytest.approx(0.2)                 # this invocation's spend only
    assert herb_ladder.score_mvv(rows2, REGISTRY, [])["verdict"] == "passed"
    assert [r["valid"] for r in sink.load().values()].count(False) == 0      # the later rows replaced the bad ones


def test_a_resume_reruns_skipped_rows_and_keeps_the_ledger_spend_toward_the_cap(monkeypatch, tmp_path):
    lib, log, ans, sink, (rows, meta) = _first_pass(monkeypatch, tmp_path, cost=0.4, cap=1.0, nq=3)
    assert [("skipped" in r) for r in rows["a"] + rows["b"]] == [False, False, False, True, True, True]
    assert meta["total_cost_usd"] == pytest.approx(1.2)
    ans2 = FakeAnswer(lib, log, cost=0.4, repair=ANCHOR_REPORT)
    monkeypatch.setattr(herb_ladder, "nx_answer_call", ans2)
    # the same cap again: 1.2 is already spent, so nothing runs
    out, _ = herb_ladder.run_mvv_nx_answer(_issue_qs(3), UNIVERSE, 1.0, lib, run_id=RUN_ID, sink=sink,
                                           prior=sink.load())
    assert ans2.n == 0 and [("skipped" in r) for r in out["b"]] == [True, True, True]
    # a cap of 2.0 leaves 0.8: two more runs (cumulative 2.0), not three
    out, meta = herb_ladder.run_mvv_nx_answer(_issue_qs(3), UNIVERSE, 2.0, lib, run_id=RUN_ID, sink=sink,
                                              prior=sink.load())
    assert ans2.n == 2 and [("skipped" in r) for r in out["b"]] == [False, False, True]
    assert meta["total_cost_usd"] == pytest.approx(0.8)


def test_the_cap_reads_back_only_this_runs_ledger_rows(monkeypatch, tmp_path):
    herb_ladder._COST_LEDGER.write_text("\n".join(json.dumps(e) for e in [
        {"run": "mvv", "mvv_run": "2026-11-03", "cost_usd": 3.9},      # another day's MVV
        {"run": "a", "cost_usd": 3.9},                                 # a Step 0 run
        {"run": "mvv", "mvv_run": RUN_ID, "cost_usd": 0.25},
        {"run": "mvv", "mvv_run": RUN_ID, "cost_usd": None},
    ]) + "\nnot json\n")
    assert herb_ladder.ledger_spent(RUN_ID) == pytest.approx(0.25)
    assert herb_ladder.ledger_spent(None) == 0.0 and herb_ladder.ledger_spent("2026-01-01") == 0.0


def test_the_mvv_ledger_label_is_mvv_with_the_stage_beside_it(monkeypatch):
    log: list = []
    lib = FakeLibrary([BUILTIN], log)
    monkeypatch.setattr(herb_ladder, "nx_answer_call", FakeAnswer(lib, log, repair=ANCHOR_REPORT))
    held = [{"herb_index": 0, "question": "h0", "type": "content", "ground_truth": "g", "citations": ["c"]}]
    herb_ladder.run_mvv(_issue_qs(5), UNIVERSE, REGISTRY, held, 4.0, lib, run_id=RUN_ID)
    ledger = _settled(_ledger_rows())
    assert len(ledger) == 11 and {e["run"] for e in ledger} == {"mvv"} and {e["mvv_run"] for e in ledger} == {RUN_ID}
    assert [e["stage"] for e in ledger] == ["a"] * 5 + ["b"] * 5 + ["heldout"]
    assert herb_ladder.ledger_spent(RUN_ID) == pytest.approx(1.1)


# --- the plan driver says which service it reached ---


def test_the_plan_driver_prints_its_service_url_and_reports_it(world, monkeypatch, capsys):
    _driver_world(world, monkeypatch)
    _StubPlans(monkeypatch, "http://127.0.0.1:9")
    _run_plan_driver(monkeypatch, "list_plans", {})
    out = capsys.readouterr()
    assert "plan service url: http://127.0.0.1:9" in out.err
    line = next(l for l in out.out.splitlines() if l.startswith(herb_ladder._NXA_SENTINEL))
    assert json.loads(line[len(herb_ladder._NXA_SENTINEL):])["service_url"] == "http://127.0.0.1:9"


def test_the_plan_driver_names_a_refused_url_before_it_refuses(world, monkeypatch, capsys):
    _driver_world(world, monkeypatch)
    _StubPlans(monkeypatch, "https://api.conexus-nexus.com")
    with pytest.raises(SystemExit, match="non-loopback"):
        _run_plan_driver(monkeypatch, "list_plans", {})
    assert "plan service url: https://api.conexus-nexus.com" in capsys.readouterr().err


def test_the_plan_library_adapter_keeps_the_url_the_driver_reported(world, monkeypatch):
    _plan_env(world, monkeypatch)

    def fake_run(cmd, **kw):
        line = herb_ladder._NXA_SENTINEL + json.dumps({"result": [], "service_url": "http://127.0.0.1:9"})
        return subprocess.CompletedProcess(cmd, 0, stdout=line.encode(), stderr=b"")

    monkeypatch.setattr(herb_ladder.subprocess, "run", fake_run)
    lib = herb_ladder.SandboxPlanLibrary()
    assert lib.service_url is None
    lib.list_plans()
    assert lib.service_url == "http://127.0.0.1:9"


# --- --mvv: its own record, the scorecard patched in place, the Step 0 baselines left alone ---


def _emit_world(monkeypatch, tmp_path, *, cost=0.05, fail_on=(), report=ANCHOR_REPORT, phases=("P1", "P2", "E2E-nx_answer")):
    results = tmp_path / "herb"
    results.mkdir()
    card = _card(*phases)
    card["manifest"]["patched_rungs"] = {"P2": {"file": "p2-resolve-x.json"}}
    (results / "herb-scorecard.json").write_text(json.dumps(card))
    baselines = results / "step0-baselines.json"
    baselines.write_text('{"baselines": {"nx_answer": {"run": {"complete": true}, "summary": {"issue5": {"f1": 0.0}}}}}')
    for name, val in (("_RESULTS_DIR", results), ("_SCORECARD", results / "herb-scorecard.json"),
                      ("_BASELINES", baselines)):
        monkeypatch.setattr(herb_ladder, name, val)
    log: list = []
    lib = FakeLibrary([BUILTIN], log)
    ans = FakeAnswer(lib, log, cost=cost, repair=report, fail_on=fail_on)
    questions = [{"question": f"issue {i}", "type": "company", "ground_truth": ["Acme"], "citations": []}
                 for i in range(5)]
    questions += [{"question": f"maximum number {i}", "type": "company", "ground_truth": ["Acme"], "citations": []}
                  for i in range(2)]
    held = [{"herb_index": 0, "question": "h0", "type": "content", "ground_truth": "g", "citations": ["c"]}]
    monkeypatch.setattr(herb_ladder, "_load_gold", lambda: {
        "members": {}, "cust_to_company": REGISTRY, "mentions": {}, "q_to_custs": {}})
    monkeypatch.setattr(herb_ladder, "_load_product", lambda: {"answerable_questions": questions})
    monkeypatch.setattr(herb_ladder, "load_heldout", lambda: held)
    monkeypatch.setattr(herb_ladder, "service_versions", lambda: dict(_VERSIONS))
    monkeypatch.setenv("HERB_NEXUS_REV", REV)
    monkeypatch.setattr(herb_ladder, "nx_answer_call", ans)
    return types.SimpleNamespace(results=results, baselines=baselines, lib=lib, ans=ans, log=log)


def test_emit_mvv_writes_its_own_record_and_leaves_the_step0_baselines_alone(monkeypatch, tmp_path):
    w = _emit_world(monkeypatch, tmp_path)
    before = w.baselines.read_bytes()
    out = herb_ladder.emit_mvv(cap_usd=4.0, date=RUN_ID, plan_library=w.lib)
    assert w.ans.n == 2 * 7 + 1                        # two labelled runs of 7 questions + 1 held-out, nothing else
    assert w.baselines.read_bytes() == before          # tests/test_herb_ladder.py reads baselines.nx_answer from it
    rec = json.loads((w.results / f"mvv-{RUN_ID}.json").read_text())
    assert sorted(rec["rows"]) == ["a", "b"] and len(rec["heldout_rows"]) == 1
    assert rec["verdict"] == "passed" and rec["score"]["verdict"] == "passed" and rec["meta"]["total_cost_usd"] > 0
    assert rec["meta"]["nexus_revision"] == REV and rec["meta"]["cap_usd"] == 4.0
    assert rec["meta"]["nx_version"] == _VERSIONS["nx_version"] and rec["meta"]["run_id"] == RUN_ID
    assert len((w.results / f"mvv-{RUN_ID}.jsonl").read_text().splitlines()) == 15
    assert out["verdict"] == "passed"


def test_emit_mvv_patches_only_the_mvv_rungs_and_section_into_the_scorecard(monkeypatch, tmp_path):
    w = _emit_world(monkeypatch, tmp_path)
    before = json.loads((w.results / "herb-scorecard.json").read_text())
    herb_ladder.emit_mvv(cap_usd=4.0, date=RUN_ID, plan_library=w.lib)
    after = json.loads((w.results / "herb-scorecard.json").read_text())
    other = lambda card: [r for r in card["rungs"] if r["phase"] != "MVV"]       # noqa: E731
    assert other(after) == other(before)                          # P1, P2 and the E2E-nx_answer rungs untouched
    assert {r["metric"] for r in after["rungs"] if r["phase"] == "MVV"} >= {
        "join_completeness_both_runs", "precision_mean_both_runs", "issue5_f1[a]", "issue5_f1[b]"}
    assert after["mvv"]["verdict"] == "passed" and after["mvv"]["file"] == f"mvv-{RUN_ID}.json"
    assert "rows" not in after["mvv"]                              # the rows live in the dated file
    assert after["manifest"]["patched_rungs"]["P2"] == before["manifest"]["patched_rungs"]["P2"]
    marker = after["manifest"]["patched_rungs"]["MVV"]
    assert marker["file"] == f"mvv-{RUN_ID}.json" and marker["nexus_revision"] == REV
    assert after["manifest"]["nx_version"] == before["manifest"]["nx_version"]


def test_a_second_emit_replaces_the_mvv_rungs_instead_of_adding_more(monkeypatch, tmp_path):
    w = _emit_world(monkeypatch, tmp_path)
    herb_ladder.emit_mvv(cap_usd=4.0, date=RUN_ID, plan_library=w.lib)
    n = len([r for r in json.loads((w.results / "herb-scorecard.json").read_text())["rungs"] if r["phase"] == "MVV"])
    herb_ladder.emit_mvv(cap_usd=4.0, date=RUN_ID, plan_library=w.lib, resume=True)
    after = json.loads((w.results / "herb-scorecard.json").read_text())
    assert len([r for r in after["rungs"] if r["phase"] == "MVV"]) == n and w.ans.n == 15   # resume: nothing re-run


def test_emit_mvv_records_the_plan_service_url_when_the_library_reports_one(monkeypatch, tmp_path):
    w = _emit_world(monkeypatch, tmp_path)
    w.lib.service_url = "http://127.0.0.1:4242"
    herb_ladder.emit_mvv(cap_usd=4.0, date=RUN_ID, plan_library=w.lib)
    rec = json.loads((w.results / f"mvv-{RUN_ID}.json").read_text())
    assert rec["meta"]["plan_service_url"] == "http://127.0.0.1:4242"
    card = json.loads((w.results / "herb-scorecard.json").read_text())
    assert card["manifest"]["patched_rungs"]["MVV"]["plan_service_url"] == "http://127.0.0.1:4242"


def test_emit_mvv_with_an_invalid_row_reports_invalid_and_still_writes_everything(monkeypatch, tmp_path):
    w = _emit_world(monkeypatch, tmp_path, fail_on=[3])
    out = herb_ladder.emit_mvv(cap_usd=4.0, date=RUN_ID, plan_library=w.lib)
    assert out["verdict"] == "invalid"
    assert w.ans.n == 15
    card = json.loads((w.results / "herb-scorecard.json").read_text())
    assert {r["status"] for r in card["rungs"] if r["metric"] == "join_completeness_both_runs"} == {"INVALID"}
    assert card["mvv"]["verdict"] == "invalid"


def test_emit_mvv_refuses_an_existing_row_file_without_resume_before_any_spend(monkeypatch, tmp_path):
    w = _emit_world(monkeypatch, tmp_path)
    (w.results / f"mvv-{RUN_ID}.jsonl").write_text("")
    with pytest.raises(herb_ladder.NxError, match="resume"):
        herb_ladder.emit_mvv(cap_usd=4.0, date=RUN_ID, plan_library=w.lib)
    assert w.ans.n == 0 and w.log == []


def test_emit_mvv_resume_without_a_row_file_is_refused(monkeypatch, tmp_path):
    w = _emit_world(monkeypatch, tmp_path)
    with pytest.raises(herb_ladder.NxError, match="nothing to resume"):
        herb_ladder.emit_mvv(cap_usd=4.0, date=RUN_ID, plan_library=w.lib, resume=True)
    assert w.ans.n == 0


def test_emit_mvv_refuses_a_scorecard_it_cannot_patch_before_any_spend(monkeypatch, tmp_path):
    w = _emit_world(monkeypatch, tmp_path)
    (w.results / "herb-scorecard.json").unlink()
    with pytest.raises(herb_ladder.NxError, match="herb-scorecard.json"):
        herb_ladder.emit_mvv(cap_usd=4.0, date=RUN_ID, plan_library=w.lib)
    assert w.ans.n == 0 and not (w.results / f"mvv-{RUN_ID}.jsonl").exists()


def test_emit_mvv_resume_after_a_crash_finishes_with_cumulative_spend(monkeypatch, tmp_path):
    w = _emit_world(monkeypatch, tmp_path, cost=0.25)
    orig = w.ans.__call__

    def crash_on_sixth(*a, **k):
        if w.ans.n == 5:
            w.lib.grow(99)
            w.lib.refuse = {99}                         # the next sweep cannot disable it: the run stops
        return orig(*a, **k)

    monkeypatch.setattr(herb_ladder, "nx_answer_call", crash_on_sixth)
    with pytest.raises(herb_ladder.NxError):
        herb_ladder.emit_mvv(cap_usd=4.0, date=RUN_ID, plan_library=w.lib)
    assert len((w.results / f"mvv-{RUN_ID}.jsonl").read_text().splitlines()) == 6
    assert not (w.results / f"mvv-{RUN_ID}.json").exists()
    w.lib.refuse = set()
    monkeypatch.setattr(herb_ladder, "nx_answer_call", w.ans)
    n_before = w.ans.n
    out = herb_ladder.emit_mvv(cap_usd=4.0, date=RUN_ID, plan_library=w.lib, resume=True)
    assert w.ans.n - n_before == 15 - 6 and out["verdict"] in ("passed", "failed", "invalid")
    assert out["meta"]["cumulative_cost_usd"] == pytest.approx(15 * 0.25, abs=0.01)


def test_main_passes_resume_and_the_mvv_cap_through(monkeypatch):
    seen = {}
    monkeypatch.setattr(herb_ladder, "emit_mvv", lambda **kw: seen.update(kw))
    monkeypatch.setattr(sys, "argv", ["herb_ladder", "--mvv", "--resume", "2026-11-04"])
    herb_ladder.main()
    assert seen["resume"] is True and seen["date"] == "2026-11-04" and seen["cap_usd"] == herb_ladder.MVV_CAP_USD
    seen.clear()
    monkeypatch.setattr(sys, "argv", ["herb_ladder", "--mvv"])
    herb_ladder.main()
    assert seen["resume"] is False and seen["date"] is None


def test_main_refuses_mvv_together_with_the_step0_nx_answer_run(monkeypatch):
    monkeypatch.setattr(herb_ladder, "emit_mvv", lambda **kw: pytest.fail("ran"))
    monkeypatch.setattr(sys, "argv", ["herb_ladder", "--mvv", "--nx-answer"])
    with pytest.raises(SystemExit):
        herb_ladder.main()


def test_build_no_longer_has_an_mvv_mode_and_never_writes_the_mvv_into_the_step0_baselines():
    import inspect as _inspect
    assert "mvv" not in _inspect.signature(herb_ladder.build).parameters
    assert "mvv" not in _inspect.signature(herb_ladder.emit_scorecard).parameters


# --- docs and the wrapper: the switch is env-only and the sandbox lets it through ---


def _readme() -> str:
    """The README with its line wraps and indentation collapsed to single spaces."""
    return " ".join((REPO / "benchmark" / "README-herb.md").read_text().split())


def test_the_readme_says_the_resolution_switch_is_env_only():
    text = _readme()
    assert ("**The switch is environment-only.** nexus reads `NX_ANSWER_RESOLUTION_REPAIR` from the process "
            "environment and never from config, and it is off unless set to an on value (`1`, `true`, `yes`, `on`).") in text
    assert "sandbox's own config carries the resolution switch" not in text
    assert ("`--mvv` refuses before the first sweep, call or ledger row unless it is an on value in the harness's "
            "own environment") in text


def test_the_readme_says_what_an_invalid_verdict_is():
    assert ("`invalid` (a run is missing, or a row did not run validly or was skipped: not a measurement, rerun "
            "with `--resume`, never a finding)") in _readme()


def test_the_readme_resume_command_carries_the_whole_sandbox_wrapper():
    text = (REPO / "benchmark" / "README-herb.md").read_text()
    lines = text.splitlines()
    (i,) = [n for n, l in enumerate(lines) if "benchmark.herb_ladder --mvv --resume" in l and l.startswith("    ")]
    assert lines[i] == "    $SB run python3 -m benchmark.herb_ladder --mvv --resume [DATE]"
    assert lines[i - 1] == "    python3 <nexus>/tests/e2e/lib/claude_credentials.py run -- \\"
    assert lines[i - 2] == "NX_ANSWER_RESOLUTION_REPAIR=1 HERB_NEXUS_REV=<sha> HERB_SANDBOX_LLM=1 \\"
    # and no bare resume line is left beside it
    assert not [l for l in lines if l.startswith("python3 -m benchmark.herb_ladder --mvv --resume")]


def test_the_readme_says_the_cap_is_per_run_date_and_cap_usd_overrides_it():
    text = _readme()
    assert "A `--resume` with another DATE starts a fresh cap; a row file moved aside does not (the ledger still " \
           "holds the date's charges). `--cap-usd` overrides the cap itself." in text
    assert "or a row file moved aside, starts a fresh cap" not in text
    assert "(cap 4 USD per run date, `MVV_CAP_USD`)" in text and "cap 4 USD total" not in text


def test_the_readme_says_an_in_band_error_is_an_invalid_row():
    text = _readme()
    assert ("An in-band error is a failed call too.** nexus returns some errors with exit 0 and an envelope whose "
            "`final_text` opens `[non-answer: error]`, `[non-answer: planner_error]` or `[budget exhausted ...]`; "
            "such a row is invalid with the notice as its reason") in text
    assert "The coverage non-answers (`[non-answer: uncovered]`, `[non-answer: no_evidence]`)" in text


@pytest.mark.parametrize("bad", ["../2026-11-04", "2026-11-04/../..", "2026-1-5", "20261104", "2026-13-45",
                                 "2026-02-30", "2026-11-04 ", "latest", "x" * 40])
def test_resume_refuses_a_date_that_is_not_yyyy_mm_dd_before_anything_else(monkeypatch, tmp_path, bad):
    w = _emit_world(monkeypatch, tmp_path)
    before = sorted(p.name for p in w.results.iterdir())
    with pytest.raises(herb_ladder.NxError, match="YYYY-MM-DD"):
        herb_ladder.emit_mvv(cap_usd=4.0, date=bad, plan_library=w.lib, resume=True)
    assert w.ans.n == 0 and sorted(p.name for p in w.results.iterdir()) == before and _ledger_rows() == []


def test_the_run_date_check_accepts_a_real_date_and_returns_it():
    assert herb_ladder.valid_run_date("2026-11-04") == "2026-11-04"
    assert herb_ladder.valid_run_date("2028-02-29") == "2028-02-29"
    with pytest.raises(herb_ladder.NxError):
        herb_ladder.valid_run_date("2027-02-29")


def test_main_refuses_a_resume_date_that_is_not_a_date_before_it_calls_emit_mvv(monkeypatch, capsys):
    monkeypatch.setattr(herb_ladder, "emit_mvv", lambda **kw: pytest.fail("ran"))
    monkeypatch.setattr(sys, "argv", ["herb_ladder", "--mvv", "--resume", "../x"])
    with pytest.raises(SystemExit) as e:
        herb_ladder.main()
    assert e.value.code == 2 and "YYYY-MM-DD" in capsys.readouterr().err


def test_the_sandbox_wrapper_passes_the_resolution_switch_through_its_allowlist():
    text = (REPO / "benchmark" / "herb_sandbox.sh").read_text()
    assert 'e+=("NX_ANSWER_RESOLUTION_REPAIR=$NX_ANSWER_RESOLUTION_REPAIR")' in text
