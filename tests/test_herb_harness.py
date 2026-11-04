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
