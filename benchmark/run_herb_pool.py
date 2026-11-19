"""Full-pool HERB company-question harness for nexus ``nx_answer``.

Design of record: T2 ``nexus_rdr/herb-full-pool-company-prereg`` (binding). One pooled corpus of
every HERB product (30 files), the 130 ``type == "company"`` answerable questions in product-file
order then question order, one ``nx_answer`` attempt each, scored with HERB's own company protocol
(``code/evaluate.py``: ``answer_to_companies`` then ``f1_score_sets``).

Subcommands (run inside the sandbox, see README-herb.md and ``herb_sandbox.sh``):

    index   build the pooled collection ``knowledge__herb-pool`` (free: local bge-768 embedder)
    answer  call nx_answer once per company question; PAID, capped, resumable, ledgered
    score   extract company names from the saved answers and score them against ground truth

Never indexed, in any subcommand's corpus path: ``answerable_questions``,
``unanswerable_questions``, ``ground_truth``.

``answer`` and ``score`` spend money (nx_answer: claude -p per operator step; score: one LLM call per
answer). Neither is run by the tests; the tests inject stubs.

Attribution: ``COMPANY_PROMPT``, ``normalize`` and ``f1_score_sets`` are copied verbatim from HERB
``code/evaluate.py`` (Salesforce AI Research, github.com/SalesforceAIResearch/HERB, commit db3bf9b);
the test suite compares them to the HERB checkout when it is present.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import string
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from benchmark import run_herb as rh
from benchmark.multihop_resolve import NxError, search_hits

console = Console()
logger = logging.getLogger("herb_pool")

# --------------------------------------------------------------------------- #
# Constants (the prereg's numbers live here and nowhere else)
# --------------------------------------------------------------------------- #

POOL_NAME = "herb-pool"  # bare collection name; store_put routes it to knowledge__herb-pool
POOL_CORPUS = f"knowledge__{POOL_NAME}"
SECTIONS = ("slack", "documents", "meeting_transcripts", "meeting_chats", "urls", "prs")
# Product-JSON keys that hold questions or gold. No indexing path reads them.
GOLD_KEYS = ("answerable_questions", "unanswerable_questions", "ground_truth")
EXPECTED_COMPANY_QUESTIONS = 130
HERB_STATED_ARTIFACTS = 39190  # HERB paper figure; the gap to the files is reported, not hidden

DEFAULT_CAP_USD = 40.0  # prereg: "$40 total"
HELD_BACK_USD = 1.0  # prereg: 1 USD held back for extraction
UNKNOWN_COST_USD = 0.30  # prereg: charged when a call reports no cost
ENTITY_PATTERN = r"CUST-\d{4}"
ANSWER_TIMEOUT_S = 900
SENTINEL = "@@HERBPOOL@@"

DEFAULT_OUT_DIR = Path(
    "/private/tmp/claude-501/-Users-hal-hildebrand/9a3cd3a6-6d30-4bcc-bbc8-ebd703c9fc6f/"
    "scratchpad/herb-pool/out"
)


# --------------------------------------------------------------------------- #
# Corpus: products, questions, artifacts, entities
# --------------------------------------------------------------------------- #


def product_files(herb_dir: Path) -> list[Path]:
    """Every product file, in filename order (the prereg's "product-file order")."""
    return sorted((herb_dir / "data" / "products").glob("*.json"))


def load_products(herb_dir: Path) -> list[tuple[str, dict[str, Any]]]:
    files = product_files(herb_dir)
    if not files:
        raise SystemExit(f"no product files under {herb_dir}/data/products (clone HERB, set HERB_DIR)")
    return [(f.stem, json.loads(f.read_text())) for f in files]


def company_questions(products: Iterable[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    """The type == "company" answerable questions: product-file order, then question order.

    Each row carries ``product``, ``index`` (position in that product's answerable_questions),
    ``question`` (verbatim), ``ground_truth`` and ``citations``. The gold is for ``score`` only and is
    never sent to nexus."""
    out: list[dict[str, Any]] = []
    for name, p in products:
        for i, q in enumerate(p.get("answerable_questions", [])):
            if q.get("type") == "company":
                out.append({"product": name, "index": i, "question": q["question"],
                            "ground_truth": q["ground_truth"], "citations": q.get("citations", [])})
    return out


def _meeting_chat_text(chat: dict[str, Any]) -> str:
    return f"[meeting_chat] {chat.get('text', '')}"


def flatten_pool_product(product: dict[str, Any]) -> list[rh.ArtifactRecord]:
    """One product's artifacts as citation-id-keyed records. Reads only the SECTIONS keys.

    ``run_herb.flatten_artifacts`` covers slack (thread replies included), documents,
    meeting_transcripts, urls and prs; this adds meeting_chats, which that SearchFlow-only flattener
    never had (SearchFlow has none)."""
    clean = {k: product.get(k, []) for k in SECTIONS}  # nothing else crosses this line
    recs = rh.flatten_artifacts(clean)
    for chat in clean["meeting_chats"]:
        recs.append(rh.ArtifactRecord(chat["id"], "meeting_chats", _meeting_chat_text(chat)))
    return recs


def pool_entity_records(
    products: Iterable[tuple[str, dict[str, Any]]], herb_dir: Path
) -> tuple[list[rh.ArtifactRecord], dict[str, list[str]]]:
    """One record per customer and per employee any product references.

    Generalizes ``run_herb.load_entity_records`` (one product's ``customers``/``team``) to the union
    over all products, with the same record text, so a CUST id resolves to its company name and an
    eid to a name. Returns the records and the referenced ids that metadata does not define."""
    meta = herb_dir / "data" / "metadata"
    cust_ids: list[str] = []
    emp_ids: list[str] = []
    seen_c: set[str] = set()
    seen_e: set[str] = set()
    for _, p in products:
        for cid in p.get("customers", []):
            if cid not in seen_c:
                seen_c.add(cid)
                cust_ids.append(cid)
        for eid in p.get("team", []):
            if eid not in seen_e:
                seen_e.add(eid)
                emp_ids.append(eid)
    cust = {c["id"]: c for c in json.loads((meta / "customers_data.json").read_text())}
    emp = json.loads((meta / "employee.json").read_text())
    recs: list[rh.ArtifactRecord] = []
    missing: dict[str, list[str]] = {"customers": [], "employees": []}
    for cid in cust_ids:
        c = cust.get(cid)
        if not c:
            missing["customers"].append(cid)
            continue
        recs.append(rh.ArtifactRecord(
            cid, "customer",
            f"Customer {cid}: contact {c.get('name', '')}, {c.get('role', '')} "
            f"at company {c.get('company', '')}."))
    for eid in emp_ids:
        e = emp.get(eid)
        if not e:
            missing["employees"].append(eid)
            continue
        recs.append(rh.ArtifactRecord(
            eid, "employee",
            f"Employee {eid}: {e.get('name', '')}, {e.get('role', '')}, "
            f"{e.get('location', '')}, org {e.get('org', '')}."))
    return recs, missing


def build_pool(
    products: list[tuple[str, dict[str, Any]]], herb_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The whole index payload: [{id, title, text, tags, product, source_type}], plus a report.

    Titles are the HERB citation ids; a repeated id (HERB repeats 33 url ids, 14 of them across
    products) keeps its id and gets ``#dup<N>`` from the second use on, which
    ``artifact_id_from_title`` strips. Empty-text records are listed and not indexed."""
    records: list[tuple[str, rh.ArtifactRecord]] = []
    for name, p in products:
        records.extend((name, r) for r in flatten_pool_product(p))
    ent, missing = pool_entity_records(products, herb_dir)
    n_art = len(records)
    records.extend(("_entities", r) for r in ent)
    titled = rh.unique_titles([r for _, r in records])
    payload: list[dict[str, Any]] = []
    for (prod, r), (_, title) in zip(records, titled, strict=True):
        payload.append({"id": r.id, "title": title, "text": r.text, "product": prod,
                        "source_type": r.source_type,
                        "tags": f"herb,pool,{prod},{r.source_type}"})
    per_product: dict[str, Counter] = defaultdict(Counter)
    for row in payload:
        per_product[row["product"]][row["source_type"]] += 1
    report = {
        "artifact_records": n_art,
        "entity_records": len(ent),
        "payload_total": len(payload),
        "empty_text": [r["title"] for r in payload if not r["text"].strip()],
        "duplicate_titles": sum(1 for r in payload if "#dup" in r["title"]),
        "missing_entities": missing,
        "per_product": {k: dict(v) for k, v in per_product.items()},
    }
    return payload, report


# --------------------------------------------------------------------------- #
# Sandbox guards
# --------------------------------------------------------------------------- #


def sandbox_guard(*, allow_llm_token: bool) -> Path:
    """Refuse to run outside the herb sandbox. Returns the sandbox config dir.

    ``index`` must hold no LLM token (its post-store hooks can queue aspect extraction, an LLM call);
    ``answer`` needs one."""
    problems: list[str] = []
    if os.environ.get("HERB_SANDBOX") != "1":
        problems.append("HERB_SANDBOX is not 1 (run through benchmark/herb_sandbox.sh run ...)")
    cfg = os.environ.get("NEXUS_CONFIG_DIR", "")
    if ".herb-sandbox" not in cfg:
        problems.append(f"NEXUS_CONFIG_DIR {cfg!r} is not under a .herb-sandbox")
    # VOYAGE_API_KEY reaches a child only through the wrapper's explicit HERB_SANDBOX_VOYAGE=1 opt-in
    # (arm B); the other ambient keys are never excused.
    voyage_opt_in = os.environ.get("HERB_SANDBOX_VOYAGE") == "1"
    for k in ("NX_SERVICE_URL", "NX_SERVICE_TOKEN", "VOYAGE_API_KEY", "ANTHROPIC_API_KEY"):
        if k == "VOYAGE_API_KEY" and voyage_opt_in:
            continue
        if os.environ.get(k):
            problems.append(f"{k} is set")
    has_token = bool(os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"))
    if allow_llm_token and not has_token:
        problems.append("CLAUDE_CODE_OAUTH_TOKEN is absent (HERB_SANDBOX_LLM=1 plus the credential helper)")
    if not allow_llm_token and has_token:
        problems.append("CLAUDE_CODE_OAUTH_TOKEN is present: indexing must never be able to spend")
    if problems:
        raise SystemExit("sandbox guard: " + "; ".join(problems))
    return Path(cfg)


def entity_pattern_configured(cfg_dir: Path) -> bool:
    """True when config.yml maps ``knowledge__herb-*`` (or this collection) to a CUST pattern.

    Without it the exact-key resolution branch never runs (the 2026-11-18 SearchFlow window's first
    attempt) and the note titles are never stamped as entity keys, so it is checked BEFORE indexing."""
    cfg = cfg_dir / "config.yml"
    if not cfg.is_file():
        return False
    text = cfg.read_text()
    return re.search(r"^entity:\s*$", text, re.MULTILINE) is not None and ENTITY_PATTERN in text


def aspect_workers(cfg_dir: Path) -> list[str]:
    """Aspect-worker processes whose argv names this sandbox. nx_answer must not run beside one: the
    index queued aspect-extraction tasks and a worker with the LLM token present would spend on them."""
    ps = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True,
                        check=False).stdout
    return [line for line in ps.splitlines() if str(cfg_dir) in line and "aspect" in line.lower()]


# --------------------------------------------------------------------------- #
# index
# --------------------------------------------------------------------------- #


def _index_batch(payload: list[dict[str, Any]], tmp_dir: Path, tag: str) -> dict[str, Any]:
    """Store one batch through run_herb's in-process note-path driver; returns its result dict."""
    py = rh.CONEXUS_PYTHON
    if not py.exists():
        raise NxError(f"nx interpreter not found at {py}; set NX_PYTHON")
    arts = [{"id": r["id"], "title": r["title"], "text": r["text"], "tags": r["tags"]} for r in payload]
    f = tmp_dir / f"batch-{tag}.json"
    f.write_text(json.dumps(arts))
    err_path = tmp_dir / f"batch-{tag}.stderr"
    with err_path.open("wb") as ef:
        proc = subprocess.run([str(py), "-c", rh._BATCH_INDEX_DRIVER, str(f), POOL_NAME],
                              stdout=subprocess.PIPE, stderr=ef, timeout=6 * 3600, check=False)
    if proc.returncode != 0:
        raise NxError(f"batch {tag} exited {proc.returncode}; see {err_path}")
    for line in proc.stdout.decode().splitlines():
        if line.startswith(rh._NXA_SENTINEL):
            res = json.loads(line[len(rh._NXA_SENTINEL):])
            f.unlink(missing_ok=True)
            return res
    raise NxError(f"batch {tag} printed no result line; see {err_path}")


def _aspect_queue_counts() -> str:
    code = ("from nexus.db.t2.http_aspect_queue import HttpAspectQueue as Q; print(Q().tuple_counts())")
    try:
        p = subprocess.run([str(rh.CONEXUS_PYTHON), "-c", code], capture_output=True, text=True,
                           timeout=120, check=False)
        return (p.stdout.strip() or p.stderr.strip()[-200:]) or "<no output>"
    except Exception as e:  # noqa: BLE001  informational only
        return f"<unavailable: {e}>"


def cmd_index(args: argparse.Namespace) -> int:
    cfg_dir = sandbox_guard(allow_llm_token=False)
    if not entity_pattern_configured(cfg_dir):
        console.print(f"[red]{cfg_dir}/config.yml has no entity: pattern ({ENTITY_PATTERN}).[/red] "
                      "Run `herb_sandbox.sh up` from a tree that writes it (_ensure_entity_pattern).")
        return 2
    herb_dir = Path(args.herb_dir).expanduser()
    products = load_products(herb_dir)
    payload, rep = build_pool(products, herb_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    state_f = out / "index-state.json"
    state: dict[str, Any] = json.loads(state_f.read_text()) if state_f.exists() and not args.fresh else {}

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in payload:
        groups[row["product"]].append(row)
    order = [n for n, _ in products] + ["_entities"]
    if args.limit_products:
        order = order[: args.limit_products]
    console.print(f"pool: {rep['artifact_records']} artifact records + {rep['entity_records']} entity "
                  f"records = {rep['payload_total']} -> [cyan]{POOL_CORPUS}[/cyan]")
    t0 = time.time()
    failures: list[dict[str, Any]] = []
    for name in order:
        rows = groups.get(name, [])
        want = sum(1 for r in rows if r["text"].strip())
        if state.get(name, {}).get("indexed") == want and not state[name].get("failed"):
            console.print(f"  {name}: done ({want}), skipping")
            continue
        res = _index_batch(rows, out, name)
        state[name] = {"records": len(rows), "indexed": res["indexed"], "failed": res["failed"],
                       "at": datetime.now(timezone.utc).isoformat()}
        state_f.write_text(json.dumps(state, indent=1))
        console.print(f"  {name}: {res['indexed']}/{want} indexed, {len(res['failed'])} failed "
                      f"({time.time() - t0:.0f}s elapsed)")
        failures.extend(res["failed"])
    report = index_report(rep, state, products)
    (out / "index-report.json").write_text(json.dumps(report, indent=1))
    print_index_report(report)
    console.print(f"aspect queue after indexing (informational; queued tasks must NOT meet a token): "
                  f"{_aspect_queue_counts()}")
    if failures:
        console.print(f"[red]{len(failures)} artifacts failed to store; first: {failures[0]}[/red]")
        return 1
    return 0


def index_report(rep: dict[str, Any], state: dict[str, Any],
                 products: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    """Counts per product and total, with the explanation of the gap to HERB's stated 39,190."""
    rows = []
    for name, _ in products:
        c = rep["per_product"].get(name, {})
        rows.append({"product": name, "records": sum(c.values()), **{s: c.get(s, 0) for s in SECTIONS},
                     "indexed": state.get(name, {}).get("indexed")})
    art_total = rep["artifact_records"]
    return {
        "per_product": rows,
        "artifact_records_in_files": art_total,
        "entity_records": rep["entity_records"],
        "indexed_total": sum((v.get("indexed") or 0) for v in state.values()),
        "empty_text_not_indexed": rep["empty_text"],
        "duplicate_id_records": rep["duplicate_titles"],
        "missing_entities": rep["missing_entities"],
        "herb_stated": HERB_STATED_ARTIFACTS,
        "gap_vs_herb_stated": HERB_STATED_ARTIFACTS - art_total,
        "gap_note": (
            "The 30 product files hold exactly the records counted here. HERB's 39,190 is not "
            "reproducible from them: the shortfall is not explained by thread replies (zero in every "
            "slack message), by duplicate ids (counted, not removed) or by the entity records."),
    }


def print_index_report(rp: dict[str, Any]) -> None:
    t = Table(title="pooled artifacts per product")
    t.add_column("product")
    for c in ("records", *SECTIONS, "indexed"):
        t.add_column(c, justify="right")
    for r in rp["per_product"]:
        t.add_row(r["product"], *(str(r.get(c, "")) for c in ("records", *SECTIONS, "indexed")))
    console.print(t)
    console.print(f"artifact records in files: [bold]{rp['artifact_records_in_files']}[/bold]; "
                  f"entity records: {rp['entity_records']}; indexed total (incl. entities): "
                  f"[bold]{rp['indexed_total']}[/bold]")
    console.print(f"HERB states {rp['herb_stated']}; gap vs files: {rp['gap_vs_herb_stated']}. {rp['gap_note']}")
    if rp["empty_text_not_indexed"]:
        console.print(f"empty text, not indexed: {len(rp['empty_text_not_indexed'])}")


# --------------------------------------------------------------------------- #
# answer: cap arithmetic, ledger, resume
# --------------------------------------------------------------------------- #


def charge_for(cost: float | None) -> float:
    """A call's charge: its reported cost, or the unknown-cost charge when none is reported."""
    return float(cost) if cost is not None else UNKNOWN_COST_USD


class Ledger:
    """Append-only JSONL: a reserve row (charged = UNKNOWN_COST_USD) before each call, a settled row
    (charged = the real charge) after. ``total`` takes the LAST row per reserve id, so a call that
    never settled (a crash) still counts at its reserve, never at zero."""

    def __init__(self, path: Path):
        self.path = path

    def total(self) -> float:
        if not self.path.exists():
            return 0.0
        last: dict[str, float] = {}
        for line in self.path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                last[r["reserve_id"]] = r["charged"]
        return sum(last.values())

    def append(self, row: dict[str, Any]) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")


def would_exceed(charged: float, cap: float = DEFAULT_CAP_USD, held_back: float = HELD_BACK_USD,
                 precheck: float = UNKNOWN_COST_USD) -> bool:
    """Prereg stop rule: stop BEFORE a question when charged + 0.30 > cap - held_back."""
    return charged + precheck > cap - held_back


def done_keys(answers: Path) -> set[tuple[str, int]]:
    """(product, index) of every row already in the answers file; an error row is a done row (one attempt)."""
    if not answers.exists():
        return set()
    out = set()
    for line in answers.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out.add((r["product"], r["index"]))
    return out


_NXA_DRIVER = textwrap.dedent("""
    import asyncio, json, os, sys
    # The RDR-250 pool key scan is OFF for this run (prereg); herb_sandbox.sh's allowlist never passes it.
    os.environ.pop("NX_ANSWER_POOL_KEY_SCAN", None)
    # The index queued aspect-extraction tasks (LLM work); no worker may start while a token is present.
    os.environ["NX_ASPECT_WORKER_AUTOSTART"] = "0"
    from nexus.mcp.core import nx_answer
    if sys.argv[3] == "1":  # --no-save-plans: nexus's own no-save path (ad_hoc_ttl 0), read at call time
        import nexus.mcp.core as _core
        _core._load_ad_hoc_ttl = lambda: 0
    async def go():
        r = await nx_answer(question=sys.argv[1], scope=sys.argv[2], structured=True)
        if isinstance(r, dict):
            out = {"final_text": r.get("final_text"), "cost_usd": r.get("cost_usd"),
                   "duration_ms": r.get("duration_ms"), "plan_id": r.get("plan_id"),
                   "step_count": r.get("step_count"),
                   "coverage_state": (r.get("coverage") or {}).get("state")}
        else:
            out = {"final_text": r, "cost_usd": None, "duration_ms": None}
        print("@@HERBPOOL@@" + json.dumps(out))
    asyncio.run(go())
""").replace("@@HERBPOOL@@", SENTINEL)

AnswerFn = Callable[[str, str], dict[str, Any]]


def nx_answer_subprocess(question: str, scope: str, *, no_save_plans: bool = False) -> dict[str, Any]:
    """One nx_answer call, out-of-process in the pinned nexus interpreter. Raises NxError on failure."""
    py = rh.CONEXUS_PYTHON
    try:
        p = subprocess.run([str(py), "-c", _NXA_DRIVER, question, scope, "1" if no_save_plans else "0"],
                           capture_output=True, timeout=ANSWER_TIMEOUT_S, check=True)
    except subprocess.CalledProcessError as e:
        raise NxError(f"exit {e.returncode}: {(e.stderr or b'').decode(errors='replace')[-500:]}") from e
    except subprocess.TimeoutExpired as e:
        raise NxError(f"timeout after {ANSWER_TIMEOUT_S}s") from e
    for line in p.stdout.decode().splitlines():
        if line.startswith(SENTINEL):
            return json.loads(line[len(SENTINEL):])
    raise NxError("no envelope on stdout")


def run_answers(
    questions: list[dict[str, Any]], out_dir: Path, answer_fn: AnswerFn, *, cap: float = DEFAULT_CAP_USD,
    held_back: float = HELD_BACK_USD, limit: int | None = None, start: int = 0,
) -> dict[str, Any]:
    """The answer loop. ``questions`` is the full ordered 130; ``start`` and ``limit`` slice it (positions in
    that list), done rows are skipped, and the cap is checked from the ledger before every call.

    ``answer_fn(question, scope)`` returns the envelope dict, or raises (the row then scores 0, charged
    UNKNOWN_COST_USD, and stays in the denominator). Returns counts and the stop reason."""
    answers = out_dir / "answers.jsonl"
    ledger = Ledger(out_dir / "ledger.jsonl")
    done = done_keys(answers)
    window = questions[start:]
    if limit is not None:
        window = window[:limit]
    stats = {"attempted": 0, "skipped_done": 0, "errors": 0, "stopped": None}
    for q in window:
        key = (q["product"], q["index"])
        if key in done:
            stats["skipped_done"] += 1
            continue
        total = ledger.total()
        if would_exceed(total, cap, held_back):
            stats["stopped"] = (f"cap: charged {total:.4f} + {UNKNOWN_COST_USD} > {cap} - {held_back}; "
                                "result is PARTIAL")
            break
        rid = uuid.uuid4().hex
        ledger.append({"reserve_id": rid, "charged": UNKNOWN_COST_USD, "settled": False,
                       "product": q["product"], "index": q["index"]})
        t0 = time.time()
        env: dict[str, Any] = {}
        err: str | None = None
        try:
            env = answer_fn(q["question"], POOL_CORPUS)
        except Exception as e:  # noqa: BLE001  one attempt, an error is a recorded zero
            err = f"{type(e).__name__}: {e}"
        cost = env.get("cost_usd") if env else None
        charged = charge_for(cost)
        ledger.append({"reserve_id": rid, "charged": charged, "settled": True,
                       "cost_recorded": cost is not None, "product": q["product"], "index": q["index"]})
        row = {"product": q["product"], "index": q["index"], "question": q["question"],
               "final_text": (env or {}).get("final_text"), "cost": cost,
               "duration": (env or {}).get("duration_ms"), "wall_s": round(time.time() - t0, 1),
               "charged": charged, "error": err}
        with answers.open("a") as f:
            f.write(json.dumps(row) + "\n")
        done.add(key)
        stats["attempted"] += 1
        stats["errors"] += err is not None
        console.print(f"[{q['product']}#{q['index']}] cost={cost} charged={charged:.4f} "
                      f"total={ledger.total():.4f} err={err}")
    stats["charged_total"] = ledger.total()
    return stats


def cmd_answer(args: argparse.Namespace) -> int:
    cfg_dir = sandbox_guard(allow_llm_token=True)
    rev = os.environ.get("HERB_NEXUS_REV", "")
    if not rev:
        console.print("[red]HERB_NEXUS_REV is not set; the nexus sha must be recorded at run start.[/red]")
        return 2
    workers = aspect_workers(cfg_dir)
    if workers:
        console.print(f"[red]an aspect worker is running in this sandbox with the LLM token present: {workers}[/red]")
        return 5
    hits = search_hits("CUST-0001", 1, corpus=POOL_CORPUS)  # local and free: the index must be there
    if not hits:
        console.print(f"[red]{POOL_CORPUS} returned no hit for a probe; run `index` first.[/red]")
        return 2
    herb_dir = Path(args.herb_dir).expanduser()
    qs = company_questions(load_products(herb_dir))
    if len(qs) != EXPECTED_COMPANY_QUESTIONS:
        console.print(f"[red]expected {EXPECTED_COMPANY_QUESTIONS} company questions, found {len(qs)}[/red]")
        return 2
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.json"
    if not manifest.exists():
        manifest.write_text(json.dumps({
            "started_at": datetime.now(timezone.utc).isoformat(), "nexus_rev": rev,
            "nx_version": rh._sh([str(Path(sys.executable).parent / "nx"), "--version"]) or rh._sh(["nx", "--version"]),
            "nx_python": str(rh.CONEXUS_PYTHON), "corpus": POOL_CORPUS, "cap_usd": args.cap_usd,
            "held_back_usd": HELD_BACK_USD, "unknown_cost_usd": UNKNOWN_COST_USD,
            "no_save_plans": args.no_save_plans, "pool_key_scan": "off",
            "assetops_sha": rh._sh(["git", "rev-parse", "--short", "HEAD"]),
            "herb_sha": rh._sh(["git", "-C", str(herb_dir), "rev-parse", "--short", "HEAD"]),
        }, indent=1))
    stats = run_answers(
        qs, out, lambda q, s: nx_answer_subprocess(q, s, no_save_plans=args.no_save_plans),
        cap=args.cap_usd, limit=args.limit, start=args.start)
    console.print(stats)
    return 0 if stats["stopped"] is None else 3


# --------------------------------------------------------------------------- #
# score: HERB's company protocol (copied) + pluggable extractor
# --------------------------------------------------------------------------- #

# Verbatim from HERB code/evaluate.py answer_to_companies (db3bf9b). The doubled braces are
# str.format escapes; format() is applied exactly as HERB applies it.
COMPANY_PROMPT = (
    "The following is a question and its answer. The answer may include reasoning and a final answer. Extract only the company names that directly answer the question—ignore any names mentioned in the reasoning or intermediate steps. If no company names are part of the final answer, return {{'names': []}}."
    "\n\nQuestion: {question}"
    "\n\nAnswer: {content}"
    "\n\nRespond in JSON format: {{'names': ['list of company names']}}"
)


def company_prompt(question: str, content: str) -> str:
    return COMPANY_PROMPT.format(question=question, content=content)


def normalize(answer: str) -> str:
    """Verbatim from HERB evaluate.py (SQuAD-style normalization)."""
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(answer))))


def f1_score_sets(y_true: Iterable[str], y_pred: Iterable[str]) -> float:
    """Verbatim from HERB evaluate.py."""
    y_true_set = set([normalize(e) for e in y_true])  # noqa: C403  verbatim from HERB
    y_pred_set = set([normalize(e) for e in y_pred])  # noqa: C403

    tp = len(y_true_set & y_pred_set)
    fp = len(y_pred_set - y_true_set)
    fn = len(y_true_set - y_pred_set)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return f1


def parse_names(raw: str | None) -> list[str]:
    """Names from an extractor reply: a JSON (or Python-literal) object with a 'names' list.

    Raises ValueError when the reply has no such object. HERB's own code crashes there (its
    get_gpt4_response returns None and ``None['names']`` raises); ``score`` records the failure and
    scores that row 0 rather than dropping it."""
    import ast

    if not raw:
        raise ValueError("empty extractor reply")
    text = raw.strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON object in {text[:80]!r}")
    blob = m.group(0)
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        try:
            obj = ast.literal_eval(blob)
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"unparseable extractor reply {blob[:80]!r}") from e
    names = obj.get("names") if isinstance(obj, dict) else None
    if not isinstance(names, list):
        raise ValueError(f"no 'names' list in {blob[:80]!r}")  # noqa: TRY004  one error type for callers
    return [str(n) for n in names]


Extractor = Callable[[str, str], list[str]]


def openai_extractor(model: str = "gpt-4o", temperature: float = 0.00001) -> Extractor:
    """HERB's protocol extractor: gpt-4o at HERB's temperature, JSON mode. Needs OPENAI_API_KEY."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("--extractor openai needs OPENAI_API_KEY in the environment (none is set; "
                         "nothing was called). Use --extractor sonnet for the labelled fallback.")

    def extract(question: str, content: str) -> list[str]:
        body = json.dumps({"model": model, "temperature": temperature,
                           "response_format": {"type": "json_object"},
                           "messages": [{"role": "user", "content": company_prompt(question, content)}]}).encode()
        req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=body,
                                     headers={"Authorization": f"Bearer {key}",
                                              "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                reply = json.loads(r.read())["choices"][0]["message"]["content"]
        except (urllib.error.URLError, KeyError, json.JSONDecodeError) as e:
            raise RuntimeError(f"openai call failed: {e}") from e
        return parse_names(reply)

    return extract


def sonnet_extractor(model: str = "sonnet", claude_bin: str | None = None) -> Extractor:
    """Fallback extractor: the same prompt, verbatim, through ``claude -p``. Results made with it are
    labelled "Sonnet-extracted, not protocol-exact"."""
    exe = claude_bin or shutil.which("claude")
    if not exe:
        raise SystemExit("--extractor sonnet needs the claude CLI on PATH (or --claude-bin); none found")

    def extract(question: str, content: str) -> list[str]:
        try:
            p = subprocess.run([exe, "-p", "--model", model], input=company_prompt(question, content),
                               capture_output=True, text=True, timeout=300, check=True)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            raise RuntimeError(f"claude extractor failed: {e}") from e
        return parse_names(p.stdout)

    return extract


EXTRACTOR_LABEL = {"openai": "gpt-4o-extracted (HERB protocol)",
                   "sonnet": "Sonnet-extracted, not protocol-exact"}


def load_answers(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[(r["product"], r["index"])] = r
    return rows


def score_answers(
    questions: list[dict[str, Any]], answers: dict[tuple[str, int], dict[str, Any]], extract: Extractor,
    cache: dict[str, list[str]] | None = None, regex_universe: list[str] | None = None,
) -> dict[str, Any]:
    """Score every one of the questions. A missing row, an answer error, an empty answer or an
    extraction failure scores 0 and stays in the denominator. ``cache`` maps "product#index" to the
    extracted names so a re-run does not re-call the extractor."""
    cache = cache if cache is not None else {}
    per_q: list[dict[str, Any]] = []
    for q in questions:
        key = f"{q['product']}#{q['index']}"
        row = answers.get((q["product"], q["index"]))
        rec: dict[str, Any] = {"product": q["product"], "index": q["index"], "key": key,
                               "ground_truth": q["ground_truth"], "pred": None, "f1": 0.0,
                               "status": "ok", "regex_f1": None}
        text = (row or {}).get("final_text")
        if row is None:
            rec["status"] = "no_answer_row"
        elif row.get("error"):
            rec["status"] = "answer_error"
        elif not (text or "").strip():
            rec["status"] = "empty_answer"
        else:
            try:
                names = cache[key] if key in cache else extract(q["question"], text)
                cache[key] = names
                rec["pred"] = names
                rec["f1"] = f1_score_sets(q["ground_truth"], names)
            except Exception as e:  # noqa: BLE001
                rec["status"] = f"extract_error: {e}"
        if regex_universe is not None and text:
            rec["regex_f1"] = regex_f1(q["ground_truth"], text, regex_universe)
        per_q.append(rec)
    by_prod: dict[str, list[float]] = defaultdict(list)
    for r in per_q:
        by_prod[r["product"]].append(r["f1"])
    out = {
        "n": len(per_q),
        "overall_f1_x100": 100 * sum(r["f1"] for r in per_q) / len(per_q) if per_q else 0.0,
        "per_product_f1_x100": {p: 100 * sum(v) / len(v) for p, v in by_prod.items()},
        "per_product_n": {p: len(v) for p, v in by_prod.items()},
        "status_counts": dict(Counter(r["status"].split(":")[0] for r in per_q)),
        "per_question": per_q,
    }
    if regex_universe is not None:
        rx = [r["regex_f1"] or 0.0 for r in per_q]
        out["regex_overall_f1_x100"] = 100 * sum(rx) / len(rx) if rx else 0.0
    return out


def regex_f1(gold: list[str], text: str, universe: list[str]) -> float:
    """Secondary scorer: the repo's own company matcher (``herb_ladder.extract_companies``, whole-word
    registry names found in the prose) with its ``score_set`` F1. Not the headline."""
    from benchmark import herb_ladder as h

    return float(h.score_set(h.extract_companies(text, universe), set(gold))["f1"])


def company_universe(herb_dir: Path) -> list[str]:
    meta = herb_dir / "data" / "metadata" / "customers_data.json"
    return sorted({c["company"] for c in json.loads(meta.read_text()) if c.get("company")})


def cmd_score(args: argparse.Namespace) -> int:
    herb_dir = Path(args.herb_dir).expanduser()
    out = Path(args.out_dir)
    answers_f = out / "answers.jsonl"
    extract = openai_extractor() if args.extractor == "openai" else sonnet_extractor(
        args.sonnet_model, args.claude_bin)  # fails clearly, before any work, when its key/CLI is absent
    if not answers_f.exists():
        console.print(f"[red]no {answers_f}; run `answer` first[/red]")
        return 2
    qs = company_questions(load_products(herb_dir))
    cache_f = out / f"extractions-{args.extractor}.jsonl"
    cache: dict[str, list[str]] = {}
    if cache_f.exists():
        for line in cache_f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                cache[r["key"]] = r["names"]

    res = score_answers(qs, load_answers(answers_f), extract, cache,
                        None if args.no_regex else company_universe(herb_dir))
    with cache_f.open("w") as f:
        for k, v in cache.items():
            f.write(json.dumps({"key": k, "names": v}) + "\n")
    res["extractor"] = args.extractor
    res["label"] = EXTRACTOR_LABEL[args.extractor]
    (out / f"score-{args.extractor}.json").write_text(json.dumps(res, indent=1))
    t = Table(title=f"company F1 x100, {res['label']}")
    t.add_column("product")
    t.add_column("n", justify="right")
    t.add_column("mean F1 x100", justify="right")
    for p, v in res["per_product_f1_x100"].items():
        t.add_row(p, str(res["per_product_n"][p]), f"{v:.2f}")
    console.print(t)
    console.print(f"[bold]overall (n={res['n']}, failed rows = 0): {res['overall_f1_x100']:.2f}[/bold]  "
                  f"statuses {res['status_counts']}  (HERB Table 2 ReAct GPT-4o one attempt: 41.35)")
    if "regex_overall_f1_x100" in res:
        console.print(f"secondary regex scorer (not the headline): {res['regex_overall_f1_x100']:.2f}")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--herb-dir", default=str(rh.DEFAULT_HERB_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ix = sub.add_parser("index", help="build the pooled collection (free)")
    ix.add_argument("--fresh", action="store_true", help="ignore index-state.json and re-store every product")
    ix.add_argument("--limit-products", type=int, default=None, help="index only the first N products (test)")
    ix.set_defaults(fn=cmd_index)
    an = sub.add_parser("answer", help="nx_answer once per company question (PAID)")
    an.add_argument("--limit", type=int, default=None)
    an.add_argument("--start", type=int, default=0, help="0-based position in the 130-question order")
    an.add_argument("--cap-usd", type=float, default=DEFAULT_CAP_USD)
    an.add_argument("--no-save-plans", action="store_true",
                    help="nexus ad_hoc_ttl 0: no grown plan is saved between questions (NOT the prereg default)")
    an.set_defaults(fn=cmd_answer)
    sc = sub.add_parser("score", help="extract company names and score (the extractor is PAID)")
    sc.add_argument("--extractor", choices=["openai", "sonnet"], required=True)
    sc.add_argument("--sonnet-model", default="sonnet")
    sc.add_argument("--claude-bin", default=None)
    sc.add_argument("--no-regex", action="store_true", help="skip the secondary regex scorer")
    sc.set_defaults(fn=cmd_score)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
