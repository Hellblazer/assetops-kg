"""Run the HERB benchmark (Salesforce AI Research) against nexus retrieval.

HERB — "Benchmarking Deep Search over Heterogeneous Enterprise Data"
(arXiv 2506.23139). Each of its 30 synthetic products is a self-contained
heterogeneous corpus (Slack, documents, meeting transcripts, URLs, GitHub PRs)
plus answerable/unanswerable Q&A. Every answerable question carries
``{question, ground_truth, citations:[artifact_id], type}`` where
``type in {person, company, url, pr, content}``.

This runner adapts HERB so nexus (the conexus knowledge system) is the
retrieval/answer engine instead of HERB's LlamaIndex ReAct agent. The same
data + the same downstream scorer (HERB's ``code/evaluate.py``), with the
engine swapped — the assetops-kg playbook applied to a third-party benchmark.

Scope of THIS scaffold (SearchFlow smoke test):
  IMPLEMENTED, deterministic, offline-runnable:
    - load a product corpus and flatten its 5 artifact types into uniform
      ``ArtifactRecord``s keyed by the citation ``id`` scheme
    - corpus statistics + gold-citation extraction
    - retrieval-recall@k metric against gold citations
    - emit HERB-compatible output (``code/evaluate.py`` consumes this shape)
  IMPLEMENTED, needs a live nexus index:
    - ``--mode index``: materialize artifacts into a T3 collection via `nx store put`
    - ``--mode eval`` retrieval arm: query `nx search --json`, map hits -> artifact ids
    - ``--answer-arm nx_answer``: the real composed answer arm — calls the
      ``nx_answer`` function out-of-process via the conexus interpreter
      (plan-match-first retrieval); ``--answer-arm retrieval`` (default) is the
      cheap recall-only baseline
  DEFERRED (clearly marked, not silently stubbed):
    - wiring HERB's ``evaluate.py`` (F1 for person/url/pr/company, GPT-4o
      Likert for content) as an in-process scoring step. For now the runner
      emits the HERB-shaped pickle and prints the score command to run.

Caveats inherited from the HERB dataset (see the indexed critique,
T3 knowledge): GPT-4o circularity on ``content`` questions, ~53 templated
question patterns, 8 products with the same question text in both
answerable and unanswerable pools, and real-OSS-PR parametric leakage.
Treat absolute scores as opaque for cross-system comparison; the
model-agnostic signal is retrieval recall + the F1 entity types.

Usage:
    python -m benchmark.run_herb --mode prep                       # offline: load + flatten + stats
    python -m benchmark.run_herb --mode index --limit 200          # materialize corpus into nexus (slow)
    python -m benchmark.run_herb --mode eval --limit 20            # retrieve + recall@k + emit HERB output
    python -m benchmark.run_herb --product SearchFlow --output results/herb_searchflow.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import subprocess
import sys
import textwrap
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_HERB_DIR = Path("~/git/HERB").expanduser()
DEFAULT_PRODUCT = "SearchFlow"
# nexus normalizes a bare name to ``knowledge__<name>``; one collection per product.
COLLECTION_PREFIX = "herb"
ARTIFACT_TYPES = ("slack", "documents", "meeting_transcripts", "urls", "prs")

# nx_answer is MCP-only (no `nx answer` CLI), but it is a plain async function in
# the conexus tool. We invoke it out-of-process via the conexus interpreter so the
# project venv stays decoupled from the nexus install. Override with NX_PYTHON.
CONEXUS_PYTHON = Path(
    os.environ.get("NX_PYTHON", "~/.local/share/uv/tools/conexus/bin/python3")
).expanduser()
# stdout sentinel so we can separate the JSON envelope from nexus's structlog noise.
_NXA_SENTINEL = "__NXA_RESULT__"


# ---------------------------------------------------------------------------
# Artifact model — the citation id scheme is the contract with the gold labels
# ---------------------------------------------------------------------------


@dataclass
class ArtifactRecord:
    """One retrievable unit. ``id`` MUST match the value used in question
    ``citations`` so retrieval recall can be scored."""

    id: str
    source_type: str
    text: str
    title: str = ""


def _slack_user(node: dict[str, Any]) -> dict[str, Any]:
    """Slack message/reply bodies nest the author under ``Message.User``."""
    return (node.get("Message") or node).get("User") or {}


def _slack_text(msg: dict[str, Any]) -> str:
    """Flatten a Slack message + its thread replies into searchable text.

    Schema (verified against the HERB data): the body lives at
    ``Message.User.{userId,text}``; ``ThreadReplies`` is empty across the whole
    dataset but is handled defensively in case future products populate it.
    """
    parts: list[str] = []
    channel = (msg.get("Channel") or {}).get("name", "")
    if channel:
        parts.append(f"[channel:{channel}]")
    author = _slack_user(msg)
    if author.get("text"):
        parts.append(f"{author.get('userId', '?')}: {author['text']}")
    for reply in msg.get("ThreadReplies") or []:
        ru = _slack_user(reply)
        if ru.get("text"):
            parts.append(f"  reply {ru.get('userId', '?')}: {ru['text']}")
    return "\n".join(parts)


def _pr_text(pr: dict[str, Any]) -> str:
    parts = [pr.get("title", ""), pr.get("summary", "")]
    parts.append(f"link: {pr.get('link', '')}  state: {pr.get('state', '')}  author: {(pr.get('user') or {}).get('login', '')}")
    for rev in pr.get("reviews") or []:
        parts.append(f"review[{rev.get('state', '')}] {(rev.get('user') or {}).get('login', '')}: {rev.get('comment', '')}")
    return "\n".join(p for p in parts if p)


def flatten_artifacts(product: dict[str, Any]) -> list[ArtifactRecord]:
    """Flatten one product JSON into uniform, citation-id-keyed records.

    Each artifact type stores its id under ``id``; this is what question
    ``citations`` reference. Text extraction is per-type because the schemas
    differ (Slack nests User/text, PRs nest reviews, etc.).
    """
    records: list[ArtifactRecord] = []

    for msg in product.get("slack", []):
        records.append(ArtifactRecord(msg["id"], "slack", _slack_text(msg)))

    for doc in product.get("documents", []):
        records.append(
            ArtifactRecord(doc["id"], "documents", f"[{doc.get('type', '')}] {doc.get('content', '')}", title=doc.get("type", ""))
        )

    for tr in product.get("meeting_transcripts", []):
        records.append(
            ArtifactRecord(tr["id"], "meeting_transcripts", f"[{tr.get('document_type', '')}] {tr.get('transcript', '')}")
        )

    for url in product.get("urls", []):
        records.append(
            ArtifactRecord(url["id"], "urls", f"{url.get('description', '')}\n{url.get('link', '')}")
        )

    for pr in product.get("prs", []):
        records.append(ArtifactRecord(pr["id"], "prs", _pr_text(pr)))

    return records


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------


def load_product(herb_dir: Path, product: str) -> dict[str, Any]:
    path = herb_dir / "data" / "products" / f"{product}.json"
    if not path.exists():
        console.print(f"[red]Product file not found:[/red] {path}")
        console.print("Clone HERB: git clone https://github.com/SalesforceAIResearch/HERB.git ~/git/HERB")
        sys.exit(1)
    return json.loads(path.read_text())


def collection_name(product: str) -> str:
    """Bare collection name passed to ``nx store put --collection``."""
    return f"{COLLECTION_PREFIX}-{product.lower()}"


def corpus_arg(product: str) -> str:
    """Corpus selector for ``nx search``/``nx_answer``.

    A bare name like ``herb-searchflow`` does NOT resolve; nexus matches the
    ``<content_type>__<name>`` prefix. store_put routes to the ``knowledge__``
    family, so retrieval must ask for ``knowledge__herb-<product>``.
    """
    return f"knowledge__{collection_name(product)}"


# ---------------------------------------------------------------------------
# Indexing arm — materialize artifacts into a nexus T3 collection
# ---------------------------------------------------------------------------


def index_corpus(records: list[ArtifactRecord], product: str, limit: int | None) -> int:
    """Store each artifact in ``knowledge__herb-<product>`` with title=id.

    Uses ``nx store put -`` (stdin) so a search hit's ``title`` recovers the
    artifact id. Deliberately one-call-per-artifact for scaffold clarity;
    a production loader would batch. Slow — use ``--limit`` for smoke tests.
    """
    coll = collection_name(product)
    subset = records[:limit] if limit else records
    console.print(f"Indexing {len(subset)} artifacts -> [cyan]knowledge__{coll}[/cyan] (this is slow; --limit to cap)")
    ok = 0
    for i, rec in enumerate(subset, 1):
        if not rec.text.strip():
            continue
        try:
            subprocess.run(
                ["nx", "store", "put", "-", "--collection", coll, "--title", rec.id,
                 "--tags", f"herb,{product},{rec.source_type}", "--ttl", "permanent"],
                input=rec.text.encode(), capture_output=True, timeout=120, check=True,
            )
            ok += 1
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning("store put failed for %s: %s", rec.id, e)
        if i % 100 == 0:
            console.print(f"  {i}/{len(subset)}")
    console.print(f"[green]Indexed {ok}/{len(subset)} artifacts[/green]")
    return ok


# ---------------------------------------------------------------------------
# Retrieval arm — query nexus, map hits back to artifact ids
# ---------------------------------------------------------------------------


def _artifact_id_from_title(title: str) -> str:
    """Strip nexus chunk suffixes (``id:chunk-N`` / ``id#section``) back to the id."""
    return title.split(":")[0].split("#")[0].strip()


def retrieve(question: str, product: str, k: int) -> list[str]:
    """Return up to ``k`` artifact ids ranked by nexus retrieval, deduped in order."""
    try:
        proc = subprocess.run(
            ["nx", "search", question, "--corpus", corpus_arg(product), "-m", str(k), "--json"],
            capture_output=True, timeout=120, check=True,
        )
        hits = json.loads(proc.stdout.decode() or "[]")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        logger.warning("search failed for %r: %s", question[:60], e)
        return []
    seen: list[str] = []
    for h in hits:
        aid = _artifact_id_from_title(h.get("title", ""))
        if aid and aid not in seen:
            seen.append(aid)
    return seen


def recall_at_k(retrieved: list[str], gold: list[str]) -> float:
    if not gold:
        return 0.0
    hit = len(set(retrieved) & set(gold))
    return hit / len(set(gold))


# ---------------------------------------------------------------------------
# Answer arm
# ---------------------------------------------------------------------------

# Inline script run by the conexus interpreter: call nx_answer(structured=True)
# and print the envelope as one sentinel-prefixed JSON line on stdout. nexus's
# own structlog output goes to stderr and is discarded.
_NXA_DRIVER = textwrap.dedent(
    """
    import asyncio, json, sys
    from nexus.mcp.core import nx_answer
    async def go():
        r = await nx_answer(question=sys.argv[1], scope=sys.argv[2],
                            structured=True, max_steps=int(sys.argv[3]))
        if isinstance(r, dict):
            out = {{"final_text": r.get("final_text"), "chunks": r.get("chunks") or [],
                    "plan_id": r.get("plan_id"), "step_count": r.get("step_count")}}
        else:
            out = {{"final_text": r, "chunks": [], "plan_id": None, "step_count": None}}
        print("{sentinel}" + json.dumps(out))
    asyncio.run(go())
    """
).format(sentinel=_NXA_SENTINEL)


def nx_answer_call(question: str, scope: str, max_steps: int, timeout: int) -> dict[str, Any] | None:
    """Invoke nx_answer out-of-process; return the structured envelope or None.

    NB latency: nx_answer spawns claude -p per operator step. Observed
    distribution skews to 30s-4min per call (one 6-step plan measured 244s).
    Budget accordingly — this is the expensive arm.
    """
    if not CONEXUS_PYTHON.exists():
        logger.warning("conexus interpreter not found at %s; set NX_PYTHON", CONEXUS_PYTHON)
        return None
    try:
        proc = subprocess.run(
            [str(CONEXUS_PYTHON), "-c", _NXA_DRIVER, question, scope, str(max_steps)],
            capture_output=True, timeout=timeout, check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.warning("nx_answer failed for %r: %s", question[:60], e)
        return None
    for line in proc.stdout.decode().splitlines():
        if line.startswith(_NXA_SENTINEL):
            return json.loads(line[len(_NXA_SENTINEL):])
    return None


def answer_question(question: str, product: str, k: int, arm: str, max_steps: int, timeout: int) -> str | None:
    """Produce a HERB-scorable answer string.

    ``arm="nx_answer"`` — the real arm: plan-match-first composed retrieval over
    the product collection. Returns nx_answer's ``final_text``.
    ``arm="retrieval"`` — cheap v0 baseline: the ids of the top-k retrieved
    artifacts joined. Useful for a zero-LLM-cost recall-only smoke run; entity-
    typed F1 will be near-zero because there is no real answer synthesis.
    """
    if arm == "nx_answer":
        env = nx_answer_call(question, corpus_arg(product), max_steps, timeout)
        return env.get("final_text") if env else None
    ids = retrieve(question, product, k)
    return " ".join(ids) if ids else None


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, timeout=30, check=True).stdout.decode().strip()
    except Exception:
        return ""


def build_manifest(product: str, k: int, arm: str, n_questions: int, n_artifacts: int,
                   herb_dir: Path) -> dict[str, Any]:
    """Capture everything needed to reproduce and interpret a run.

    Versioned alongside the results so a number is never orphaned from the
    config that produced it: tool/model/backend identity, both repo SHAs,
    the embedding model, and the isolation context.
    """
    embed_model = ""
    for line in _sh(["nx", "doctor"]).splitlines():
        if "Embedding model:" in line:
            embed_model = line.split("Embedding model:")[-1].strip()
            break
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "nx_version": _sh(["nx", "--version"]),
        "embedding_model": embed_model,
        "chroma_backend": "local" if os.environ.get("NX_LOCAL") == "1" else "cloud",
        "nexus_config_dir": os.environ.get("NEXUS_CONFIG_DIR", "<default>"),
        "local_chroma_path": os.environ.get("NX_LOCAL_CHROMA_PATH", "<default>"),
        "assetops_sha": _sh(["git", "rev-parse", "--short", "HEAD"]),
        "herb_sha": _sh(["git", "-C", str(herb_dir), "rev-parse", "--short", "HEAD"]),
        "product": product,
        "collection": corpus_arg(product),
        "k": k,
        "answer_arm": arm,
        "n_questions": n_questions,
        "n_artifacts": n_artifacts,
    }


@dataclass
class HerbDataset:
    """HERB-``evaluate.py``-compatible result container (pickled as-is)."""

    question: list[str] = field(default_factory=list)
    answer: list[Any] = field(default_factory=list)
    ground_truth: list[str] = field(default_factory=list)
    citations: list[list[str]] = field(default_factory=list)
    type: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question, "answer": self.answer,
            "ground_truth": self.ground_truth, "citations": self.citations, "type": self.type,
        }


def run_eval(product_json: dict[str, Any], product: str, k: int, limit: int | None,
             arm: str, max_steps: int, timeout: int) -> dict[str, Any]:
    """Eval over the answerable questions.

    Retrieval recall@k is always measured via `nx search` (title -> artifact id),
    independent of the answer arm — it is the model-agnostic, near-zero-cost
    signal. The answer arm (``retrieval`` or ``nx_answer``) only fills the
    HERB ``answer`` field for downstream evaluate.py scoring.
    """
    questions = product_json.get("answerable_questions", [])
    if limit:
        questions = questions[:limit]

    ds = HerbDataset()
    recalls: list[float] = []
    by_type: dict[str, list[float]] = {}
    per_question: list[dict[str, Any]] = []

    for i, q in enumerate(questions, 1):
        gold = q.get("citations", [])
        t0 = time.monotonic()
        retrieved = retrieve(q["question"], product, k)
        retr_ms = round((time.monotonic() - t0) * 1000)
        r = recall_at_k(retrieved, gold)
        recalls.append(r)
        by_type.setdefault(q.get("type", "?"), []).append(r)

        t1 = time.monotonic()
        answer = answer_question(q["question"], product, k, arm, max_steps, timeout)
        ans_ms = round((time.monotonic() - t1) * 1000)

        ds.question.append(q["question"])
        ds.answer.append(answer)
        ds.ground_truth.append(q.get("ground_truth", ""))
        ds.citations.append(gold)
        ds.type.append(q.get("type", "?"))

        per_question.append({
            "question": q["question"], "type": q.get("type", "?"),
            "gold_citations": gold, "retrieved_ids": retrieved,
            "recall_at_k": r, "retrieval_ms": retr_ms, "answer_ms": ans_ms,
            "answer_is_none": answer is None,
        })
        if arm == "nx_answer":  # the slow arm — show progress
            console.print(f"  [{i}/{len(questions)}] recall@{k}={r:.2f} ({retr_ms}ms+{ans_ms}ms)  {q['question'][:50]}")

    summary = {
        "questions_evaluated": len(questions),
        "mean_recall_at_k": sum(recalls) / len(recalls) if recalls else 0.0,
        "recall_by_type": {t: sum(v) / len(v) for t, v in by_type.items()},
        "count_by_type": {t: len(v) for t, v in by_type.items()},
        "total_retrieval_ms": sum(p["retrieval_ms"] for p in per_question),
        "total_answer_ms": sum(p["answer_ms"] for p in per_question),
    }
    return {"summary": summary, "per_question": per_question, "herb_dataset": ds.as_dict()}


def print_stats(records: list[ArtifactRecord], product_json: dict[str, Any], product: str) -> None:
    by_type: dict[str, int] = {}
    for r in records:
        by_type[r.source_type] = by_type.get(r.source_type, 0) + 1
    ans = product_json.get("answerable_questions", [])
    unans = product_json.get("unanswerable_questions", [])
    q_types: dict[str, int] = {}
    for q in ans:
        q_types[q.get("type", "?")] = q_types.get(q.get("type", "?"), 0) + 1

    console.print(f"\n[bold]HERB product:[/bold] {product}")
    console.print(f"  artifacts: {len(records)}  {by_type}")
    console.print(f"  answerable questions: {len(ans)}  {q_types}")
    console.print(f"  unanswerable questions: {len(unans)}")
    console.print(f"  target collection: knowledge__{collection_name(product)}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Run the HERB benchmark against nexus retrieval.")
    parser.add_argument("--mode", choices=["prep", "index", "eval"], default="prep",
                        help="prep: offline load+flatten+stats; index: materialize into nexus; eval: retrieve+recall")
    parser.add_argument("--herb-dir", type=Path, default=DEFAULT_HERB_DIR, help="Path to the cloned HERB repo")
    parser.add_argument("--product", default=DEFAULT_PRODUCT, help="Product corpus to use (e.g. SearchFlow)")
    parser.add_argument("--k", type=int, default=10, help="Retrieval depth for recall@k")
    parser.add_argument("--limit", type=int, default=None, help="Cap artifacts (index) or questions (eval) for smoke tests")
    parser.add_argument("--answer-arm", choices=["retrieval", "nx_answer"], default="retrieval",
                        help="retrieval: cheap recall-only baseline; nx_answer: real composed arm (slow, ~30s-4min/question)")
    parser.add_argument("--max-steps", type=int, default=6, help="nx_answer plan DAG cap")
    parser.add_argument("--answer-timeout", type=int, default=320, help="Per-question nx_answer subprocess timeout (s)")
    parser.add_argument("--output", type=Path, default=None, help="Write eval results JSON here")
    args = parser.parse_args()

    product_json = load_product(args.herb_dir, args.product)
    records = flatten_artifacts(product_json)
    print_stats(records, product_json, args.product)

    if args.mode == "prep":
        return

    if args.mode == "index":
        index_corpus(records, args.product, args.limit)
        return

    # mode == eval
    console.print(f"\n[bold]Evaluating[/bold] (k={args.k}, limit={args.limit}, answer-arm={args.answer_arm})…")
    n_q = len(product_json.get("answerable_questions", [])[: args.limit] if args.limit
              else product_json.get("answerable_questions", []))
    manifest = build_manifest(args.product, args.k, args.answer_arm, n_q, len(records), args.herb_dir)
    results = run_eval(product_json, args.product, args.k, args.limit,
                       args.answer_arm, args.max_steps, args.answer_timeout)
    results = {"manifest": manifest, **results}
    console.print("[bold]manifest:[/bold]")
    console.print(json.dumps(manifest, indent=2))
    console.print("[bold]summary:[/bold]")
    console.print(json.dumps(results["summary"], indent=2))

    # Versioned output path: results/herb/<product>-<arm>-<assetops_sha>.json by default.
    out = args.output or (
        Path("results/herb") / f"{args.product.lower()}-{args.answer_arm}-{manifest['assetops_sha']}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    console.print(f"[green]Wrote[/green] {out}")
    # HERB's code/evaluate.py consumes a pickle of the dataset dict.
    pkl = out.with_suffix(".pk")
    with pkl.open("wb") as f:
        pickle.dump(results["herb_dataset"], f)
    console.print(f"[green]Wrote HERB-shaped pickle[/green] {pkl}  "
                  f"(score with: python HERB/code/evaluate.py --output_file {pkl})")


if __name__ == "__main__":
    main()
