"""Deterministic multi-hop entity resolution for HERB company questions.

Demonstrates the discovered plan (not derived from HERB, but from the structural
type-mismatch between the requested answer and the retrievable evidence):

    hop 1  search(question)                 -> issue/slack artifacts naming CUST ids
    hop 2  extract CUST ids from hop-1 text  -> {CUST-0089, CUST-0115, ...}
    hop 3  for each id: search WHERE title=id -> the exact customer record   (deterministic)
    hop 4  extract company name from each record

Why hop 3 uses a metadata exact-filter, not semantic re-query: the 120 customer
records are near-identical sentences ("Customer CUST-xxxx: contact N, R at company C"),
so they collide in embedding space — a fuzzy search for "CUST-0035" returns CUST-0097.
Exact-key resolution requires `where title=<id>`. This is the load-bearing finding:
nexus could DISCOVER the multi-hop need (a type-aware planner says "these are ids, not
names") but could not EXECUTE it on the fuzzy-retrieval substrate; the join needs a
deterministic key lookup.

Run inside the isolated sandbox:
    ./benchmark/herb_sandbox.sh .venv/bin/python -m benchmark.multihop_resolve
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from rich.console import Console

console = Console()

CORPUS = "knowledge__herb-searchflow"
HERB_DIR = Path("~/git/HERB").expanduser()
_RE_CUST = re.compile(r"CUST-\d{4}")
_RE_COMPANY = re.compile(r"at company ([^.\n]+?)\.")


def _search(query: str, k: int, where: str | None = None) -> str:
    cmd = ["nx", "search", query, "--corpus", CORPUS, "-m", str(k), "-c"]
    if where:
        cmd += ["--where", where]
    try:
        return subprocess.run(cmd, capture_output=True, timeout=120, check=True).stdout.decode()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return ""


def resolve_company(cust_id: str) -> str | None:
    """Hop 3+4: exact-filter retrieve the customer record, extract its company."""
    out = _search("customer company contact", k=1, where=f"title={cust_id}")
    m = _RE_COMPANY.search(out)
    return m.group(1).strip() if m else None


def answer_company(question: str, k: int = 20) -> set[str]:
    issues = _search(question, k=k)                       # hop 1
    ids = sorted(set(_RE_CUST.findall(issues)))           # hop 2
    companies = {c for cid in ids if (c := resolve_company(cid))}  # hop 3+4
    return companies


def main() -> None:
    product = json.loads((HERB_DIR / "data/products/SearchFlow.json").read_text())
    qs = [q for q in product["answerable_questions"] if q.get("type") == "company"]
    recalls = []
    for q in qs:
        gold = {g.lower() for g in q["ground_truth"]}
        pred = {c.lower() for c in answer_company(q["question"])}
        hit = len(gold & pred)
        recall = hit / len(gold) if gold else 0.0
        recalls.append(recall)
        console.print(f"recall={recall:.2f}  hit={hit}/{len(gold)}  "
                      f"resolved={sorted(pred)[:5]}  | {q['question'][:48]}")
    mean = sum(recalls) / len(recalls) if recalls else 0.0
    console.print(f"\n[bold]deterministic multi-hop company recall: {mean:.3f}[/bold]  "
                  f"(single-hop nx_answer baseline was 0.000)")


if __name__ == "__main__":
    main()
