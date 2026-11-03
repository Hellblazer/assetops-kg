"""Deterministic multi-hop entity resolution for HERB company questions.

This is the FIXED HAND-WRITTEN SCRIPT, a diagnostic baseline. It is not nx_answer and
must never be scored as the end-to-end rung (RDR-147 re-plan 2026-11-02, finding 2).

    hop 1  search(question)                 -> issue/slack artifacts naming CUST ids
    hop 2  extract CUST ids from hop-1 text  -> {CUST-0089, CUST-0115, ...}
    hop 3  for each id: search WHERE title=id -> the exact customer record   (deterministic)
    hop 4  extract company name from each record

Why hop 3 uses a metadata exact-filter, not semantic re-query: the customer records are
near-identical sentences ("Customer CUST-xxxx: contact N, R at company C"), so they
collide in embedding space; a fuzzy search for "CUST-0035" returns CUST-0097. Exact-key
resolution needs ``--where title=<id>``.

Reads go through ``nx search --json`` and use each hit's FULL ``content``. The old
harness used ``nx search -c``, which truncates every hit to 200 characters; in SearchFlow
25 of the 37 Slack messages that carry a CUST id put the first id at offset >= 190, so
the truncated read lost most of them. Any subprocess or parse failure raises ``NxError``;
the old harness returned "" and a broken sandbox read as recall 0.

Run inside the isolated sandbox:
    ./benchmark/herb_sandbox.sh run python3 -m benchmark.multihop_resolve
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()

CORPUS = "knowledge__herb-searchflow"
HERB_DIR = Path(os.environ.get("HERB_DIR") or "~/git/HERB").expanduser()
_RE_CUST = re.compile(r"CUST-\d{4}")
_RE_COMPANY = re.compile(r"at company ([^.\n]+?)\.")

SEARCH_TIMEOUT_S = 120


class NxError(RuntimeError):
    """An nx subprocess failed, timed out, or returned something that is not the JSON we asked for."""


def artifact_id_from_title(title: str) -> str:
    """Strip chunk and duplicate suffixes (``id:chunk-N``, ``id#section``, ``id#dupN``) back to the id."""
    return title.split(":")[0].split("#")[0].strip()


def search_hits(query: str, k: int, where: str | None = None, corpus: str = CORPUS) -> list[dict[str, Any]]:
    """Run ``nx search --json`` and return the hit dicts (full chunk text under ``content``).

    Raises ``NxError`` on a non-zero exit, a timeout, or stdout that is not a JSON list. A
    ``--corpus`` that matches no collection exits 0 with a warning and no JSON, so the parse
    check is what turns a missing index into an error instead of an empty result.
    """
    cmd = ["nx", "search", query, "--corpus", corpus, "-m", str(k), "--json"]
    if where:
        cmd += ["--where", where]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=SEARCH_TIMEOUT_S, check=True)
    except subprocess.CalledProcessError as e:
        raise NxError(f"nx search failed (exit {e.returncode}): {(e.stderr or b'').decode(errors='replace')[-400:]}") from e
    except subprocess.TimeoutExpired as e:
        raise NxError(f"nx search timed out after {SEARCH_TIMEOUT_S}s: {query[:60]!r}") from e
    out = proc.stdout.decode(errors="replace")
    try:
        hits = json.loads(out)
    except json.JSONDecodeError as e:
        raise NxError(f"nx search --json returned non-JSON ({out[:200]!r})") from e
    if not isinstance(hits, list):
        raise NxError(f"nx search --json returned {type(hits).__name__}, expected a list")
    return hits


def _search(query: str, k: int, where: str | None = None) -> str:
    """Full text of the top-k hits, joined. Compatibility shape for callers that regex the text."""
    return "\n".join(h.get("content", "") for h in search_hits(query, k, where))


def resolve_company(cust_id: str) -> str | None:
    """Hop 3+4: exact-filter retrieve the customer record, extract its company."""
    out = _search("customer company contact", k=1, where=f"title={cust_id}")
    m = _RE_COMPANY.search(out)
    return m.group(1).strip() if m else None


def companies_for_ids(ids: list[str]) -> set[str]:
    """Hop 3+4 over a list of CUST ids."""
    return {c for cid in ids if (c := resolve_company(cid))}


def answer_company(question: str, k: int = 20, only_artifacts: set[str] | None = None) -> set[str]:
    """The script. ``only_artifacts`` is the oracle filter: hop-1 hits whose artifact id is not in
    the set are dropped before id extraction (a ceiling variant, never the product)."""
    hits = search_hits(question, k)                                    # hop 1
    if only_artifacts is not None:
        hits = [h for h in hits if artifact_id_from_title(h.get("title", "")) in only_artifacts]
    ids = sorted({cid for h in hits for cid in _RE_CUST.findall(h.get("content", ""))})  # hop 2
    return companies_for_ids(ids)                                      # hop 3+4


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
    console.print(f"\n[bold]hand-written script company recall: {mean:.3f}[/bold]  (diagnostic baseline, not nx_answer)")


if __name__ == "__main__":
    main()
