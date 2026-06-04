"""Deterministic, local HERB scorer for the entity-typed (F1) question types.

HERB's own ``code/evaluate.py`` scores ``person/url/pr/company`` by asking GPT-4o
to extract the relevant entities from a candidate answer, then set-F1 against the
gold list. That bakes a GPT-4o dependency (and circularity) into the metric. For
the structured types the extraction is a regex job — this scorer does it
deterministically and locally, reusing HERB's exact ``normalize`` + ``f1_score_sets``
so the F1 numbers are directly comparable, minus the model dependency.

Coverage:
  - person / url / pr  -> full deterministic set-F1 (regex extract -> normalize -> F1)
  - company            -> free-text proper nouns; reported as gold-membership RECALL
                          only (precision needs NER/a judge). Flagged, not a clean F1.
  - content            -> requires a Likert judge (GPT-4o in HERB); SKIPPED here.

Usage:
    python -m benchmark.score_herb results/herb/searchflow-nx_answer-full.json
    python -m benchmark.score_herb a.json b.json        # compare two runs side by side
"""

from __future__ import annotations

import argparse
import json
import re
import string
import subprocess
import sys
from pathlib import Path
from typing import Any

# --- HERB's normalize + F1 (reimplemented; identical semantics to evaluate.py) ---


def _normalize(answer: str) -> str:
    answer = answer.lower()
    answer = "".join(ch for ch in answer if ch not in set(string.punctuation))
    answer = re.sub(r"\b(a|an|the)\b", " ", answer)
    return " ".join(answer.split())


def f1_score_sets(y_true: list[str], y_pred: list[str]) -> float:
    t = {_normalize(e) for e in y_true if _normalize(e)}
    p = {_normalize(e) for e in y_pred if _normalize(e)}
    if not t and not p:
        return 1.0
    tp = len(t & p)
    fp = len(p - t)
    fn = len(t - p)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


# --- deterministic per-type entity extractors over the candidate answer text ---

_RE_PERSON = re.compile(r"eid_[0-9a-fA-F]{6,}")
_RE_PR = re.compile(r"https?://github\.com/\S+?/pull/\d+")
_RE_URL = re.compile(r"https?://[^\s)\]\">]+")


def _extract(answer: str, qtype: str, gold: list[str]) -> list[str]:
    if not answer:
        return []
    if qtype == "person":
        return _RE_PERSON.findall(answer)
    if qtype == "pr":
        return [u.rstrip(".,);") for u in _RE_PR.findall(answer)]
    if qtype == "url":
        return [u.rstrip(".,);") for u in _RE_URL.findall(answer)]
    if qtype == "company":
        # free-text: recall-only membership (precision not deterministically measurable)
        return [c for c in gold if c.lower() in answer.lower()]
    return []


# --- content Likert via a local claude-as-judge (no OpenAI; not GPT-4o => no circularity) ---

# Mirrors HERB evaluate.py:answer_likert_score so the rubric is comparable; only
# the judge model differs (claude, local, via `claude -p`). Reported as a SEPARATE
# metric ("content_likert_claude"), never as HERB's official GPT-4o score.
_LIKERT_PROMPT = (
    "You are an expert evaluator. Given a question, a reference answer, and a candidate "
    "answer, evaluate how well the candidate aligns with the reference. Focus on accuracy, "
    "completeness, and relevance. If the candidate includes extra information, check if it is "
    "correct and appropriate; if it omits key points from the reference, weigh that down.\n\n"
    "Question: {q}\nReference Answer: {r}\nCandidate Answer: {c}\n\n"
    'Respond with ONLY a JSON object: {{"score": <integer 0-100>, "reason": "<brief>"}}'
)


def claude_likert(question: str, reference: str, candidate: str, trials: int, timeout: int) -> float | None:
    """Mean Likert score (0-100) from `trials` independent claude-judge calls, or None."""
    prompt = _LIKERT_PROMPT.format(q=question, r=reference, c=candidate or "(no answer)")
    scores: list[int] = []
    for _ in range(max(1, trials)):
        try:
            proc = subprocess.run(
                ["claude", "-p", prompt], stdin=subprocess.DEVNULL,
                capture_output=True, timeout=timeout, check=True,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            continue
        out = proc.stdout.decode()
        match = re.search(r"\{[^{}]*\"score\"[^{}]*\}", out, re.DOTALL)
        if not match:
            continue
        try:
            scores.append(int(json.loads(match.group(0))["score"]))
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            continue
    return sum(scores) / len(scores) if scores else None


def score_file(path: Path, judge_content: bool = False, trials: int = 1, timeout: int = 120) -> dict[str, Any]:
    data = json.loads(path.read_text())
    hd = data["herb_dataset"]
    by_type: dict[str, list[float]] = {}
    company_recall: list[float] = []
    content_likert: list[float] = []
    n_content = 0
    for q, ans, gt, qtype in zip(hd["question"], hd["answer"], hd["ground_truth"], hd["type"]):
        gold = gt if isinstance(gt, list) else []
        if qtype == "content":
            n_content += 1
            if judge_content:
                ref = gt if isinstance(gt, str) else "\n".join(gt)
                s = claude_likert(q, ref, ans or "", trials, timeout)
                if s is not None:
                    content_likert.append(s)
            continue
        if qtype == "company":
            hit = len({c for c in gold if c.lower() in (ans or "").lower()})
            company_recall.append(hit / len(gold) if gold else 0.0)
            continue
        pred = _extract(ans or "", qtype, gold)
        by_type.setdefault(qtype, []).append(f1_score_sets(gold, pred))

    mean = lambda v: sum(v) / len(v) if v else 0.0  # noqa: E731
    f1_types = {t: round(mean(v), 4) for t, v in by_type.items()}
    all_f1 = [x for v in by_type.values() for x in v]
    return {
        "file": path.name,
        "answer_arm": data.get("manifest", {}).get("answer_arm", "?"),
        "f1_by_type": f1_types,
        "n_by_type": {t: len(v) for t, v in by_type.items()},
        "macro_f1_structured": round(mean(list(f1_types.values())), 4),
        "micro_f1_structured": round(mean(all_f1), 4),
        "company_recall": round(mean(company_recall), 4),
        "n_company": len(company_recall),
        "content_likert_claude": round(mean(content_likert), 2) if content_likert else None,
        "n_content": n_content,
        "n_content_judged": len(content_likert),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic local HERB F1 scorer (person/url/pr/company).")
    parser.add_argument("results", nargs="+", type=Path, help="One or more run JSONs from run_herb.py --mode eval")
    parser.add_argument("--judge-content", action="store_true",
                        help="Score content questions with a local claude-as-judge Likert (0-100); slow (~12s/call)")
    parser.add_argument("--trials", type=int, default=1, help="Judge trials per content question (mean); default 1")
    parser.add_argument("--judge-timeout", type=int, default=120, help="Per-judge-call timeout (s)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Persist the scored comparison to JSON (pins non-deterministic content Likert)")
    args = parser.parse_args()

    scored = [score_file(p, args.judge_content, args.trials, args.judge_timeout) for p in args.results]
    print(json.dumps(scored, indent=2))

    if args.output:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True).stdout.decode().strip()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "scorer": "benchmark/score_herb.py",
            "judged_content": args.judge_content,
            "judge": "local claude-as-judge (non-deterministic; NOT HERB's GPT-4o)" if args.judge_content else None,
            "trials": args.trials,
            "assetops_sha": sha,
            "inputs": [str(p) for p in args.results],
            "results": scored,
        }, indent=2))
        print(f"\nWrote {args.output}")

    # compact comparison table
    types = ["person", "pr", "url"]
    print("\n" + "=" * 64)
    hdr = f"{'metric':<22}" + "".join(f"{s['answer_arm'][:16]:>18}" for s in scored)
    print(hdr)
    print("-" * len(hdr))
    for t in types:
        row = f"F1 {t:<19}" + "".join(f"{s['f1_by_type'].get(t, 0.0):>18.3f}" for s in scored)
        print(row)
    for label, key in [("macro-F1 (struct)", "macro_f1_structured"),
                       ("micro-F1 (struct)", "micro_f1_structured"),
                       ("company recall", "company_recall")]:
        print(f"{label:<22}" + "".join(f"{s[key]:>18.3f}" for s in scored))
    cl = lambda s: f"{s['content_likert_claude']:>18.1f}" if s.get("content_likert_claude") is not None else f"{'(unscored)':>18}"  # noqa: E731
    print(f"{'content Likert/100':<22}" + "".join(cl(s) for s in scored))
    print("=" * 64)
    print("note: person/pr/url are deterministic set-F1 (HERB f1_score_sets, regex extract);")
    print("      company is recall-only (free-text). content Likert is a LOCAL claude-as-judge")
    print("      (0-100) mirroring HERB's rubric — a different judge than HERB's GPT-4o, so it")
    print("      is comparable in structure but NOT HERB's official score (and avoids circularity).")


if __name__ == "__main__":
    main()
