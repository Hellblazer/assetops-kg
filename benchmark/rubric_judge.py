"""Re-score existing NLQ trajectories against the AssetOpsBench 3-axis rubric.

Maps onto the LLM-As-Judge methodology described in the AssetOpsBench KDD 2026
paper (Section 5.1). Each trajectory is scored on:

  y1 = Task Completion        — did the agent address the user's intent?
  y2 = Data Retrieval Accuracy — were the underlying queries/results correct?
  y3 = Result Verification    — is the synthesised answer faithful & complete?

Each score is in [0, 1]. We use gpt-4o as the judge (proxy for the IBM
llama-4-maverick judge — same prompt structure) and run each trajectory 3
times to estimate judge variance, reporting the mean.

Input:  NLQ result JSON (list of per-scenario dicts as produced by run_nlq.py).
Output: a CSV with columns:
        id, type, passed, score, y1, y2, y3, y1_std, y2_std, y3_std
plus a JSON dump of the full per-trial scores for auditability.

Usage:
    python -m benchmark.rubric_judge \\
        --input  results/nlq_gpt4_full_results.json \\
        --output results/rubric/nlq_gpt4_rubric.csv \\
        --judge-model gpt-4o \\
        --trials 3
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

# Same async OpenAI dependency the benchmark already uses.
try:
    import openai
except ImportError:
    sys.exit("pip install openai")


RUBRIC_PROMPT = """\
You are an expert evaluator for AI agents that answer industrial-asset
maintenance questions over a knowledge graph. For ONE trajectory below,
return three scalar scores in [0, 1] following the AssetOpsBench rubric.

  y1 = Task Completion        — did the agent address the user's intent?
                                 0 = ignored the question; 1 = fully complete.
  y2 = Data Retrieval Accuracy — were the underlying graph queries appropriate
                                 and the retrieved data correct?
                                 0 = wrong/no data; 1 = exact, sufficient data.
  y3 = Result Verification    — is the final natural-language answer faithful
                                 to the retrieved data and complete?
                                 0 = hallucinated/contradicts data;
                                 1 = correct, well-grounded synthesis.

Return ONLY a JSON object: {{"y1": <float>, "y2": <float>, "y3": <float>}}.
No prose, no markdown fences.

--- TRAJECTORY ---
Question:        {question}
Scenario type:   {type}

Cypher queries the agent generated:
{cypher_block}

Cypher execution results (record counts):
{cypher_results}

Final answer the agent produced:
{response}

Rationale recorded by the test harness (may include keyword-overlap notes):
{rationale}
--- END TRAJECTORY ---
"""


def _format_trajectory(row: dict[str, Any]) -> str:
    details = row.get("nlq_details", {}) or {}
    cy = details.get("cypher_generated") or []
    cy_results = details.get("cypher_results") or []
    cy_block = "\n".join(f"  - {q}" for q in cy) if cy else "  (none)"
    cy_res_block = "\n".join(
        f"  - {r.get('record_count', '?')} records, "
        f"{r.get('execution_ms', 0):.1f}ms, success={r.get('success')}"
        for r in cy_results
    ) or "  (none)"
    return RUBRIC_PROMPT.format(
        question=row.get("question", ""),
        type=row.get("type", ""),
        cypher_block=cy_block,
        cypher_results=cy_res_block,
        response=row.get("response", "") or "(no answer)",
        rationale=row.get("rationale", "") or "(no rationale)",
    )


async def _score_once(
    client: openai.AsyncOpenAI, prompt: str, judge_model: str
) -> tuple[float, float, float]:
    resp = await client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,  # IBM's judge temperature
        max_tokens=120,
    )
    text = (resp.choices[0].message.content or "").strip()
    # Strip accidental markdown fences just in case.
    if text.startswith("```"):
        text = text.strip("`")
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1 :]
    obj = json.loads(text)
    return (
        float(obj.get("y1", 0.0)),
        float(obj.get("y2", 0.0)),
        float(obj.get("y3", 0.0)),
    )


async def _score_trajectory(
    client: openai.AsyncOpenAI,
    prompt: str,
    judge_model: str,
    trials: int,
) -> dict[str, Any]:
    runs: list[tuple[float, float, float]] = []
    for _ in range(trials):
        try:
            runs.append(await _score_once(client, prompt, judge_model))
        except Exception as exc:  # network/parse error → record and continue
            print(f"  judge error: {exc!r}", file=sys.stderr)
    if not runs:
        return {"y1": 0.0, "y2": 0.0, "y3": 0.0,
                "y1_std": 0.0, "y2_std": 0.0, "y3_std": 0.0,
                "trials": 0}
    y1s, y2s, y3s = zip(*runs)
    def mean_std(xs):
        m = statistics.fmean(xs)
        s = statistics.stdev(xs) if len(xs) > 1 else 0.0
        return m, s
    y1, y1_std = mean_std(y1s)
    y2, y2_std = mean_std(y2s)
    y3, y3_std = mean_std(y3s)
    return {"y1": y1, "y2": y2, "y3": y3,
            "y1_std": y1_std, "y2_std": y2_std, "y3_std": y3_std,
            "trials": len(runs)}


async def main_async(args) -> None:
    with open(args.input) as f:
        rows = json.load(f)
    if args.limit:
        rows = rows[: args.limit]

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("OPENAI_API_KEY not set.")
    client = openai.AsyncOpenAI(api_key=api_key)

    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(args.concurrency)
    results: list[dict[str, Any]] = []

    async def run_one(row):
        async with sem:
            prompt = _format_trajectory(row)
            scores = await _score_trajectory(
                client, prompt, args.judge_model, args.trials
            )
            rec = {
                "id": row.get("id"),
                "type": row.get("type"),
                "passed": row.get("passed"),
                "score": row.get("score"),
                **scores,
            }
            results.append(rec)
            print(
                f"  id={rec['id']:<6} type={rec['type']:<6} "
                f"y1={rec['y1']:.2f}±{rec['y1_std']:.2f}  "
                f"y2={rec['y2']:.2f}±{rec['y2_std']:.2f}  "
                f"y3={rec['y3']:.2f}±{rec['y3_std']:.2f}",
                flush=True,
            )

    await asyncio.gather(*[run_one(r) for r in rows])
    results.sort(key=lambda r: r.get("id") or 0)

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    json_path = args.output.replace(".csv", ".json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    by_type: dict[str, list[dict]] = {}
    for r in results:
        by_type.setdefault(r["type"], []).append(r)

    print("\n=== Aggregate ===")
    print(f"{'type':<10} {'n':>4} {'y1':>10} {'y2':>10} {'y3':>10}")
    for t in sorted(by_type) + ["__total__"]:
        rs = results if t == "__total__" else by_type[t]
        n = len(rs)
        if not n:
            continue
        y1 = statistics.fmean(r["y1"] for r in rs)
        y2 = statistics.fmean(r["y2"] for r in rs)
        y3 = statistics.fmean(r["y3"] for r in rs)
        print(f"{t:<10} {n:>4} {y1:>10.3f} {y2:>10.3f} {y3:>10.3f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="NLQ result JSON")
    p.add_argument("--output", required=True, help="Output CSV path")
    p.add_argument("--judge-model", default="gpt-4o")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--concurrency", type=int, default=5)
    p.add_argument("--limit", type=int, default=0,
                   help="Limit number of trajectories (0 = all)")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
