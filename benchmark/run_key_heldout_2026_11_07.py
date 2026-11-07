"""nexus-bwct4.4: run ONLY the key-bearing held-out arm (8 nx_answer calls), under Sam's 2026-11-07 spend go.

Run from the assetops-kg worktree through benchmark/herb_sandbox.sh, like --mvv. Reuses the harness's own guards,
budget, cost ledger (label mvv, stage key_heldout, mvv_run RUN_ID), row sink and report; adds nothing to the scoring.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.getcwd())
from benchmark import herb_ladder as h  # noqa: E402

CAP_USD = 1.6          # Sam's go: about 1.5 USD
RUN_ID = "2026-11-07"  # the ledger's mvv_run key: the date of this one run (no MVV ran on it); a real --mvv on the
                       # same date would inherit this run's charges, so a later arm run takes its own date



def main() -> None:
    out = h._RESULTS_DIR / f"key-heldout-{RUN_ID}.json"
    if out.exists():
        # review: a re-run spends nothing (valid rows are kept) but would restamp the record's versions and its spend
        raise SystemExit(f"{out} exists: the record of this run is kept as written")
    bad = h.mvv_switch_violations(os.environ) + h.mvv_llm_violations(os.environ)
    if bad:
        raise SystemExit("refusing to spend: " + "; ".join(bad))
    key = h.load_heldout(h._GOLD_KEY_HELDOUT)
    sink = h.MvvRows(h._RESULTS_DIR / f"key-heldout-{RUN_ID}.jsonl")
    prior = sink.load() if sink.exists() else None
    budget = h.Budget(CAP_USD, spent=h.ledger_spent(RUN_ID))
    if budget.spent:
        print(f"resuming: {budget.spent:.4f} USD already charged to {RUN_ID}")
    lib = h.SandboxPlanLibrary()
    rows, meta = h.run_heldout_rows(key, CAP_USD, lib, label="key_heldout", budget=budget,
                                    ledger_label=h.MVV_LEDGER_LABEL, run_id=RUN_ID, sink=sink, prior=prior)
    # The report's issue_entered precondition: the final MVV (2026-11-06) entered 4 issue questions in each run.
    final = json.loads((h._RESULTS_DIR / "mvv-2026-11-06.json").read_text())
    report = h.heldout_report(rows, final["score"]["runs"], len(key))
    rec = {"bead": "nexus-bwct4.4", "step": "key_heldout", "run_id": RUN_ID,
           "timestamp_utc": datetime.now(timezone.utc).isoformat(), "cap_usd": CAP_USD,
           "nexus_revision": os.environ.get("HERB_NEXUS_REV"), "versions": h.service_versions(),
           "issue_entered_from": "mvv-2026-11-06.json", "report": report, "meta": meta, "rows": rows,
           "conditions": {"sandbox": os.environ.get("HERB_SANDBOX") == "1", "collection": h.CORPUS,
                          "resolution_switch": os.environ.get("NX_ANSWER_RESOLUTION_REPAIR"),
                          "plan_service_url": getattr(lib, "service_url", None)}}
    out.write_text(json.dumps(rec, indent=2))
    print(json.dumps({"report": report, "spend": meta}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
