"""Narrated terminal demo: Industrial asset-operations failure-impact on Samyama.

Record with asciinema:
    asciinema rec --overwrite --cols 92 --rows 32 --idle-time-limit 2.0 \
        -c "bash -c 'source ~/projects/venv/bin/activate && PYTHONUNBUFFERED=1 python -m demo.demo'" \
        demo/assetops.cast

Loads the AssetOpsBench-derived industrial KG (ISO 14224 + ISA-95) into a
Samyama graph and walks through the question every maintenance team asks:
"what breaks if this asset fails, and where do we act first?"
"""

from __future__ import annotations

import time

from rich.console import Console
from rich.panel import Panel
from samyama import SamyamaClient

from etl.eamlite_loader import load_eamlite
from etl.couchdb_loader import load_couchdb
from etl.fmsr_loader import load_fmsr
from etl.workorder_loader import load_workorders

console = Console()
G = "industrial"
DATA = "data"


def pause(s: float = 1.4) -> None:
    time.sleep(s)


def step(title: str) -> None:
    console.print()
    console.rule(f"[bold cyan]{title}")
    pause(0.6)


def run(client, q, label):
    console.print(f"  [dim]cypher>[/dim] [yellow]{q}[/yellow]")
    rows = client.query(q, G).records
    one = len(rows) == 1 and len(rows[0]) == 1
    console.print(f"  [green]→[/green] {label}: [bold]{rows[0][0] if one else rows}[/bold]")
    pause()
    return rows


def main() -> None:
    console.print(Panel.fit(
        "[bold]Samyama · AssetOps Knowledge Graph[/bold]\n"
        "\"What breaks if this asset fails — and where do we act first?\"\n"
        "[dim]data: IBM AssetOpsBench · ISO 14224 + ISA-95[/dim]",
        border_style="cyan",
    ))
    pause(1.2)

    step("1 · Load the industrial asset graph into Samyama")
    client = SamyamaClient.embedded()
    for fn in (load_eamlite, load_couchdb, load_fmsr, load_workorders):
        fn(client, DATA, G)
    total = client.query("MATCH (n) RETURN count(n)", G).records[0][0]
    console.print(f"  [green]loaded[/green] {total} nodes "
                  "(Site → Location → Equipment → Sensor / FailureMode / WorkOrder)")
    run(client, "MATCH (e:Equipment) RETURN count(e) AS equipment", "equipment items tracked")

    step("2 · Which assets are most critical? (ISO 14224 criticality)")
    run(
        client,
        "MATCH (e:Equipment) "
        "RETURN e.name AS asset, e.criticality_score AS criticality "
        "ORDER BY criticality DESC LIMIT 5",
        "highest-criticality assets",
    )

    step("3 · Failure-impact: what goes down if Pump-CW-1 fails?")
    console.print("  [dim]walking DEPENDS_ON 1..3 hops upstream from the cooling-water pump…[/dim]")
    pause()
    run(
        client,
        "MATCH (e:Equipment {name: \"Pump-CW-1\"})<-[:DEPENDS_ON*1..3]-(d:Equipment) "
        "RETURN DISTINCT d.name AS impacted, d.criticality_score AS criticality "
        "ORDER BY criticality DESC",
        "downstream assets isolated by one pump failure",
    )

    step("4 · Triage: high-severity anomalies that already raised work orders")
    run(
        client,
        "MATCH (a:Anomaly)-[:TRIGGERED]->(w:WorkOrder)-[:FOR_EQUIPMENT]->(e:Equipment) "
        "WHERE a.severity = \"high\" AND w.priority < 2.0 "
        "RETURN e.name AS asset, w.description AS work_order "
        "ORDER BY asset LIMIT 5",
        "priority-1 work orders from high-severity anomalies",
    )

    console.print()
    console.print(Panel.fit(
        "[bold green]Failure impact, criticality and triage — one Cypher query each,\n"
        "zero LLM tokens.[/bold green] 137/139 AssetOpsBench scenarios pass at 63 ms,\n"
        "vs 65% for GPT-4 over flat docs. The win is the data model.",
        border_style="green",
    ))
    pause(1.5)


if __name__ == "__main__":
    main()
