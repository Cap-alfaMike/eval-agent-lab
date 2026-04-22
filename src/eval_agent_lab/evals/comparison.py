"""Run comparison: A/B testing between two evaluation reports.

Loads two report JSON files, aligns their metrics, computes deltas,
and classifies each metric as improved / regressed / unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

from eval_agent_lab.exceptions import EvaluationError


class ComparisonError(EvaluationError):
    """Failed to compare evaluation runs."""


class MetricDelta(BaseModel):
    """Delta for a single metric between two runs."""
    metric: str
    run_a: float
    run_b: float
    delta: float
    pct_change: float | None = None
    direction: str = ""  # "improved", "regressed", "unchanged"


class RunComparison(BaseModel):
    """Full comparison result between two evaluation runs."""
    run_a_id: str = ""
    run_b_id: str = ""
    run_a_model: str = ""
    run_b_model: str = ""
    run_a_dataset: str = ""
    run_b_dataset: str = ""
    metric_deltas: list[MetricDelta] = Field(default_factory=list)
    improved_count: int = 0
    regressed_count: int = 0
    unchanged_count: int = 0
    summary: str = ""


def _load_report(path: str | Path) -> dict[str, Any]:
    """Load a report JSON file."""
    path = Path(path)
    if not path.exists():
        raise ComparisonError(f"Report file not found: {path}")
    try:
        with open(path, encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
            return data
    except json.JSONDecodeError as exc:
        raise ComparisonError(f"Invalid JSON in report {path}: {exc}") from exc


def _classify_delta(delta: float, threshold: float = 0.005) -> str:
    """Classify a delta as improved / regressed / unchanged."""
    if delta > threshold:
        return "improved"
    elif delta < -threshold:
        return "regressed"
    return "unchanged"


def compare_runs(
    run_a_path: str | Path,
    run_b_path: str | Path,
    threshold: float = 0.005,
) -> RunComparison:
    """Compare two evaluation runs and produce a structured comparison.

    Args:
        run_a_path: Path to the baseline report JSON.
        run_b_path: Path to the candidate report JSON.
        threshold: Minimum absolute delta to count as improved/regressed.

    Returns:
        RunComparison with per-metric deltas and summary statistics.
    """
    report_a = _load_report(run_a_path)
    report_b = _load_report(run_b_path)

    metrics_a: dict[str, float] = report_a.get("aggregate_metrics", {})
    metrics_b: dict[str, float] = report_b.get("aggregate_metrics", {})

    # Union of all metric keys
    all_keys = sorted(set(metrics_a.keys()) | set(metrics_b.keys()))

    deltas: list[MetricDelta] = []
    improved = regressed = unchanged = 0

    for key in all_keys:
        val_a = metrics_a.get(key, 0.0)
        val_b = metrics_b.get(key, 0.0)
        d = round(val_b - val_a, 6)
        pct = round(d / val_a * 100, 2) if val_a != 0 else None
        direction = _classify_delta(d, threshold)

        deltas.append(MetricDelta(
            metric=key,
            run_a=round(val_a, 4),
            run_b=round(val_b, 4),
            delta=round(d, 4),
            pct_change=pct,
            direction=direction,
        ))

        if direction == "improved":
            improved += 1
        elif direction == "regressed":
            regressed += 1
        else:
            unchanged += 1

    # Also compare top-level success rates
    total_a = report_a.get("total_items", 0) or 1
    total_b = report_b.get("total_items", 0) or 1
    success_a = report_a.get("successful_items", 0) / total_a
    success_b = report_b.get("successful_items", 0) / total_b
    sr_delta = round(success_b - success_a, 4)
    sr_dir = _classify_delta(sr_delta, threshold)

    deltas.insert(0, MetricDelta(
        metric="success_rate",
        run_a=round(success_a, 4),
        run_b=round(success_b, 4),
        delta=sr_delta,
        pct_change=round(sr_delta / success_a * 100, 2) if success_a else None,
        direction=sr_dir,
    ))
    if sr_dir == "improved":
        improved += 1
    elif sr_dir == "regressed":
        regressed += 1
    else:
        unchanged += 1

    # Build summary
    if improved > regressed:
        verdict = "Run B is BETTER overall"
    elif regressed > improved:
        verdict = "Run A is BETTER overall"
    else:
        verdict = "Runs are roughly EQUIVALENT"

    summary = (
        f"{verdict}: {improved} improved, {regressed} regressed, "
        f"{unchanged} unchanged across {len(deltas)} metrics."
    )

    return RunComparison(
        run_a_id=report_a.get("report_id", ""),
        run_b_id=report_b.get("report_id", ""),
        run_a_model=report_a.get("model", ""),
        run_b_model=report_b.get("model", ""),
        run_a_dataset=report_a.get("dataset_name", ""),
        run_b_dataset=report_b.get("dataset_name", ""),
        metric_deltas=deltas,
        improved_count=improved,
        regressed_count=regressed,
        unchanged_count=unchanged,
        summary=summary,
    )


def display_comparison(comparison: RunComparison, console: Console | None = None) -> None:
    """Render a RunComparison as a rich terminal table."""
    console = console or Console(highlight=False)

    console.print("\n[bold]== Run Comparison ==[/bold]\n")
    console.print(f"  Run A: [cyan]{comparison.run_a_id}[/cyan] ({comparison.run_a_model})")
    console.print(f"  Run B: [cyan]{comparison.run_b_id}[/cyan] ({comparison.run_b_model})\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", width=30)
    table.add_column("Run A", justify="right", width=10)
    table.add_column("Run B", justify="right", width=10)
    table.add_column("Delta", justify="right", width=10)
    table.add_column("Direction", width=12)

    for md in comparison.metric_deltas:
        if md.direction == "improved":
            delta_str = f"[green]+{md.delta:.4f}[/green]"
            dir_str = "[green]improved[/green]"
        elif md.direction == "regressed":
            delta_str = f"[red]{md.delta:.4f}[/red]"
            dir_str = "[red]regressed[/red]"
        else:
            delta_str = f"[dim]{md.delta:.4f}[/dim]"
            dir_str = "[dim]unchanged[/dim]"

        table.add_row(
            md.metric,
            f"{md.run_a:.4f}",
            f"{md.run_b:.4f}",
            delta_str,
            dir_str,
        )

    console.print(table)
    console.print(f"\n  [bold]{comparison.summary}[/bold]\n")
