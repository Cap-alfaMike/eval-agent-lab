"""Pipeline engine: dataset → LLM/agent → evaluation → report."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from eval_agent_lab.agents import AgentTrace, ReActAgent
from eval_agent_lab.config import AppConfig, LLMProviderType
from eval_agent_lab.datasets import DatasetItem, DatasetLoader
from eval_agent_lab.evals import EvaluationEngine, EvaluationReport
from eval_agent_lab.evals.metrics import get_default_metrics
from eval_agent_lab.evals.rubric import RubricConfig
from eval_agent_lab.llm import BaseLLMProvider, LLMMessage
from eval_agent_lab.mcp import ToolRegistry
from eval_agent_lab.mcp.tools import register_default_tools
from eval_agent_lab.observability import CostTracker, TraceLogger, get_logger

console = Console(highlight=False)
logger = get_logger("pipeline")


class PipelineMode:
    LLM_ONLY = "llm_only"
    AGENT = "agent"


class Pipeline:
    """End-to-end evaluation pipeline.

    Orchestrates: dataset loading → inference (LLM or agent) → evaluation → reporting.
    """

    def __init__(
        self,
        config: AppConfig,
        mode: str = PipelineMode.AGENT,
        llm: BaseLLMProvider | None = None,
        tools: ToolRegistry | None = None,
        eval_engine: EvaluationEngine | None = None,
        rubric: RubricConfig | None = None,
    ):
        self.config = config
        self.mode = mode
        self.rubric = rubric or RubricConfig.balanced()
        self.cost_tracker = CostTracker()
        self.trace_logger = TraceLogger(output_dir=config.pipeline.output_dir / "traces")

        # Initialize LLM
        if llm:
            self.llm = llm
        else:
            self.llm = self._create_llm()

        # Initialize tools
        self.tools = tools or ToolRegistry()
        if len(self.tools) == 0:
            register_default_tools(self.tools)

        # Initialize evaluation engine
        self.eval_engine = eval_engine or EvaluationEngine(
            metrics=get_default_metrics(),
            judge_llm=self.llm,
            use_judge=False,  # Disable judge by default for speed
            rubric=self.rubric,
        )

    def _create_llm(self) -> BaseLLMProvider:
        """Create LLM provider based on config."""
        if self.config.llm.provider == LLMProviderType.OPENAI:
            from eval_agent_lab.llm.openai_provider import OpenAIProvider

            return OpenAIProvider(self.config.llm, self.config.cache)
        elif self.config.llm.provider == LLMProviderType.HUGGINGFACE:
            from eval_agent_lab.llm.huggingface_provider import HuggingFaceProvider

            return HuggingFaceProvider(self.config.llm, self.config.cache)
        else:
            raise ValueError(f"Unknown provider: {self.config.llm.provider}")

    async def run(self, dataset_path: str, output_dir: str | None = None) -> EvaluationReport:
        """Run the full evaluation pipeline."""
        run_id = str(uuid.uuid4())[:8]
        out_dir = Path(output_dir) if output_dir else self.config.pipeline.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        header = f"\n[bold cyan]== EvalAgentLab Pipeline ==[/bold cyan] [dim](run: {run_id})[/dim]"
        console.print(header)
        console.print(f"   Mode: [yellow]{self.mode}[/yellow]")
        console.print(f"   Model: [yellow]{self.config.llm.model}[/yellow]")
        console.print(f"   Rubric: [yellow]{self.rubric.name}[/yellow]")
        console.print(f"   Dataset: [yellow]{dataset_path}[/yellow]\n")

        # Step 1: Load dataset
        console.print("[bold]>> Loading dataset...[/bold]")
        dataset = DatasetLoader.from_json(dataset_path)
        warnings = DatasetLoader.validate(dataset)
        if warnings:
            for w in warnings:
                console.print(f"  [yellow][!] {w}[/yellow]")
        console.print(f"  Loaded {dataset.size} items from '{dataset.name}'")

        # Step 2: Run inference
        console.print(f"\n[bold]>> Running inference ({self.mode})...[/bold]")
        results: list[dict[str, Any]] = []
        traces: list[AgentTrace] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Processing items...", total=dataset.size)

            for item in dataset.items:
                try:
                    if self.mode == PipelineMode.AGENT:
                        output, trace = await self._run_agent(item)
                        traces.append(trace)
                        self.trace_logger.log_trace(trace)
                    else:
                        output = await self._run_llm(item)
                        trace = None

                    results.append(
                        {
                            "id": item.id,
                            "input": item.input,
                            "expected_output": item.expected_output,
                            "actual_output": output,
                            "expected_tools": item.expected_tools,
                            "context": item.context,
                            "trace": trace,
                            # Behavioural fields
                            "acceptable_outputs": item.acceptable_outputs,
                            "tool_strategy": item.tool_strategy,
                            "max_steps": item.max_steps,
                            "penalize_overuse": item.penalize_overuse,
                            "expected_contains": item.expected_contains,
                        }
                    )
                except Exception as exc:
                    logger.error("item_failed", item_id=item.id, error=str(exc))
                    results.append(
                        {
                            "id": item.id,
                            "input": item.input,
                            "expected_output": item.expected_output,
                            "actual_output": f"ERROR: {exc}",
                            "expected_tools": item.expected_tools,
                            "context": item.context,
                            "trace": None,
                            "acceptable_outputs": item.acceptable_outputs,
                            "tool_strategy": item.tool_strategy,
                            "max_steps": item.max_steps,
                            "penalize_overuse": item.penalize_overuse,
                            "expected_contains": item.expected_contains,
                        }
                    )

                progress.advance(task)

        # Step 3: Evaluate
        console.print("\n[bold]>> Running evaluation...[/bold]")
        report = EvaluationReport(
            report_id=run_id,
            dataset_name=dataset.name,
            model=self.config.llm.model,
        )
        start = time.perf_counter()

        for item_data in results:
            trace = item_data.pop("trace", None)
            eval_result = await self.eval_engine.evaluate_item(
                input_text=item_data["input"],
                expected_output=item_data["expected_output"],
                actual_output=item_data["actual_output"],
                item_id=item_data["id"],
                trace=trace,
                expected_tools=item_data.get("expected_tools"),
                context=item_data.get("context"),
                acceptable_outputs=item_data.get("acceptable_outputs"),
                tool_strategy=item_data.get("tool_strategy", "optional"),
                max_steps=item_data.get("max_steps", 10),
                penalize_overuse=item_data.get("penalize_overuse", False),
                expected_contains=item_data.get("expected_contains"),
            )
            report.results.append(eval_result)

        # Aggregate
        report.total_items = len(report.results)
        pass_threshold = self.rubric.pass_threshold
        report.successful_items = sum(
            1 for r in report.results if r.composite_score >= pass_threshold
        )
        report.failed_items = report.total_items - report.successful_items
        report.total_time_ms = round((time.perf_counter() - start) * 1000, 2)

        # Aggregate metrics
        metric_scores: dict[str, list[float]] = {}
        for result in report.results:
            for name, mr in result.metrics.items():
                if mr.error is None:
                    metric_scores.setdefault(name, []).append(mr.score)

        for name, scores in metric_scores.items():
            report.aggregate_metrics[f"avg_{name}"] = round(sum(scores) / len(scores), 4)

        composite_scores = [r.composite_score for r in report.results]
        if composite_scores:
            report.aggregate_metrics["avg_composite"] = round(
                sum(composite_scores) / len(composite_scores), 4
            )

        # Add cost data
        report.metadata["cost"] = self.cost_tracker.summary()
        report.metadata["llm_stats"] = self.llm.stats.model_dump()
        report.metadata["rubric"] = self.rubric.model_dump()

        # Step 4: Display results
        self._display_results(report)

        # Step 5: Save report
        report_path = out_dir / f"report_{run_id}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(), f, indent=2, default=str)
        console.print(f"\n[bold green][OK] Report saved to {report_path}[/bold green]\n")

        return report

    async def _run_agent(self, item: DatasetItem) -> tuple[str, AgentTrace]:
        """Run an agent on a single item."""
        agent = ReActAgent(
            llm=self.llm,
            tools=self.tools,
            max_steps=self.config.pipeline.batch_size,
        )
        trace = await agent.run(item.input, context=item.context)
        return trace.final_answer, trace

    async def _run_llm(self, item: DatasetItem) -> str:
        """Run direct LLM inference on a single item."""
        messages = [LLMMessage(role="user", content=item.input)]
        if item.context:
            messages.insert(0, LLMMessage(role="system", content=f"Context: {item.context}"))
        response = await self.llm.generate(messages)
        return response.content

    def _display_results(self, report: EvaluationReport) -> None:
        """Display evaluation results in a rich table."""
        console.print("\n[bold]━━━ Evaluation Results ━━━[/bold]\n")

        # Summary table
        summary_table = Table(title="Summary", show_header=True, header_style="bold magenta")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")

        summary_table.add_row("Dataset", report.dataset_name)
        summary_table.add_row("Model", report.model)
        summary_table.add_row("Total Items", str(report.total_items))
        summary_table.add_row("Passed (≥0.5)", str(report.successful_items))
        summary_table.add_row("Failed (<0.5)", str(report.failed_items))
        success_rate = report.successful_items / max(report.total_items, 1)
        summary_table.add_row("Success Rate", f"{success_rate:.1%}")
        console.print(summary_table)

        # Metrics table
        if report.aggregate_metrics:
            metrics_table = Table(
                title="\nAggregate Metrics", show_header=True, header_style="bold magenta"
            )
            metrics_table.add_column("Metric", style="cyan")
            metrics_table.add_column("Score", style="green")

            for name, score in sorted(report.aggregate_metrics.items()):
                color = "green" if score >= 0.7 else "yellow" if score >= 0.4 else "red"
                metrics_table.add_row(name, f"[{color}]{score:.4f}[/{color}]")
            console.print(metrics_table)

        # Per-item results
        items_table = Table(
            title="\nPer-Item Results", show_header=True, header_style="bold magenta"
        )
        items_table.add_column("ID", style="cyan", width=15)
        items_table.add_column("Composite", style="green", width=10)
        items_table.add_column("Status", width=8)
        items_table.add_column("Input (truncated)", width=40)

        for result in report.results:
            status = "[green]✓[/green]" if result.composite_score >= 0.5 else "[red]✗[/red]"
            if result.composite_score >= 0.7:
                score_color = "green"
            elif result.composite_score >= 0.4:
                score_color = "yellow"
            else:
                score_color = "red"
            items_table.add_row(
                result.item_id,
                f"[{score_color}]{result.composite_score:.4f}[/{score_color}]",
                status,
                result.input_text[:40] + ("..." if len(result.input_text) > 40 else ""),
            )
        console.print(items_table)
