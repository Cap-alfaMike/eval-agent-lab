"""CLI interface for EvalAgentLab."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console

from eval_agent_lab import __version__
from eval_agent_lab.config import AppConfig, LLMConfig, LLMProviderType, LogLevel

app = typer.Typer(
    name="eval-agent-lab",
    help="EvalAgentLab: LLM Evaluation & Agent Reliability Framework",
    add_completion=False,
)
console = Console(highlight=False)


@app.command("run")
def run_pipeline(
    dataset: str = typer.Argument(..., help="Path to the evaluation dataset JSON file"),
    mode: str = typer.Option("agent", "--mode", "-m", help="Execution mode: 'agent' or 'llm_only'"),
    model: str = typer.Option("gpt-4o-mini", "--model", help="LLM model to use"),
    provider: str = typer.Option(
        "openai", "--provider", "-p",
        help="LLM provider: 'openai' or 'huggingface'",
    ),
    output_dir: str = typer.Option(
        "outputs", "--output", "-o",
        help="Output directory for reports",
    ),
    max_concurrent: int = typer.Option(5, "--concurrent", "-c", help="Max concurrent evaluations"),
    use_judge: bool = typer.Option(False, "--judge", help="Enable LLM-as-judge evaluation"),
    temperature: float = typer.Option(0.0, "--temperature", "-t", help="LLM temperature"),
    stream: bool = typer.Option(False, "--stream", help="Enable streaming LLM responses"),
    rubric: str | None = typer.Option(None, "--rubric", "-r", help="Path to rubric JSON file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    """Run an evaluation pipeline on a dataset."""
    from eval_agent_lab.config import ObservabilityConfig, PipelineConfig

    config = AppConfig(
        llm=LLMConfig(
            provider=LLMProviderType(provider),
            model=model,
            temperature=temperature,
            stream=stream,
        ),
        pipeline=PipelineConfig(max_concurrent=max_concurrent, output_dir=Path(output_dir)),
        observability=ObservabilityConfig(log_level=LogLevel("DEBUG" if verbose else "INFO")),
    )

    # Load rubric if provided
    rubric_config = None
    if rubric:
        from eval_agent_lab.evals.rubric import RubricConfig
        rubric_config = RubricConfig.from_json(rubric)
        console.print(f"  Rubric: [yellow]{rubric_config.name}[/yellow]")

    from eval_agent_lab.pipelines import Pipeline, PipelineMode

    pipeline_mode = PipelineMode.AGENT if mode == "agent" else PipelineMode.LLM_ONLY
    pipeline = Pipeline(config=config, mode=pipeline_mode, rubric=rubric_config)

    try:
        report = asyncio.run(pipeline.run(dataset, output_dir))
        raise typer.Exit(0 if report.successful_items > 0 else 1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Pipeline interrupted by user.[/yellow]")
        raise typer.Exit(130) from None
    except Exception as exc:
        console.print(f"\n[red]Pipeline failed: {exc}[/red]")
        raise typer.Exit(1) from None


@app.command("validate")
def validate_dataset(
    dataset: str = typer.Argument(..., help="Path to the dataset JSON file"),
) -> None:
    """Validate a dataset file."""
    from eval_agent_lab.datasets import DatasetLoader

    try:
        ds = DatasetLoader.from_json(dataset)
        warnings = DatasetLoader.validate(ds)
        console.print(f"[green][OK] Dataset '{ds.name}' is valid[/green]")
        console.print(f"   Items: {ds.size}")
        console.print(f"   Version: {ds.version}")
        if warnings:
            console.print(f"   [yellow]Warnings ({len(warnings)}):[/yellow]")
            for w in warnings:
                console.print(f"     [!] {w}")
    except Exception as exc:
        console.print(f"[red][FAIL] Validation failed: {exc}[/red]")
        raise typer.Exit(1) from None


@app.command("list-tools")
def list_tools() -> None:
    """List all registered MCP tools."""
    from rich.table import Table

    from eval_agent_lab.mcp import ToolRegistry
    from eval_agent_lab.mcp.tools import register_default_tools

    registry = ToolRegistry()
    register_default_tools(registry)

    table = Table(title="Registered MCP Tools", show_header=True, header_style="bold magenta")
    table.add_column("Name", style="cyan")
    table.add_column("Category", style="green")
    table.add_column("Description")
    table.add_column("Parameters")

    for tool_def in registry.list_tools():
        params = ", ".join(f"{p.name}:{p.type.value}" for p in tool_def.parameters)
        table.add_row(tool_def.name, tool_def.category, tool_def.description[:60], params)

    console.print(table)


@app.command("version")
def show_version() -> None:
    """Show the version of EvalAgentLab."""
    console.print(f"EvalAgentLab v{__version__}")


@app.command("compare")
def compare_runs_cmd(
    run_a: str = typer.Argument(..., help="Path to the baseline report JSON (Run A)"),
    run_b: str = typer.Argument(..., help="Path to the candidate report JSON (Run B)"),
    threshold: float = typer.Option(0.005, "--threshold", help="Min delta to classify as changed"),
    output: str | None = typer.Option(None, "--output", "-o", help="Save comparison JSON to path"),
) -> None:
    """Compare two evaluation runs (A/B testing)."""
    from eval_agent_lab.evals.comparison import compare_runs, display_comparison

    try:
        comparison = compare_runs(run_a, run_b, threshold=threshold)
        display_comparison(comparison, console)

        if output:
            import json
            out_path = Path(output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(comparison.model_dump(), f, indent=2)
            console.print(f"  [green][OK] Comparison saved to {out_path}[/green]")
    except Exception as exc:
        console.print(f"[red][FAIL] Comparison failed: {exc}[/red]")
        raise typer.Exit(1) from None


@app.command("push-dataset")
def push_dataset_cmd(
    dataset: str = typer.Argument(..., help="Path to the dataset JSON file"),
    repo: str = typer.Option(..., "--repo", help="HF Hub repo id (e.g. username/eval-benchmark)"),
    token: str | None = typer.Option(None, "--token", help="HF API token (or set HF_TOKEN env)"),
    private: bool = typer.Option(False, "--private", help="Create a private repository"),
) -> None:
    """Push an evaluation dataset to Hugging Face Hub."""
    from eval_agent_lab.datasets.export_hf import push_dataset_to_hf

    try:
        console.print(f"  Uploading [cyan]{dataset}[/cyan] to [cyan]{repo}[/cyan]...")
        url = push_dataset_to_hf(dataset, repo, token=token, private=private)
        console.print(f"  [green][OK] Published at {url}[/green]")
    except Exception as exc:
        console.print(f"[red][FAIL] Push failed: {exc}[/red]")
        raise typer.Exit(1) from None


@app.command("demo")
def run_demo() -> None:
    """Run a quick demo with built-in tools (no API key required)."""
    import asyncio

    from eval_agent_lab.mcp import ToolRegistry
    from eval_agent_lab.mcp.tools import register_default_tools

    async def _demo() -> None:
        registry = ToolRegistry()
        register_default_tools(registry)

        console.print("[bold cyan]== EvalAgentLab Demo ==[/bold cyan]\n")

        # Demo search
        console.print("[bold]1. Search Tool[/bold]")
        result = await registry.invoke("search", {"query": "transformer architecture"})
        console.print("   Query: 'transformer architecture'")
        console.print(f"   Result: {result.output}\n")

        # Demo calculator
        console.print("[bold]2. Calculator Tool[/bold]")
        result = await registry.invoke("calculator", {"expression": "sqrt(144) + 2 * pi"})
        console.print("   Expression: 'sqrt(144) + 2 * pi'")
        console.print(f"   Result: {result.output}\n")

        # Demo vector retrieval
        console.print("[bold]3. Vector Retrieval Tool[/bold]")
        result = await registry.invoke("vector_retrieval", {"query": "neural networks", "top_k": 2})
        console.print("   Query: 'neural networks'")
        console.print(f"   Result: {result.output}\n")

        # Demo evaluation
        console.print("[bold]4. Evaluation Metrics[/bold]")
        from eval_agent_lab.evals.metrics import (
            ContainsAnswerMetric,
            ExactMatchMetric,
            SemanticSimilarityMetric,
        )
        prediction = "Python is a programming language created by Guido van Rossum"
        reference = "Python"

        em = await ExactMatchMetric().compute(prediction, reference)
        ca = await ContainsAnswerMetric().compute(prediction, reference)
        ss = await SemanticSimilarityMetric().compute(prediction, reference)

        console.print(f"   Prediction: '{prediction}'")
        console.print(f"   Reference:  '{reference}'")
        console.print(f"   Exact Match:        {em.score}")
        console.print(f"   Contains Answer:    {ca.score}")
        console.print(f"   Semantic Similarity: {ss.score}")

        # Demo rubric
        console.print("\n[bold]5. Rubric System[/bold]")
        from eval_agent_lab.evals.rubric import RubricConfig
        rubric = RubricConfig.accuracy_focused()
        console.print(f"   Rubric: '{rubric.name}' ({len(rubric.metrics)} metrics)")
        console.print(f"   Pass threshold: {rubric.pass_threshold}")
        for mw in rubric.metrics[:4]:
            console.print(f"     {mw.name}: weight={mw.weight}")
        console.print("     ...")

        console.print("\n[bold green][OK] Demo complete![/bold green]")

    asyncio.run(_demo())


if __name__ == "__main__":
    app()

