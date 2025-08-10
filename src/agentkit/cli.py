"""Command Line Interface for AgentKit."""

import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.panel import Panel

from .core.logger import get_logger

app = typer.Typer(
    name="agentkit",
    help="AgentKit - A framework for creating and running AI agents defined through YAML configuration",
    no_args_is_help=True,
)

console = Console()
logger = get_logger(__name__)


@app.command("run")
def run_agent(
    config_path: str = typer.Argument(..., help="Path to the agent YAML configuration file"),
    input_query: str = typer.Option(..., "--input", "-i", help="Input query for the agent"),
    output_format: str = typer.Option("text", "--format", "-f", help="Output format: text or json"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug logging"),
) -> None:
    """Run an agent with the specified configuration and input query."""
    
    if verbose or debug:
        logger.info(f"Running agent with config: {config_path}")
        logger.info(f"Input query: {input_query}")
        logger.info(f"Output format: {output_format}")
    
    # Convert string path to Path object
    config_file = Path(config_path)
    
    # Placeholder implementation - will be replaced in later prompts
    console.print(Panel.fit(
        f"[green]AgentKit MVP - Prompt 1 Scaffold[/green]\n\n"
        f"[yellow]Config:[/yellow] {config_file}\n"
        f"[yellow]Query:[/yellow] {input_query}\n"
        f"[yellow]Format:[/yellow] {output_format}\n\n"
        f"[dim]Real agent execution will be implemented in upcoming prompts...[/dim]",
        title="🤖 AgentKit",
        border_style="blue"
    ))


@app.command("version")
def show_version() -> None:
    """Show AgentKit version information."""
    from . import __version__
    console.print(f"AgentKit v{__version__}")


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()