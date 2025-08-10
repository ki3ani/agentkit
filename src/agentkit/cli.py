"""Command Line Interface for AgentKit."""

import typer
import sys
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.panel import Panel

from .core.logger import get_logger
from .core.schema import load_and_validate_config, ConfigValidationError

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
    
    # Load and validate the agent configuration
    try:
        config_data = load_and_validate_config(config_path)
        
        if verbose or debug:
            logger.info("Configuration validation successful")
            logger.info(f"Agent name: {config_data['agent']['name']}")
            logger.info(f"Model: {config_data['agent']['model']}")
            logger.info(f"Tools: {config_data['agent'].get('tools', [])}")
        
    except ConfigValidationError as e:
        # Display rich error panel and exit
        e.display_error()
        sys.exit(1)
    except Exception as e:
        # Handle unexpected errors
        console.print(Panel.fit(
            f"[bold red]Unexpected Error[/bold red]\n\n"
            f"[red]Failed to load configuration: {str(e)}[/red]",
            title="❌ Error",
            border_style="red"
        ))
        sys.exit(1)
    
    # Agent execution placeholder - will be replaced in later prompts
    console.print(Panel.fit(
        f"[green]AgentKit MVP - Prompt 2 Schema Validation[/green]\n\n"
        f"[yellow]Config:[/yellow] {config_path} ✅\n"
        f"[yellow]Agent:[/yellow] {config_data['agent']['name']}\n"
        f"[yellow]Model:[/yellow] {config_data['agent']['model']}\n"
        f"[yellow]Tools:[/yellow] {', '.join(config_data['agent'].get('tools', ['None']))}\n"
        f"[yellow]Query:[/yellow] {input_query}\n"
        f"[yellow]Format:[/yellow] {output_format}\n\n"
        f"[dim]Real agent execution will be implemented in upcoming prompts...[/dim]",
        title="🤖 AgentKit",
        border_style="green"
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