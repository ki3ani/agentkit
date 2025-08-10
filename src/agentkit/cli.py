"""Command Line Interface for AgentKit."""

import typer
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .core.logger import get_logger
from .core.schema import load_and_validate_config, ConfigValidationError
from .core.model_interface import get_model_provider, ModelError

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
    max_tokens: int = typer.Option(1024, "--max-tokens", help="Maximum tokens to generate"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug logging"),
) -> None:
    """Run an agent with the specified configuration and input query."""
    
    if verbose or debug:
        logger.info(f"Running agent with config: {config_path}")
        logger.info(f"Input query: {input_query}")
        logger.info(f"Output format: {output_format}")
        logger.info(f"Max tokens: {max_tokens}")
    
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
    
    # Create model provider and generate response
    try:
        agent_config = config_data['agent']
        model_name = agent_config['model']
        system_prompt = agent_config['prompts']['system']
        task_prompt = agent_config['prompts']['task'] + "\n\nUser query: " + input_query
        
        # Create model provider
        if verbose or debug:
            logger.info(f"Creating model provider for {model_name}")
        
        model_provider = get_model_provider(model_name)
        
        # Display generation start info
        console.print(Panel.fit(
            f"[cyan]Generating response...[/cyan]\n\n"
            f"[yellow]Agent:[/yellow] {agent_config['name']}\n"
            f"[yellow]Model:[/yellow] {model_name}\n"
            f"[yellow]Tools:[/yellow] {', '.join(agent_config.get('tools', ['None']))}\n"
            f"[yellow]Max Tokens:[/yellow] {max_tokens}",
            title="🤖 AgentKit - Processing",
            border_style="cyan"
        ))
        
        # Generate response
        response = model_provider.generate(
            system_prompt=system_prompt,
            task_prompt=task_prompt,
            max_tokens=max_tokens
        )
        
        # Format and display the response
        if output_format.lower() == "json":
            output_data = {
                "agent": {
                    "name": agent_config['name'],
                    "model": model_name,
                    "tools": agent_config.get('tools', [])
                },
                "query": input_query,
                "response": response,
                "metadata": {
                    "max_tokens": max_tokens,
                    "config_file": config_path
                }
            }
            console.print(json.dumps(output_data, indent=2))
        else:
            # Text format with rich formatting
            console.print()
            console.print(Panel.fit(
                f"[bold green]Response from {agent_config['name']}[/bold green]\n\n"
                f"{response}",
                title="💭 Agent Response",
                border_style="green"
            ))
        
        if verbose or debug:
            logger.info("Agent response generated successfully")
        
    except ModelError as e:
        # Display model-specific error
        console.print(Panel.fit(
            f"[bold red]Model Error[/bold red]\n\n"
            f"[red]{e.message}[/red]",
            title="❌ Model Error",
            border_style="red"
        ))
        sys.exit(1)
    except Exception as e:
        # Handle unexpected errors during generation
        console.print(Panel.fit(
            f"[bold red]Unexpected Error[/bold red]\n\n"
            f"[red]Failed to generate response: {str(e)}[/red]",
            title="❌ Error",
            border_style="red"
        ))
        if debug:
            import traceback
            console.print(f"\n[dim]Debug traceback:[/dim]\n{traceback.format_exc()}")
        sys.exit(1)


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