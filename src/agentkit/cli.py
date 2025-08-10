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
from .core.tool_executor import ToolExecutor

app = typer.Typer(
    name="agentkit",
    help="AgentKit - A framework for creating and running AI agents defined through YAML configuration",
    no_args_is_help=True,
)

console = Console()
logger = get_logger(__name__)


def _format_tool_parameters(schema: Dict[str, Any]) -> str:
    """Format tool parameters schema for display.
    
    Args:
        schema: JSON schema for tool parameters
        
    Returns:
        Formatted parameter description
    """
    if not schema or not schema.get("properties"):
        return "  No parameters required"
    
    params = []
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    
    for param_name, param_schema in properties.items():
        param_type = param_schema.get("type", "any")
        param_desc = param_schema.get("description", "")
        is_required = param_name in required
        
        required_text = "[red]*[/red]" if is_required else "[dim](optional)[/dim]"
        param_line = f"  • [cyan]{param_name}[/cyan] ({param_type}) {required_text}"
        
        if param_desc:
            param_line += f": {param_desc}"
        
        # Add enum values if available
        if "enum" in param_schema:
            enum_values = ", ".join(str(v) for v in param_schema["enum"])
            param_line += f" [dim](values: {enum_values})[/dim]"
        
        params.append(param_line)
    
    return "\n".join(params)


@app.command("run")
def run_agent(
    config_path: str = typer.Argument(..., help="Path to the agent YAML configuration file"),
    input_query: str = typer.Option(..., "--input", "-i", help="Input query for the agent"),
    output_format: str = typer.Option("text", "--format", "-f", help="Output format: text or json"),
    max_tokens: int = typer.Option(1024, "--max-tokens", help="Maximum tokens to generate"),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="Override model provider (anthropic or bedrock)"),
    region: Optional[str] = typer.Option(None, "--region", help="AWS region for bedrock provider"),
    tools: Optional[str] = typer.Option(None, "--tools", help="Override agent tools (comma-separated list)"),
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
        tools_config = agent_config.get('tools', [])
        
        # Override tools if specified via CLI
        if tools:
            from .tools import has_tool
            tool_names = [name.strip() for name in tools.split(',')]
            
            # Validate that all specified tools exist
            invalid_tools = [name for name in tool_names if not has_tool(name)]
            if invalid_tools:
                console.print(Panel.fit(
                    f"[bold red]Invalid Tools[/bold red]\n\n"
                    f"[red]Unknown tools: {', '.join(invalid_tools)}[/red]\n\n"
                    f"[yellow]Use 'agentkit list-tools' to see available tools.[/yellow]",
                    title="❌ Tool Error",
                    border_style="red"
                ))
                sys.exit(1)
            
            # Override tools configuration
            tools_config = [{"name": name, "parameters": {}} for name in tool_names]
            
            if verbose or debug:
                logger.info(f"Overriding tools from CLI: {tool_names}")
        
        # Create tool executor
        tool_executor = ToolExecutor(tools_config)
        
        # Add tools context to system prompt if tools are available
        if tools_config:
            tools_context = tool_executor.get_tools_context()
            system_prompt += f"\n\n{tools_context}"
        
        # Get provider and region from CLI or config
        provider_name = provider or agent_config.get('provider', 'anthropic')
        region_name = region or agent_config.get('region', 'us-east-1')
        
        # Create model provider
        if verbose or debug:
            logger.info(f"Creating model provider for {model_name} using {provider_name}")
            if provider_name == 'bedrock':
                logger.info(f"Using AWS region: {region_name}")
            if tools_config:
                tool_names = [t['name'] for t in tools_config]
                logger.info(f"Available tools: {tool_names}")
        
        model_provider = get_model_provider(model_name, provider_name, region_name)
        
        # Display generation start info
        tool_names = [t['name'] for t in tools_config] if tools_config else ['None']
        provider_info = f"{provider_name}"
        if provider_name == 'bedrock':
            provider_info += f" ({region_name})"
            
        console.print(Panel.fit(
            f"[cyan]Generating response...[/cyan]\n\n"
            f"[yellow]Agent:[/yellow] {agent_config['name']}\n"
            f"[yellow]Model:[/yellow] {model_name}\n"
            f"[yellow]Provider:[/yellow] {provider_info}\n"
            f"[yellow]Tools:[/yellow] {', '.join(tool_names)}\n"
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
        
        # Process tool calls if tools are available
        tool_results = []
        if tools_config:
            response, tool_results = tool_executor.process_agent_response(response)
            
            if tool_results and (verbose or debug):
                for i, tool_result in enumerate(tool_results):
                    logger.info(f"Tool call {i+1}: {tool_result['tool_call']['name']} -> Success: {tool_result['result']['success']}")
        
        # Format and display the response
        if output_format.lower() == "json":
            output_data = {
                "agent": {
                    "name": agent_config['name'],
                    "model": model_name,
                    "tools": [t['name'] for t in tools_config] if tools_config else []
                },
                "query": input_query,
                "response": response,
                "tool_results": tool_results,
                "metadata": {
                    "max_tokens": max_tokens,
                    "config_file": config_path,
                    "tools_used": len(tool_results)
                }
            }
            console.print(json.dumps(output_data, indent=2))
        else:
            # Text format with rich formatting
            console.print()
            
            # Show tool usage summary if tools were used
            if tool_results:
                tool_summary = []
                for result in tool_results:
                    tool_name = result['tool_call']['name']
                    success = "✅" if result['result']['success'] else "❌"
                    tool_summary.append(f"{success} {tool_name}")
                
                console.print(Panel.fit(
                    f"[blue]Tools Used:[/blue] {' | '.join(tool_summary)}",
                    title="🔧 Tool Execution",
                    border_style="blue"
                ))
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


@app.command("list-tools")
def list_available_tools(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed tool information")
) -> None:
    """List all available tools with their descriptions."""
    from .tools import get_global_registry
    
    registry = get_global_registry()
    tools_info = registry.get_all_tool_info()
    
    if not tools_info:
        console.print("[yellow]No tools are currently available.[/yellow]")
        return
    
    if verbose:
        # Detailed view
        for tool_name, info in sorted(tools_info.items()):
            console.print(Panel.fit(
                f"[bold cyan]{info['name']}[/bold cyan]\n\n"
                f"[yellow]Description:[/yellow] {info['description']}\n\n"
                f"[yellow]Parameters:[/yellow]\n{_format_tool_parameters(info.get('parameters_schema', {}))}\n",
                title=f"🔧 {tool_name}",
                border_style="cyan"
            ))
            console.print()
    else:
        # Simple list view
        console.print(Panel.fit(
            "[bold green]Available Tools[/bold green]\n\n" +
            "\n".join([f"• [cyan]{name}[/cyan]: {info['description']}" 
                      for name, info in sorted(tools_info.items())]),
            title="🔧 AgentKit Tools",
            border_style="green"
        ))
        
        console.print(f"\n[dim]Use --verbose for detailed parameter information[/dim]")


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