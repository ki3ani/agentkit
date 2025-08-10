# 🤖 AgentKit

**AgentKit** is an open-source framework for creating and running AI agents defined entirely through YAML configuration files. Build powerful AI agents without writing code - just define your agent's behavior, tools, and prompts in YAML and let AgentKit handle the rest.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Poetry](https://img.shields.io/badge/packaging-poetry-blue.svg)](https://python-poetry.org/)

## 🌟 Features

- **YAML-First Configuration**: Define agents entirely in YAML - no coding required
- **Multiple AI Models**: Support for Claude 3 (Opus, Sonnet, Haiku) with more models coming soon
- **Robust Validation**: Comprehensive schema validation with helpful error messages
- **Cloud-Ready**: Built for AWS Lambda deployment with local development support
- **Rich CLI Interface**: Beautiful command-line interface with structured output
- **Extensible Architecture**: Clean abstractions for adding new models and tools
- **Production-Ready**: Comprehensive error handling, logging, and testing

## 🚀 Quick Start

### Installation

```bash
# Install with Poetry (recommended)
git clone https://github.com/agentkit-team/agentkit.git
cd agentkit
poetry install

# Or install with pip
pip install agentkit
```

### Configuration

1. **Set up your API keys** by copying the example environment file:
   ```bash
   cp .env.example .env
   ```

2. **Add your Anthropic API key** to `.env`:
   ```env
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```

### Create Your First Agent

Create a YAML file defining your agent (e.g., `my-agent.yaml`):

```yaml
agent:
  name: "helpful-assistant"
  model: "claude-3-sonnet"
  tools: []
  prompts:
    system: "You are a helpful AI assistant who provides clear, accurate information."
    task: "Answer the user's question thoroughly and helpfully."
  metadata:
    version: "1.0"
    description: "A general-purpose helpful assistant"
```

### Run Your Agent

```bash
# Run with Poetry
poetry run agentkit run my-agent.yaml --input "What is machine learning?"

# Or if installed with pip
agentkit run my-agent.yaml --input "What is machine learning?"
```

## 📋 YAML Configuration Reference

### Required Fields

```yaml
agent:
  name: "agent-name"          # String: Name of your agent
  model: "claude-3-sonnet"    # String: AI model to use
  prompts:
    system: "System prompt"   # String: Instructions for the AI
    task: "Task description"  # String: What the agent should do
```

### Optional Fields

```yaml
agent:
  tools: ["web_search"]       # Array: Tools available to agent (coming soon)
  metadata:                   # Object: Additional metadata
    version: "1.0"
    description: "Agent description"
    author: "Your Name"
```

### Supported Models

- `claude-3-opus` - Most capable, best for complex tasks
- `claude-3-sonnet` - Balanced performance and cost
- `claude-3-haiku` - Fastest, best for simple tasks

## 🔧 CLI Reference

### Basic Usage

```bash
agentkit run <config.yaml> --input "Your query"
```

### Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--input` | `-i` | Input query for the agent | Required |
| `--format` | `-f` | Output format: `text` or `json` | `text` |
| `--max-tokens` | | Maximum tokens to generate | `1024` |
| `--verbose` | `-v` | Enable verbose logging | `false` |
| `--debug` | `-d` | Enable debug logging | `false` |

### Examples

```bash
# Basic usage
agentkit run my-agent.yaml --input "Explain quantum computing"

# JSON output for programmatic use
agentkit run my-agent.yaml --input "Hello" --format json

# Verbose logging for debugging
agentkit run my-agent.yaml --input "Test" --verbose

# Custom token limit
agentkit run my-agent.yaml --input "Write a story" --max-tokens 2000
```

## ⚙️ Configuration

### Local Development

For local development, set environment variables in `.env`:

```env
# Required: Anthropic API key
ANTHROPIC_API_KEY=your_api_key_here

# Optional: Logging configuration
LOG_LEVEL=INFO
LOG_FORMAT=json

# Optional: Model defaults
DEFAULT_MODEL=claude-3-sonnet
MAX_TOKENS=1024
TEMPERATURE=0.7
```

### AWS Lambda Deployment

For AWS deployment, AgentKit supports multiple authentication methods:

#### Method 1: Environment Variables
```env
ANTHROPIC_API_KEY=your_api_key_here
```

#### Method 2: AWS Secrets Manager
```env
ANTHROPIC_SECRET_ARN=arn:aws:secretsmanager:us-east-1:123456789012:secret:anthropic-api-key
```

The secret should contain JSON with your API key:
```json
{
  "anthropic_api_key": "your_api_key_here"
}
```

#### AWS Infrastructure Setup

1. **IAM Role**: Ensure your Lambda has permissions for Secrets Manager (if using)
2. **Environment Variables**: Set `ANTHROPIC_SECRET_ARN` if using Secrets Manager
3. **Dependencies**: Include the built package with all dependencies
4. **Timeout**: Set appropriate timeout for AI model calls (30s recommended)

## 🐳 Docker Support

### Run with Docker

```bash
# Build the image
docker build -t agentkit .

# Run with environment variables
docker run -e ANTHROPIC_API_KEY=your_key agentkit run my-agent.yaml --input "Hello"
```

### Docker Compose for Development

```bash
# Start development environment
docker-compose up -d

# Run commands in the container
docker-compose exec agentkit agentkit run my-agent.yaml --input "Test"
```

## 🧪 Development & Testing

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/agentkit-team/agentkit.git
cd agentkit

# Install with development dependencies
poetry install

# Install pre-commit hooks
poetry run pre-commit install
```

### Run Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src/agentkit

# Run specific test file
poetry run pytest tests/test_schema.py -v
```

### Code Quality

```bash
# Format code
poetry run black src tests

# Sort imports
poetry run isort src tests

# Lint code
poetry run flake8 src tests

# Type checking
poetry run mypy src
```

## 🏗️ Architecture

AgentKit follows a modular architecture:

```
src/agentkit/
├── core/
│   ├── config.py          # Configuration management
│   ├── logger.py          # Logging utilities
│   ├── schema.py          # YAML validation
│   └── model_interface.py # AI model abstraction
├── tools/                 # Agent tools (coming soon)
└── cli.py                 # Command-line interface
```

### Key Components

- **Schema Validation**: Strict YAML validation with helpful error messages
- **Model Interface**: Extensible abstraction for multiple AI providers
- **Cloud-Ready**: Designed for serverless deployment with local development
- **Error Handling**: Comprehensive error handling with user-friendly messages

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run tests and ensure they pass
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔮 Roadmap

- [ ] **Tool System**: Web search, file operations, API calls
- [ ] **More Models**: OpenAI GPT, Mistral, local models
- [ ] **Conversation Memory**: Multi-turn conversations
- [ ] **Agent Chaining**: Connect multiple agents
- [ ] **Web Interface**: GUI for non-technical users
- [ ] **Agent Marketplace**: Share and discover agents

## 💬 Support

- **Documentation**: [docs.agentkit.dev](https://docs.agentkit.dev)
- **Issues**: [GitHub Issues](https://github.com/agentkit-team/agentkit/issues)
- **Discussions**: [GitHub Discussions](https://github.com/agentkit-team/agentkit/discussions)

## 🙏 Acknowledgments

- [Anthropic](https://anthropic.com) for Claude API
- [Rich](https://github.com/Textualize/rich) for beautiful CLI output
- [Typer](https://github.com/tiangolo/typer) for CLI framework
- [Poetry](https://python-poetry.org/) for dependency management

---

Built with ❤️ by the AgentKit team