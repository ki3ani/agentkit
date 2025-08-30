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

### Model Providers

AgentKit supports multiple providers for AI models:

**Anthropic API (Default)**
- Direct access to Anthropic's Claude models
- Requires `ANTHROPIC_API_KEY` environment variable
- Best for direct Claude API usage

**AWS Bedrock**
- Access Claude models through AWS Bedrock
- Uses AWS credentials and IAM permissions
- Better for enterprise deployments

**Goose (Multi-Model Orchestration)**
- Route requests to multiple LLM providers through a single interface
- Model agnostic - supports GPT, Claude, Llama, and other models
- Requires `GOOSE_API_KEY` environment variable
- Perfect for multi-model applications and A/B testing

### Supported Models

**Anthropic & Bedrock Providers:**
- `claude-3-opus` - Most capable, best for complex tasks
- `claude-3-sonnet` - Balanced performance and cost  
- `claude-3-haiku` - Fastest, best for simple tasks

**Goose Provider (Model Agnostic):**
- `gpt-4`, `gpt-4-turbo`, `gpt-4o`, `gpt-4o-mini` - OpenAI models
- `claude-3-opus`, `claude-3-sonnet`, `claude-3-haiku` - Anthropic models
- `gemini-pro` - Google models
- `llama-2-70b`, `mixtral-8x7b` - Open source models
- Any custom model supported by your Goose deployment

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
| `--provider` | `-p` | Model provider: `anthropic`, `bedrock`, or `goose` | From config |
| `--model` | `-m` | Override model name | From config |
| `--region` | | AWS region for bedrock provider | `us-east-1` |
| `--tools` | | Override agent tools (comma-separated) | From config |
| `--max-tokens` | | Maximum tokens to generate | `1024` |
| `--verbose` | `-v` | Enable verbose logging | `false` |
| `--debug` | `-d` | Enable debug logging | `false` |

### Examples

```bash
# Basic usage
agentkit run my-agent.yaml --input "Explain quantum computing"

# Use AWS Bedrock provider
agentkit run my-agent.yaml --provider bedrock --input "Hello world"

# Bedrock with custom region
agentkit run my-agent.yaml --provider bedrock --region eu-west-1 --input "Hello"

# Use Goose provider with GPT-4
agentkit run my-agent.yaml --provider goose --model gpt-4 --input "Hello world"

# Switch between models with Goose
agentkit run my-agent.yaml --provider goose --model claude-3-opus --input "Complex reasoning task"
agentkit run my-agent.yaml --provider goose --model gpt-4o-mini --input "Simple question"

# Override tools
agentkit run my-agent.yaml --tools "echo,calculator" --input "Calculate 2+2"

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

### AWS Bedrock Setup

AgentKit supports AWS Bedrock for enterprise deployments with better security and compliance.

#### Prerequisites

1. **AWS Account** with Bedrock access
2. **Model Access** - Request access to Claude models in AWS Bedrock console
3. **AWS Credentials** configured (CLI, IAM role, or environment variables)

#### Installation with Bedrock Support

```bash
# Install with AWS dependencies
poetry install --extras aws

# Or with pip
pip install agentkit[aws]
```

#### Authentication Methods

**Method 1: AWS CLI Profile**
```bash
aws configure
# Follow prompts to set up your credentials
```

**Method 2: Environment Variables**
```env
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1
```

**Method 3: IAM Role (Lambda/EC2)**
- No configuration needed - uses attached IAM role automatically

#### Required IAM Permissions

Your AWS credentials need these permissions:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:ListFoundationModels"
            ],
            "Resource": [
                "arn:aws:bedrock:*::foundation-model/anthropic.claude-*"
            ]
        }
    ]
}
```

#### Bedrock Configuration

**YAML Configuration with Bedrock:**
```yaml
agent:
  name: "bedrock-assistant" 
  provider: "bedrock"          # Use AWS Bedrock
  model: "claude-3-sonnet"
  region: "us-east-1"          # AWS region
  prompts:
    system: "You are a helpful assistant."
    task: "Help the user with their query."
```

**CLI Override:**
```bash
# Override provider via CLI
agentkit run my-agent.yaml --provider bedrock --region us-west-2 --input "Hello"
```

### AWS Lambda Deployment

For AWS deployment, AgentKit supports multiple authentication methods:

#### Method 1: Environment Variables (Anthropic)
```env
ANTHROPIC_API_KEY=your_api_key_here
```

#### Method 2: AWS Secrets Manager (Anthropic)
```env
ANTHROPIC_SECRET_ARN=arn:aws:secretsmanager:us-east-1:123456789012:secret:anthropic-api-key
```

The secret should contain JSON with your API key:
```json
{
  "anthropic_api_key": "your_api_key_here"
}
```

#### Method 3: AWS Bedrock (Recommended for Enterprise)
No API keys needed - uses IAM role permissions.

### Goose Multi-Model Setup

AgentKit supports Goose for multi-model orchestration, allowing you to route requests to different LLM providers through a single interface.

#### Prerequisites

1. **Goose API Access** - Get API key from your Goose deployment
2. **Model Access** - Ensure your Goose instance has access to desired models

#### Authentication

**Environment Variable:**
```env
GOOSE_API_KEY=your_goose_api_key_here
GOOSE_BASE_URL=https://your-goose-instance.com/v1  # Optional, defaults to https://api.goose.ai/v1
```

#### Goose Configuration

**YAML Configuration with Goose:**
```yaml
agent:
  name: "multi-model-assistant"
  provider: "goose"
  model: "gpt-4o"  # Any model supported by your Goose instance
  tools: ["calculator", "text_count"]
  prompts:
    system: "You are a helpful AI assistant with multi-model capabilities."
    task: "Answer questions and solve tasks accurately using available tools."
```

**Advanced Multi-Model Examples:**
```yaml
# GPT-4 for complex reasoning
agent:
  name: "reasoning-agent"
  provider: "goose"
  model: "gpt-4-turbo"
  prompts:
    system: "You excel at complex reasoning and analysis."
    task: "Provide detailed analysis and reasoning for user queries."
---
# Claude for creative writing
agent:
  name: "creative-agent"
  provider: "goose"
  model: "claude-3-opus"
  prompts:
    system: "You are a creative writing assistant."
    task: "Help users with creative writing and storytelling."
---
# Llama for cost-effective tasks
agent:
  name: "efficient-agent"
  provider: "goose"  
  model: "llama-2-70b"
  prompts:
    system: "You provide helpful responses efficiently."
    task: "Answer user questions concisely and accurately."
```

**CLI Model Switching:**
```bash
# A/B test different models for the same task
agentkit run my-agent.yaml --provider goose --model gpt-4 --input "Explain quantum computing"
agentkit run my-agent.yaml --provider goose --model claude-3-opus --input "Explain quantum computing"
agentkit run my-agent.yaml --provider goose --model gemini-pro --input "Explain quantum computing"
```

#### Method 4: Goose Multi-Model (Recommended for Flexibility)
Provides access to multiple model providers through a single interface.

#### AWS Infrastructure Setup

1. **IAM Role**: Ensure your Lambda has appropriate permissions:
   - For Bedrock: `bedrock:InvokeModel` permissions
   - For Secrets Manager: `secretsmanager:GetSecretValue` permissions (if using)
2. **Environment Variables**: Set region and any secret ARNs
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