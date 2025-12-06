# Data Extraction Agent

AI-powered data extraction agent built on LangChain's [Deep Agents](https://github.com/langchain-ai/deepagents) framework with **multi-provider LLM support** prioritizing **open-source models**.

## Features

- **Multi-Provider LLM Support**: Groq, Together AI, OpenRouter, HuggingFace, Google Gemini, Anthropic Claude
- **Open-Source First**: Prioritizes free/open-source models (Llama, Mixtral, Qwen)
- **Automatic Fallback**: Seamlessly switches between providers on failures
- **Cost Optimization**: Smart routing based on task complexity
- **Data Sources**: Web, APIs, Databases (SQL/MongoDB), Files (CSV, Excel, JSON, PDF)
- **Built on Deep Agents**: Planning, file system, sub-agent delegation

## Quick Start (GitHub Codespaces)

### 1. Open in Codespace

Click the button below to launch directly in GitHub Codespaces:

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/SharminSirajudeen/data_extraction_agent)

Or manually:
1. Go to the repository on GitHub
2. Click "Code" -> "Codespaces" -> "Create codespace on main"

### 2. Configure API Keys

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add at least ONE provider key
# Groq is recommended - it's FREE!
```

**Minimum requirement**: At least one API key:
| Provider | Free Tier | Get Key |
|----------|-----------|---------|
| **Groq** | Yes (Recommended!) | [console.groq.com](https://console.groq.com) |
| Together AI | Limited | [together.ai](https://api.together.xyz) |
| OpenRouter | Some models | [openrouter.ai](https://openrouter.ai) |
| HuggingFace | Yes | [huggingface.co](https://huggingface.co/settings/tokens) |
| Google | Yes | [aistudio.google.com](https://aistudio.google.com/apikey) |
| Anthropic | No | [console.anthropic.com](https://console.anthropic.com) |

### 3. Run the Agent

**Option A: Jupyter Notebook** (Interactive)
```bash
jupyter notebook data_extraction_agent.ipynb
```

**Option B: LangGraph Server** (Production)
```bash
langgraph dev
```

**Option C: Python Script**
```python
from data_extraction_agent import create_data_extraction_agent

agent = create_data_extraction_agent()

result = await agent.ainvoke({
    "messages": [{"role": "user", "content": "Extract all products from https://fakestoreapi.com/products"}]
})
```

## Architecture

```
data_extraction_agent/
├── data_extraction_agent/
│   ├── __init__.py          # Package exports
│   ├── agent.py             # Main agent configuration
│   ├── tools.py             # 12 extraction tools
│   ├── prompts.py           # Workflow prompts
│   └── providers/           # Multi-provider LLM support
│       ├── factory.py       # Provider factory
│       ├── router.py        # Smart routing
│       └── fallback.py      # Automatic fallback
├── data_extraction_agent.ipynb  # Interactive notebook
├── langgraph.json           # LangGraph server config
├── pyproject.toml           # Dependencies
└── .env.example             # Environment template
```

## LLM Providers

### Priority Order (Open-Source First)

1. **Groq** - Free tier, ultra-fast (750 tokens/sec), Llama models
2. **Together AI** - Budget-friendly, access to 405B models
3. **OpenRouter** - Unified API, some free models
4. **HuggingFace** - Free experimentation
5. **Google Gemini** - Low cost, long context
6. **Anthropic Claude** - Commercial fallback for complex tasks

### Model Selection by Task

| Task Type | Recommended Model | Provider |
|-----------|-------------------|----------|
| Simple extraction | llama-3.1-8b-instant | Groq |
| API parsing | llama-4-scout-17b | Groq |
| Schema analysis | Qwen2.5-72B | Together |
| Complex reasoning | claude-sonnet-4 | Anthropic |
| Long documents | gemini-1.5-pro | Google |

### Routing Strategies

```python
from data_extraction_agent.providers import RoutingStrategy

# Open-source only (default)
agent = create_data_extraction_agent(strategy=RoutingStrategy.OPEN_SOURCE_ONLY)

# Cost optimized
agent = create_data_extraction_agent(strategy=RoutingStrategy.COST_OPTIMIZED)

# Quality first
agent = create_data_extraction_agent(strategy=RoutingStrategy.QUALITY_FIRST)

# Speed first
agent = create_data_extraction_agent(strategy=RoutingStrategy.SPEED_FIRST)
```

## Extraction Tools

| Tool | Purpose |
|------|---------|
| `web_search` | Search web for data sources |
| `fetch_url` | Get full webpage content |
| `call_api` | Make REST API calls |
| `query_sql_database` | Query PostgreSQL, MySQL, SQLite |
| `query_mongodb` | Query MongoDB collections |
| `extract_from_csv` | Parse CSV files |
| `extract_from_excel` | Parse Excel files |
| `extract_from_json` | Parse JSON files |
| `extract_from_pdf` | Extract text from PDFs |
| `analyze_schema` | Understand data structure |
| `transform_data` | Clean and filter data |
| `think_tool` | Strategic reflection |

## Deep Agents Built-in Tools

In addition to extraction tools, you get:
- `write_todos` / `read_todos` - Task planning
- `write_file` / `read_file` - File operations
- `task` - Sub-agent delegation
- `ls` / `glob` / `grep` - File system exploration
- `execute` - Shell commands (sandboxed)

## Example Use Cases

### 1. API Data Extraction
```python
request = """
Extract all products from https://fakestoreapi.com/products
Return: id, title, price, category, rating
Format: JSON
"""
```

### 2. Web Research
```python
request = """
Search for the top 10 Python web scraping libraries.
For each: name, GitHub URL, stars, main features.
Return as structured JSON report.
"""
```

### 3. Database Query
```python
request = """
Connect to PostgreSQL: postgresql://user:pass@host/db
Extract all users created in the last 30 days.
Fields: id, email, created_at, status
"""
```

### 4. File Processing
```python
request = """
Extract data from these files:
- /data/sales.csv
- /data/inventory.xlsx
- /data/config.json

Merge into a single dataset and save to /output/combined.json
"""
```

## Running in Codespaces

GitHub Codespaces provides 60 hours/month free (2-core) or 30 hours (4-core).

### Advantages
- **No local storage needed** - Everything runs in the cloud
- **Pre-configured environment** - DevContainer handles setup
- **Persistent workspace** - Code and data saved between sessions
- **Easy API key management** - Use Codespace secrets

### Storage Tips
- Codespace storage: 15GB default
- Use `/tmp` for temporary files
- Clean up downloaded data after extraction
- Use external storage (S3, GCS) for large datasets

### Resource Configuration
Edit `.devcontainer/devcontainer.json`:
```json
{
  "hostRequirements": {
    "cpus": 4,
    "memory": "8gb",
    "storage": "32gb"
  }
}
```

## Development

### Local Setup
```bash
# Clone repository
git clone https://github.com/SharminSirajudeen/data_extraction_agent.git
cd data_extraction_agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -e ".[dev]"

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Running Tests
```bash
pytest tests/
```

### Code Quality
```bash
ruff check .
black .
mypy .
```

## Deployment

### LangGraph Cloud
```bash
# Deploy to LangGraph Cloud
langgraph deploy
```

### Docker
```bash
docker build -t data-extraction-agent .
docker run -p 8000:8000 --env-file .env data-extraction-agent
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

MIT License - See [LICENSE](LICENSE) for details.

## Acknowledgments

- [LangChain Deep Agents](https://github.com/langchain-ai/deepagents) - Core framework
- [LangGraph](https://github.com/langchain-ai/langgraph) - Agent orchestration
- Inspired by the Deep Agents video by Lance from LangChain
