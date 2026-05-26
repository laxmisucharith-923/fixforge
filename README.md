# FixForge

AI-powered GitHub issue fixer (basic MVP). Given a repository URL and issue number, FixForge fetches the issue, clones the repo, finds relevant files, and uses an LLM to suggest a patch you can review and apply manually.

**No auto-commit or PR** in this version.

## Features

- GitHub issue fetch via PyGithub
- Shallow clone to `./temp_repos/`
- Keyword-based file relevance scoring (top 5–7 files)
- LangChain LLM orchestration (Groq, DeepSeek, Anthropic, OpenAI)
- Unified diff / patch style output
- Cost tracking (estimated USD from token usage)
- Async pipeline for I/O-bound steps

## Project layout

```
fixforge/
├── main.py           # CLI entry point
├── github_client.py  # GitHub API
├── repo_analyzer.py  # Clone + file search
├── llm_fixer.py      # LLM fix generation
├── utils.py          # Env, logging, parsing
├── .env.example
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.11+
- Git installed and on `PATH`
- `GITHUB_TOKEN` (recommended; required for private repos)
- `LLM_API_KEY` for your chosen provider

## Quick start

```bash
cd FixForge
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env   # Windows
# cp .env.example .env   # macOS/Linux
```

Edit `.env` with your tokens, then run:

```bash
python -m fixforge.main https://github.com/owner/repo 123
```

### Useful flags

| Flag | Description |
|------|-------------|
| `--no-llm` | Fetch issue + analyze files only (no API cost) |
| `--refresh` | Force re-clone |
| `--provider groq` | Override LLM provider |
| `-o report.txt` | Save full report to file |
| `--log-level DEBUG` | Verbose logging |

### Lint (ruff)

```bash
ruff check fixforge/
ruff format fixforge/
```

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GITHUB_TOKEN` | Recommended | GitHub personal access token |
| `LLM_API_KEY` | Yes (for fixes) | Provider API key |
| `LLM_PROVIDER` | No | `groq`, `deepseek`, `anthropic`, `openai` (default: `groq`) |
| `LLM_MODEL` | No | Model id override |
| `LOG_LEVEL` | No | `DEBUG`, `INFO`, etc. |

## Extending later

- Agent loops: wrap `LLMFixer.generate_fix` in a LangGraph graph with tool nodes
- Auto-PR: add a `pr_creator.py` module after human approval gate
- Better search: swap keyword scorer for tree-sitter or embeddings

## License

MIT (add your license file as needed)
