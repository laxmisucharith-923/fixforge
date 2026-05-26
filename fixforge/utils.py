"""Shared utilities: logging, env loading, repo parsing, cost tracking."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final
from urllib.parse import urlparse

from dotenv import load_dotenv

# Approximate USD per 1M tokens (input, output) — update as pricing changes.
COST_PER_MILLION: Final[dict[str, tuple[float, float]]] = {
    "groq": (0.05, 0.08),
    "deepseek": (0.14, 0.28),
    "anthropic": (3.0, 15.0),
    "openai": (0.15, 0.60),
}

DEFAULT_LOG_FORMAT: Final[str] = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)


@dataclass
class RepoRef:
    """Parsed GitHub repository reference."""

    owner: str
    name: str

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.name}"


@dataclass
class CostTracker:
    """Accumulates LLM token usage and estimated cost."""

    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    estimated_usd: float = 0.0
    details: list[str] = field(default_factory=list)

    def record(
        self,
        input_tokens: int,
        output_tokens: int,
        label: str = "llm_call",
    ) -> None:
        """Record token usage from a single LLM invocation."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.calls += 1

        rates = COST_PER_MILLION.get(self.provider, (1.0, 2.0))
        cost = (input_tokens / 1_000_000) * rates[0] + (
            output_tokens / 1_000_000
        ) * rates[1]
        self.estimated_usd += cost
        self.details.append(
            f"{label}: in={input_tokens}, out={output_tokens}, ~${cost:.6f}"
        )

    def summary(self) -> str:
        """Human-readable cost summary."""
        return (
            f"LLM calls: {self.calls} | "
            f"Tokens: {self.input_tokens} in / {self.output_tokens} out | "
            f"Estimated cost: ~${self.estimated_usd:.4f} ({self.provider})"
        )


def setup_logging(level: str | None = None) -> None:
    """Configure root logging once."""
    log_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format=DEFAULT_LOG_FORMAT,
        force=True,
    )


def load_environment(env_path: Path | None = None) -> None:
    """Load variables from .env (project root or cwd)."""
    if env_path and env_path.is_file():
        load_dotenv(env_path)
        return
    # Try package parent (repo root) then cwd.
    candidates = [
        Path(__file__).resolve().parent.parent / ".env",
        Path.cwd() / ".env",
    ]
    for path in candidates:
        if path.is_file():
            load_dotenv(path)
            return
    load_dotenv()


def get_env(key: str, default: str | None = None, required: bool = False) -> str:
    """Read an environment variable with optional requirement."""
    value = os.getenv(key, default)
    if required and not value:
        raise RuntimeError(
            f"Missing required environment variable: {key}. "
            f"Copy .env.example to .env and set your keys."
        )
    return value or ""


def parse_repo_url(repo_url: str) -> RepoRef:
    """
    Parse owner/repo from a GitHub URL or 'owner/repo' shorthand.

    Supports:
      - https://github.com/owner/repo
      - https://github.com/owner/repo.git
      - git@github.com:owner/repo.git
      - owner/repo
    """
    repo_url = repo_url.strip().rstrip("/")

    if re.match(r"^[\w.-]+/[\w.-]+$", repo_url):
        owner, name = repo_url.split("/", 1)
        return RepoRef(owner=owner, name=name.removesuffix(".git"))

    if repo_url.startswith("git@"):
        # git@github.com:owner/repo.git
        match = re.match(r"git@[^:]+:([^/]+)/(.+?)(?:\.git)?$", repo_url)
        if match:
            return RepoRef(owner=match.group(1), name=match.group(2))
        raise ValueError(f"Invalid git SSH URL: {repo_url}")

    parsed = urlparse(repo_url)
    if "github.com" not in parsed.netloc:
        raise ValueError(f"Not a GitHub URL: {repo_url}")

    parts = [p for p in parsed.path.strip("/").split("/") if p]
    if len(parts) < 2:
        raise ValueError(f"Could not parse owner/repo from: {repo_url}")

    return RepoRef(owner=parts[0], name=parts[1].removesuffix(".git"))


def truncate_text(text: str, max_chars: int = 12_000) -> str:
    """Truncate long text with an ellipsis marker."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return (
        text[:half]
        + f"\n\n... [truncated {len(text) - max_chars} chars] ...\n\n"
        + text[-half:]
    )
