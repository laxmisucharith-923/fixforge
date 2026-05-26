"""LLM orchestration for generating issue fixes via LangChain."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from fixforge.github_client import IssueDetails
from fixforge.repo_analyzer import RelevantFile
from fixforge.utils import CostTracker, get_env, truncate_text

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior software engineer helping fix GitHub issues.

Your job:
1. Understand the issue thoroughly.
2. Analyze the provided repository files.
3. Propose a minimal, correct fix — no drive-by refactors.
4. Output changes in unified diff / patch format when possible.

Rules:
- Prefer small, focused changes.
- Explain root cause briefly before the patch.
- If information is missing, state assumptions clearly.
- Do NOT invent files that are not shown unless you explain why they are needed.
- Do NOT suggest auto-committing or opening PRs — the human will apply changes manually.
"""

OUTPUT_FORMAT_INSTRUCTIONS = """
Respond in this exact structure:

## Summary
(2-4 sentences: root cause and fix approach)

## Files to change
(bullet list of file paths)

## Explanation
(step-by-step reasoning)

## Patch
(One or more unified diff blocks. Use ```diff fences. Example:)

```diff
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -1,3 +1,3 @@
-old line
+new line
```

## Manual steps
(any commands, tests, or config the user should run)

## Risks / edge cases
(optional, brief)
"""


@dataclass
class FixResult:
    """LLM-generated fix proposal."""

    explanation: str
    raw_response: str
    provider: str
    model: str
    cost_tracker: CostTracker


class LLMFixer:
    """Generate code fixes using LangChain chat models."""

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.provider = (provider or get_env("LLM_PROVIDER", "groq")).lower()
        self.model = model or get_env("LLM_MODEL", "")
        self.api_key = api_key or get_env("LLM_API_KEY", required=True)
        self.cost_tracker = CostTracker(provider=self.provider)
        self._llm = self._build_llm()

    def _default_model(self) -> str:
        defaults = {
            "groq": "llama-3.3-70b-versatile",
            "deepseek": "deepseek-chat",
            "anthropic": "claude-sonnet-4-20250514",
            "openai": "gpt-4o-mini",
        }
        return defaults.get(self.provider, "gpt-4o-mini")

    def _build_llm(self) -> BaseChatModel:
        """Instantiate the configured LangChain chat model."""
        model_name = self.model or self._default_model()
        logger.info("Using LLM provider=%s model=%s", self.provider, model_name)

        if self.provider == "groq":
            from langchain_groq import ChatGroq

            return ChatGroq(
                api_key=self.api_key,
                model=model_name,
                temperature=0.1,
            )

        if self.provider == "deepseek":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com",
                model=model_name,
                temperature=0.1,
            )

        if self.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(
                api_key=self.api_key,
                model=model_name,
                temperature=0.1,
                max_tokens=8192,
            )

        if self.provider == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                api_key=self.api_key,
                model=model_name,
                temperature=0.1,
            )

        raise ValueError(
            f"Unsupported LLM_PROVIDER '{self.provider}'. "
            "Use: groq, deepseek, anthropic, openai"
        )

    def _build_user_prompt(
        self,
        issue: IssueDetails,
        relevant_files: list[RelevantFile],
        repo_language: str | None,
    ) -> str:
        """Assemble the user message with issue + file context."""
        file_sections: list[str] = []
        for rf in relevant_files:
            file_sections.append(
                f"### File: `{rf.path}` (relevance score: {rf.score:.1f})\n"
                f"```\n{rf.content}\n```"
            )

        files_blob = (
            "\n\n".join(file_sections)
            if file_sections
            else "_No relevant files found — reason from issue description only._"
        )

        return (
            f"## GitHub Issue\n\n"
            f"{truncate_text(issue.full_description, max_chars=8_000)}\n\n"
            f"## Repository\n"
            f"- Repo: {issue.repo_full_name}\n"
            f"- Primary language: {repo_language or 'unknown'}\n\n"
            f"## Relevant source files\n\n"
            f"{files_blob}\n\n"
            f"{OUTPUT_FORMAT_INSTRUCTIONS}"
        )

    def _extract_usage(self, response: object) -> tuple[int, int]:
        """Parse token usage from LangChain response metadata."""
        input_tokens = 0
        output_tokens = 0

        meta = getattr(response, "response_metadata", None) or {}
        usage = meta.get("token_usage") or meta.get("usage") or {}

        if isinstance(usage, dict):
            input_tokens = int(
                usage.get("prompt_tokens") or usage.get("input_tokens") or 0
            )
            output_tokens = int(
                usage.get("completion_tokens") or usage.get("output_tokens") or 0
            )

        # Anthropic puts usage at top level sometimes.
        if not input_tokens and hasattr(response, "usage_metadata"):
            um = response.usage_metadata or {}
            if isinstance(um, dict):
                input_tokens = int(um.get("input_tokens", 0))
                output_tokens = int(um.get("output_tokens", 0))

        return input_tokens, output_tokens

    def generate_fix(
        self,
        issue: IssueDetails,
        relevant_files: list[RelevantFile],
        repo_language: str | None = None,
    ) -> FixResult:
        """Call the LLM and return a structured fix proposal (sync)."""
        user_prompt = self._build_user_prompt(issue, relevant_files, repo_language)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        logger.info("Invoking LLM for fix generation...")
        response = self._llm.invoke(messages)
        raw = StrOutputParser().invoke(response)

        input_tok, output_tok = self._extract_usage(response)
        # Rough estimate if provider omits usage metadata.
        if not output_tok:
            output_tok = max(len(raw) // 4, 1)
        if not input_tok:
            input_tok = max(len(user_prompt) // 4, 1)

        self.cost_tracker.record(input_tok, output_tok, label="generate_fix")
        logger.info(self.cost_tracker.summary())

        return FixResult(
            explanation=self._extract_section(raw, "Summary"),
            raw_response=raw,
            provider=self.provider,
            model=self.model or self._default_model(),
            cost_tracker=self.cost_tracker,
        )

    async def generate_fix_async(
        self,
        issue: IssueDetails,
        relevant_files: list[RelevantFile],
        repo_language: str | None = None,
    ) -> FixResult:
        """Async wrapper around generate_fix."""
        return await asyncio.to_thread(
            self.generate_fix, issue, relevant_files, repo_language
        )

    @staticmethod
    def _extract_section(text: str, heading: str) -> str:
        """Pull content under a ## Heading marker."""
        pattern = rf"##\s*{re.escape(heading)}\s*\n(.*?)(?=\n##\s|\Z)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""
