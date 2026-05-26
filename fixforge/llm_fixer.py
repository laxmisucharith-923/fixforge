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
from fixforge.validator import (
    FixLoopResult,
    PatchValidator,
    ValidationResult,
    extract_patch_from_response,
)

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
- Patches MUST apply cleanly with `git apply` (correct --- a/ and +++ b/ paths).
- Use paths relative to the repository root exactly as they exist in the project.
"""

REFINEMENT_SYSTEM_PROMPT = """\
You are a senior software engineer fixing a FAILED patch attempt.

The previous patch was applied and/or validated automatically. It did not pass.
Your job is to produce a CORRECTED, COMPLETE replacement patch — not a partial edit.

Critical rules:
1. Read validation errors carefully — they are ground truth.
2. Fix ONLY what is broken; do not introduce unrelated changes.
3. Output a full unified diff that replaces the previous attempt entirely.
4. Ensure paths in the diff match the real repository layout.
5. The patch must pass `ruff check` on changed files when possible.
6. Consider test failures — fix the code, not the tests, unless tests are wrong.
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

REFINEMENT_USER_TEMPLATE = """
## CORRECTION REQUIRED — Attempt {attempt} of {max_attempts}

Your previous fix did **not** pass automated validation.
Study the errors and produce a corrected patch.

### Validation failures
{validation_errors}

### Your previous patch (failed — do not repeat the same mistakes)
```diff
{previous_patch}
```

### Your previous response summary
{previous_summary}

### Original GitHub issue
{issue_description}

### Relevant source files (for context)
{files_context}

### Instructions
1. Explain briefly what went wrong in the previous attempt.
2. Provide a **complete new unified diff** in ## Patch
   (not incremental edits on top of broken state).
3. Ensure the diff applies with standard `git apply` / `patch -p1`.
4. Address every validation error listed above.

{output_format}
"""


@dataclass
class FixResult:
    """LLM-generated fix proposal."""

    explanation: str
    raw_response: str
    provider: str
    model: str
    cost_tracker: CostTracker
    patch_text: str = ""


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

    def _build_files_context(self, relevant_files: list[RelevantFile]) -> str:
        file_sections: list[str] = []
        for rf in relevant_files:
            file_sections.append(
                f"### File: `{rf.path}` (relevance score: {rf.score:.1f})\n"
                f"```\n{rf.content}\n```"
            )
        return (
            "\n\n".join(file_sections)
            if file_sections
            else "_No relevant files found._"
        )

    def _build_user_prompt(
        self,
        issue: IssueDetails,
        relevant_files: list[RelevantFile],
        repo_language: str | None,
    ) -> str:
        """Assemble the user message with issue + file context."""
        return (
            f"## GitHub Issue\n\n"
            f"{truncate_text(issue.full_description, max_chars=8_000)}\n\n"
            f"## Repository\n"
            f"- Repo: {issue.repo_full_name}\n"
            f"- Primary language: {repo_language or 'unknown'}\n\n"
            f"## Relevant source files\n\n"
            f"{self._build_files_context(relevant_files)}\n\n"
            f"{OUTPUT_FORMAT_INSTRUCTIONS}"
        )

    def _build_refinement_prompt(
        self,
        issue: IssueDetails,
        relevant_files: list[RelevantFile],
        validation_errors: str,
        previous_patch: str,
        previous_response: str,
        attempt: int,
        max_attempts: int,
    ) -> str:
        """Prompt for self-correction after validation failure."""
        summary = self._extract_section(previous_response, "Summary") or (
            previous_response[:500] + "..."
        )
        return REFINEMENT_USER_TEMPLATE.format(
            attempt=attempt,
            max_attempts=max_attempts,
            validation_errors=validation_errors,
            previous_patch=truncate_text(previous_patch, max_chars=6_000),
            previous_summary=summary,
            issue_description=truncate_text(issue.full_description, max_chars=4_000),
            files_context=self._build_files_context(relevant_files),
            output_format=OUTPUT_FORMAT_INSTRUCTIONS,
        )

    def _invoke_llm(
        self,
        system: str,
        user: str,
        label: str,
    ) -> str:
        """Call LLM and record token usage."""
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=user),
        ]
        logger.info("Invoking LLM (%s)...", label)
        response = self._llm.invoke(messages)
        raw = StrOutputParser().invoke(response)

        input_tok, output_tok = self._extract_usage(response)
        if not output_tok:
            output_tok = max(len(raw) // 4, 1)
        if not input_tok:
            input_tok = max(len(user) // 4, 1)

        self.cost_tracker.record(input_tok, output_tok, label=label)
        logger.info(self.cost_tracker.summary())
        return raw

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

        if not input_tokens and hasattr(response, "usage_metadata"):
            um = response.usage_metadata or {}
            if isinstance(um, dict):
                input_tokens = int(um.get("input_tokens", 0))
                output_tokens = int(um.get("output_tokens", 0))

        return input_tokens, output_tokens

    def _to_fix_result(self, raw: str) -> FixResult:
        patch = extract_patch_from_response(raw)
        return FixResult(
            explanation=self._extract_section(raw, "Summary"),
            raw_response=raw,
            provider=self.provider,
            model=self.model or self._default_model(),
            cost_tracker=self.cost_tracker,
            patch_text=patch,
        )

    def generate_fix(
        self,
        issue: IssueDetails,
        relevant_files: list[RelevantFile],
        repo_language: str | None = None,
    ) -> FixResult:
        """Call the LLM and return a structured fix proposal (sync)."""
        user_prompt = self._build_user_prompt(issue, relevant_files, repo_language)
        raw = self._invoke_llm(SYSTEM_PROMPT, user_prompt, label="generate_fix")
        return self._to_fix_result(raw)

    def refine_fix(
        self,
        issue: IssueDetails,
        relevant_files: list[RelevantFile],
        validation_errors: str,
        previous_patch: str,
        previous_response: str,
        attempt: int,
        max_attempts: int = 3,
    ) -> FixResult:
        """Generate a corrected patch after validation failure."""
        user_prompt = self._build_refinement_prompt(
            issue,
            relevant_files,
            validation_errors,
            previous_patch,
            previous_response,
            attempt,
            max_attempts,
        )
        raw = self._invoke_llm(
            REFINEMENT_SYSTEM_PROMPT,
            user_prompt,
            label=f"refine_fix_attempt_{attempt}",
        )
        return self._to_fix_result(raw)

    def run_validated_fix_loop(
        self,
        issue: IssueDetails,
        relevant_files: list[RelevantFile],
        validator: PatchValidator,
        repo_language: str | None = None,
        max_iterations: int = 3,
    ) -> FixLoopResult:
        """
        Generate fix → validate → refine up to max_iterations times.

        Returns final outcome with success flag, diff, and validation history.
        """
        history: list[ValidationResult] = []
        fix_result: FixResult | None = None
        patch = ""
        attempts_used = 0

        for attempt in range(1, max_iterations + 1):
            attempts_used = attempt
            logger.info("Fix loop iteration %d/%d", attempt, max_iterations)

            if attempt == 1:
                fix_result = self.generate_fix(issue, relevant_files, repo_language)
            else:
                assert fix_result is not None
                last = history[-1]
                fix_result = self.refine_fix(
                    issue,
                    relevant_files,
                    validation_errors=last.combined_errors,
                    previous_patch=patch,
                    previous_response=fix_result.raw_response,
                    attempt=attempt,
                    max_attempts=max_iterations,
                )

            patch = fix_result.patch_text or extract_patch_from_response(
                fix_result.raw_response
            )
            if not patch:
                logger.error(
                    "No patch extracted from LLM response on attempt %d", attempt
                )
                history.append(
                    ValidationResult(
                        attempt=attempt,
                        success=False,
                        patch_applied=False,
                        apply_output=(
                            "LLM response did not contain a parseable unified diff. "
                            "Ensure the model outputs a ## Patch section "
                            "with ```diff blocks."
                        ),
                    )
                )
                if attempt >= max_iterations:
                    break
                continue

            validation = validator.validate(patch, attempt)
            history.append(validation)

            if validation.success:
                logger.info("Validation passed on attempt %d", attempt)
                return FixLoopResult(
                    success=True,
                    attempts_used=attempts_used,
                    final_diff=patch,
                    final_raw_response=fix_result.raw_response,
                    validation_history=history,
                    cost_summary=self.cost_tracker.summary(),
                )

            logger.warning(
                "Validation failed on attempt %d — %s",
                attempt,
                "will refine" if attempt < max_iterations else "max iterations reached",
            )

        assert fix_result is not None
        return FixLoopResult(
            success=False,
            attempts_used=attempts_used,
            final_diff=patch,
            final_raw_response=fix_result.raw_response,
            validation_history=history,
            cost_summary=self.cost_tracker.summary(),
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

    async def run_validated_fix_loop_async(
        self,
        issue: IssueDetails,
        relevant_files: list[RelevantFile],
        validator: PatchValidator,
        repo_language: str | None = None,
        max_iterations: int = 3,
    ) -> FixLoopResult:
        """Async wrapper for run_validated_fix_loop."""
        return await asyncio.to_thread(
            self.run_validated_fix_loop,
            issue,
            relevant_files,
            validator,
            repo_language,
            max_iterations,
        )

    @staticmethod
    def _extract_section(text: str, heading: str) -> str:
        """Pull content under a ## Heading marker."""
        pattern = rf"##\s*{re.escape(heading)}\s*\n(.*?)(?=\n##\s|\Z)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""
