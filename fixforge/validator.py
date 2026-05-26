"""Apply patches to a working copy and run automated validation (ruff, pytest)."""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

DIFF_FENCE_RE = re.compile(
    r"```(?:diff)?\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)
DIFF_HEADER_RE = re.compile(r"^---\s+", re.MULTILINE)
CHANGED_FILE_RE = re.compile(r"^\+\+\+\s+[ab]/(.+)$", re.MULTILINE)

DEFAULT_PYTEST_TIMEOUT = 120
DEFAULT_CMD_TIMEOUT = 90


@dataclass
class ValidationResult:
    """Outcome of one validate attempt."""

    attempt: int
    success: bool
    patch_applied: bool
    apply_output: str
    changed_files: list[str] = field(default_factory=list)
    ruff_passed: bool = True
    ruff_output: str = ""
    pytest_ran: bool = False
    pytest_passed: bool | None = None
    pytest_output: str = ""
    work_dir: Path | None = None

    @property
    def combined_errors(self) -> str:
        """Formatted errors for LLM self-correction."""
        if self.success:
            return ""
        sections: list[str] = []
        if not self.patch_applied:
            sections.append(
                "### Patch application failed\n"
                "The unified diff could not be applied to the repository copy.\n"
                f"```\n{self.apply_output.strip()}\n```"
            )
        if not self.ruff_passed:
            sections.append(
                "### Ruff lint errors\n"
                f"```\n{self.ruff_output.strip() or '(no output)'}\n```"
            )
        if self.pytest_ran and self.pytest_passed is False:
            sections.append(
                "### Pytest failures\n"
                f"```\n{self.pytest_output.strip() or '(no output)'}\n```"
            )
        return "\n\n".join(sections)


@dataclass
class FixLoopResult:
    """Final result after iterative fix + validation."""

    success: bool
    attempts_used: int
    final_diff: str
    final_raw_response: str
    validation_history: list[ValidationResult] = field(default_factory=list)
    cost_summary: str = ""


def extract_patch_from_response(text: str) -> str:
    """
    Extract unified diff text from LLM markdown (```diff blocks or raw diff).
    """
    blocks = DIFF_FENCE_RE.findall(text)
    if blocks:
        combined = "\n".join(b.strip() for b in blocks if b.strip())
        if combined:
            return _normalize_patch(combined)

    # Raw diff without fences: from first --- line onward.
    match = DIFF_HEADER_RE.search(text)
    if match:
        return _normalize_patch(text[match.start() :].strip())

    return ""


def _normalize_patch(patch: str) -> str:
    """Strip markdown artifacts and ensure trailing newline."""
    lines: list[str] = []
    for line in patch.splitlines():
        if line.strip().startswith("```"):
            continue
        lines.append(line.rstrip())
    result = "\n".join(lines)
    if result and not result.endswith("\n"):
        result += "\n"
    return result


def parse_changed_files(patch: str) -> list[str]:
    """Parse file paths from +++ b/... headers."""
    files: list[str] = []
    seen: set[str] = set()
    for match in CHANGED_FILE_RE.finditer(patch):
        path = match.group(1).strip()
        if path.startswith("b/"):
            path = path[2:]
        if path not in seen and path != "/dev/null":
            seen.add(path)
            files.append(path)
    return files


class PatchValidator:
    """Apply patches on isolated repo copies and run ruff + pytest."""

    def __init__(
        self,
        source_repo: Path,
        work_root: Path | None = None,
        issue_number: int | None = None,
    ) -> None:
        self.source_repo = source_repo.resolve()
        self.work_root = (work_root or Path("./temp_repos/_validate")).resolve()
        self.work_root.mkdir(parents=True, exist_ok=True)
        self._issue_tag = f"issue{issue_number}" if issue_number else "run"

    def _work_dir_for_attempt(self, attempt: int) -> Path:
        name = f"{self.source_repo.name}_{self._issue_tag}_attempt{attempt}"
        return self.work_root / name

    def prepare_work_copy(self, attempt: int) -> Path:
        """Create a fresh copy of the source repo for one attempt."""
        dest = self._work_dir_for_attempt(attempt)
        if dest.exists():
            logger.debug("Removing prior work copy: %s", dest)
            shutil.rmtree(dest, ignore_errors=True)
        logger.info("Creating validation work copy: %s", dest)
        shutil.copytree(
            self.source_repo,
            dest,
            ignore=shutil.ignore_patterns(
                ".git",
                ".pytest_cache",
                "__pycache__",
                ".ruff_cache",
                "*.pyc",
            ),
            dirs_exist_ok=True,
        )
        return dest

    def apply_patch(self, work_dir: Path, patch: str) -> tuple[bool, str]:
        """Apply unified diff via git apply, then patch -p1 as fallback."""
        if not patch.strip():
            return False, "No patch content extracted from LLM response."

        patch_file = work_dir / ".fixforge.patch"
        patch_file.write_text(patch, encoding="utf-8")
        logger.info("Applying patch (%d bytes) in %s", len(patch), work_dir)

        last_output = ""
        for cmd, label in (
            (["git", "apply", "--verbose", str(patch_file)], "git apply"),
            (["git", "apply", "--verbose", "-p0", str(patch_file)], "git apply -p0"),
            (
                ["patch", "-p1", "--forward", "-i", str(patch_file)],
                "patch -p1",
            ),
        ):
            try:
                result = subprocess.run(
                    cmd,
                    cwd=work_dir,
                    capture_output=True,
                    text=True,
                    timeout=DEFAULT_CMD_TIMEOUT,
                    check=False,
                )
            except FileNotFoundError:
                logger.debug("%s not available on PATH", cmd[0])
                continue

            last_output = _combine_output(result)
            if result.returncode == 0:
                logger.info("Patch applied successfully via %s", label)
                return True, last_output

            logger.warning("%s failed (exit %s)", label, result.returncode)

        return False, (
            "All patch application methods failed.\n"
            "Ensure the diff uses correct paths (--- a/file, +++ b/file) "
            "and matches the repository.\n"
            f"Last output:\n{last_output or 'N/A'}"
        )

    def run_ruff(self, work_dir: Path, targets: list[str]) -> tuple[bool, str]:
        """Run ruff check on changed files or entire project."""
        args = ["ruff", "check"]
        if targets:
            existing = [str(work_dir / t) for t in targets if (work_dir / t).exists()]
            args.extend(existing or ["."])
        else:
            args.append(".")

        logger.info("Running: %s (cwd=%s)", " ".join(args[:4]), work_dir)
        try:
            result = subprocess.run(
                args,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=DEFAULT_CMD_TIMEOUT,
                check=False,
            )
        except FileNotFoundError:
            logger.warning("ruff not found on PATH — skipping lint")
            return True, "ruff not installed; lint skipped."

        output = _combine_output(result)
        passed = result.returncode == 0
        if passed:
            logger.info("Ruff check passed")
        else:
            logger.warning("Ruff check failed (exit %s)", result.returncode)
        return passed, output

    def discover_pytest_targets(
        self, work_dir: Path, changed_files: list[str]
    ) -> list[str]:
        """Map changed source files to likely test modules."""
        targets: list[str] = []
        seen: set[str] = set()

        for rel in changed_files:
            path = Path(rel)
            stem = path.stem
            parent = path.parent

            candidates = [
                Path("tests") / f"test_{stem}.py",
                Path("tests") / parent / f"test_{stem}.py",
                Path("test") / f"test_{stem}.py",
                parent / f"test_{stem}.py",
                parent / f"{stem}_test.py",
            ]
            if stem.startswith("test_"):
                candidates.insert(0, path)

            for cand in candidates:
                full = work_dir / cand
                key = str(cand).replace("\\", "/")
                if full.is_file() and key not in seen:
                    seen.add(key)
                    targets.append(key)

        return targets

    def pytest_available(self, work_dir: Path) -> bool:
        """Heuristic: repo likely has pytest if tests/ or pytest config exists."""
        has_tests = (work_dir / "tests").is_dir() or (work_dir / "test").is_dir()
        has_config = (
            (work_dir / "pytest.ini").is_file()
            or (work_dir / "setup.cfg").is_file()
            or (work_dir / "tox.ini").is_file()
            or _pyproject_has_pytest(work_dir / "pyproject.toml")
        )
        return has_tests or has_config

    def run_pytest(
        self, work_dir: Path, changed_files: list[str]
    ) -> tuple[bool, bool, str]:
        """
        Run pytest on related tests or a quick smoke run.

        Returns:
            (ran, passed, output)
        """
        if not self.pytest_available(work_dir):
            logger.info("No pytest suite detected — skipping tests")
            return False, None, "pytest skipped (no tests/ or config found)"

        targets = self.discover_pytest_targets(work_dir, changed_files)
        args = ["python", "-m", "pytest", "-q", "--tb=short", "--no-header"]
        if targets:
            args.extend(targets[:8])
            logger.info("Running pytest on: %s", targets[:8])
        else:
            args.extend(["--maxfail=3", "-x"])
            logger.info("Running pytest smoke (maxfail=3)")

        try:
            result = subprocess.run(
                args,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=DEFAULT_PYTEST_TIMEOUT,
                check=False,
                env=_pytest_env(),
            )
        except FileNotFoundError:
            return False, None, "pytest not available"
        except subprocess.TimeoutExpired:
            return True, False, f"pytest timed out after {DEFAULT_PYTEST_TIMEOUT}s"

        output = _combine_output(result)
        passed = result.returncode == 0
        if passed:
            logger.info("Pytest passed")
        else:
            logger.warning("Pytest failed (exit %s)", result.returncode)
        return True, passed, output

    def validate(
        self,
        patch: str,
        attempt: int,
    ) -> ValidationResult:
        """Full validation pipeline for one attempt."""
        logger.info("=== Validation attempt %d ===", attempt)
        work_dir = self.prepare_work_copy(attempt)
        changed_files = parse_changed_files(patch)

        applied, apply_out = self.apply_patch(work_dir, patch)
        if not applied:
            return ValidationResult(
                attempt=attempt,
                success=False,
                patch_applied=False,
                apply_output=apply_out,
                changed_files=changed_files,
                work_dir=work_dir,
            )

        ruff_ok, ruff_out = self.run_ruff(work_dir, changed_files)
        pytest_ran, pytest_ok, pytest_out = self.run_pytest(work_dir, changed_files)

        success = ruff_ok and (not pytest_ran or pytest_ok is True)
        logger.info(
            "Attempt %d result: success=%s (ruff=%s, pytest=%s)",
            attempt,
            success,
            ruff_ok,
            pytest_ok,
        )

        return ValidationResult(
            attempt=attempt,
            success=success,
            patch_applied=True,
            apply_output=apply_out,
            changed_files=changed_files,
            ruff_passed=ruff_ok,
            ruff_output=ruff_out,
            pytest_ran=pytest_ran,
            pytest_passed=pytest_ok,
            pytest_output=pytest_out,
            work_dir=work_dir,
        )

    async def validate_async(self, patch: str, attempt: int) -> ValidationResult:
        """Async wrapper for validate."""
        return await asyncio.to_thread(self.validate, patch, attempt)


def _combine_output(result: subprocess.CompletedProcess[str]) -> str:
    parts = []
    if result.stdout:
        parts.append(result.stdout)
    if result.stderr:
        parts.append(result.stderr)
    return "\n".join(parts).strip()


def _pytest_env() -> dict[str, str]:
    import os

    env = os.environ.copy()
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    return env


def _pyproject_has_pytest(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return "pytest" in text
