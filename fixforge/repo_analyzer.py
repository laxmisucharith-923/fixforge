"""Clone repositories and locate files relevant to a GitHub issue."""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from git import GitCommandError, Repo

from fixforge.github_client import IssueDetails, RepoInfo
from fixforge.utils import RepoRef, truncate_text

logger = logging.getLogger(__name__)

# Directories and files to skip when scanning.
SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".github",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        "dist",
        "build",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        "coverage",
        ".eggs",
        "vendor",
        "target",
    }
)

SKIP_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".ico",
        ".svg",
        ".woff",
        ".woff2",
        ".ttf",
        ".eot",
        ".mp4",
        ".mp3",
        ".zip",
        ".tar",
        ".gz",
        ".pdf",
        ".lock",
        ".min.js",
        ".min.css",
        ".pyc",
        ".pyo",
        ".so",
        ".dll",
        ".exe",
        ".bin",
    }
)

CODE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".java",
        ".go",
        ".rs",
        ".rb",
        ".php",
        ".cs",
        ".cpp",
        ".c",
        ".h",
        ".hpp",
        ".swift",
        ".kt",
        ".scala",
        ".vue",
        ".svelte",
        ".html",
        ".css",
        ".scss",
        ".sql",
        ".yaml",
        ".yml",
        ".toml",
        ".json",
        ".md",
        ".sh",
        ".bash",
        ".zsh",
        ".dockerfile",
    }
)

MAX_FILE_BYTES: int = 80_000
MAX_FILES: int = 7
MAX_SCAN_FILES: int = 500


@dataclass
class RelevantFile:
    """A repository file scored for relevance to an issue."""

    path: str
    score: float
    content: str
    size_bytes: int


@dataclass
class AnalysisResult:
    """Output of repository analysis."""

    clone_path: Path
    repo_info: RepoInfo
    relevant_files: list[RelevantFile]
    total_files_scanned: int
    keywords: list[str]


class RepoAnalyzer:
    """Clone and analyze a GitHub repository for issue-related files."""

    def __init__(
        self,
        temp_root: Path | None = None,
        github_token: str | None = None,
    ) -> None:
        self.temp_root = temp_root or Path("./temp_repos")
        self.temp_root.mkdir(parents=True, exist_ok=True)
        self._github_token = github_token

    def _clone_path(self, repo: RepoRef) -> Path:
        return self.temp_root / repo.full_name.replace("/", "_")

    def _extract_keywords(self, issue: IssueDetails) -> list[str]:
        """Build search keywords from issue title, body, and labels."""
        text = f"{issue.title} {issue.body} {' '.join(issue.labels)}"
        # Identifiers: snake_case, camelCase, dotted paths, file names.
        patterns = [
            r"`([^`]+)`",
            r"\b([a-zA-Z_][\w]*(?:\.[a-zA-Z_][\w]*)+)\b",
            r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b",
            r"\b([A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+)\b",
            r"([\w-]+\.(?:py|js|ts|tsx|jsx|go|rs|java|rb|php|cs|md))\b",
        ]
        keywords: set[str] = set()
        for pattern in patterns:
            for match in re.findall(pattern, text):
                token = match.strip().lower()
                if len(token) >= 3 and token not in _STOP_WORDS:
                    keywords.add(token)

        # Title words (longer than 3 chars).
        for word in re.findall(r"\b[a-zA-Z]{4,}\b", issue.title):
            w = word.lower()
            if w not in _STOP_WORDS:
                keywords.add(w)

        return sorted(keywords)[:40]

    def clone_repository(
        self,
        repo: RepoRef,
        clone_url: str,
        branch: str | None = None,
        force_refresh: bool = False,
    ) -> Path:
        """Clone or refresh a local copy of the repository."""
        dest = self._clone_path(repo)
        if dest.exists():
            if force_refresh:
                logger.info("Removing existing clone at %s", dest)
                shutil.rmtree(dest, ignore_errors=True)
            else:
                logger.info("Using existing clone at %s", dest)
                return dest

        url = clone_url
        if self._github_token and url.startswith("https://github.com"):
            url = url.replace(
                "https://github.com",
                f"https://{self._github_token}@github.com",
            )

        logger.info("Cloning %s -> %s", repo.full_name, dest)
        try:
            kwargs: dict = {"depth": 1}
            if branch:
                kwargs["branch"] = branch
            Repo.clone_from(url, dest, **kwargs)
        except GitCommandError as exc:
            raise RuntimeError(
                f"Failed to clone '{repo.full_name}'. "
                f"Check repo URL and token access.\n{exc}"
            ) from exc

        return dest

    def _iter_code_files(self, root: Path) -> list[Path]:
        """Collect scannable code files under root."""
        files: list[Path] = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            if path.suffix.lower() in SKIP_EXTENSIONS:
                continue
            if path.suffix.lower() not in CODE_EXTENSIONS and path.name not in (
                "Dockerfile",
                "Makefile",
            ):
                continue
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            files.append(path)
            if len(files) >= MAX_SCAN_FILES:
                break
        return files

    def _score_file(self, rel_path: str, content: str, keywords: list[str]) -> float:
        """Score file relevance based on keyword hits."""
        path_lower = rel_path.lower()
        content_lower = content.lower()
        score = 0.0

        for kw in keywords:
            kw_l = kw.lower()
            if kw_l in path_lower:
                score += 5.0
            count = content_lower.count(kw_l)
            if count:
                score += min(count * 1.5, 15.0)

        # Boost common entry points.
        basename = Path(rel_path).name.lower()
        if basename in ("main.py", "app.py", "index.js", "index.ts", "__init__.py"):
            score += 1.0

        depth = len(Path(rel_path).parts)
        score += max(0, 3 - depth * 0.3)

        return score

    def find_relevant_files(
        self,
        clone_path: Path,
        issue: IssueDetails,
        max_files: int = MAX_FILES,
    ) -> tuple[list[RelevantFile], list[str], int]:
        """Return top-N files most related to the issue keywords."""
        keywords = self._extract_keywords(issue)
        logger.info("Extracted %d keywords: %s", len(keywords), keywords[:15])

        candidates: list[RelevantFile] = []
        all_files = self._iter_code_files(clone_path)

        for file_path in all_files:
            rel = str(file_path.relative_to(clone_path)).replace("\\", "/")
            try:
                raw = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                logger.debug("Skip unreadable %s: %s", rel, exc)
                continue

            score = self._score_file(rel, raw, keywords)
            if score <= 0 and keywords:
                continue

            candidates.append(
                RelevantFile(
                    path=rel,
                    score=score,
                    content=truncate_text(raw, max_chars=10_000),
                    size_bytes=len(raw.encode("utf-8")),
                )
            )

        # If no keyword hits, include README + a few shallow source files.
        if not candidates:
            logger.warning("No keyword matches; falling back to README + root files")
            for file_path in all_files[:max_files]:
                rel = str(file_path.relative_to(clone_path)).replace("\\", "/")
                try:
                    raw = file_path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                candidates.append(
                    RelevantFile(
                        path=rel,
                        score=0.1,
                        content=truncate_text(raw, max_chars=10_000),
                        size_bytes=len(raw.encode("utf-8")),
                    )
                )

        candidates.sort(key=lambda f: f.score, reverse=True)
        top = candidates[:max_files]
        logger.info(
            "Selected %d relevant files (scanned %d)",
            len(top),
            len(all_files),
        )
        return top, keywords, len(all_files)

    def analyze(
        self,
        repo: RepoRef,
        issue: IssueDetails,
        repo_info: RepoInfo,
        force_refresh: bool = False,
    ) -> AnalysisResult:
        """Full sync pipeline: clone + find relevant files."""
        clone_path = self.clone_repository(
            repo,
            repo_info.clone_url,
            branch=repo_info.default_branch,
            force_refresh=force_refresh,
        )
        relevant, keywords, scanned = self.find_relevant_files(clone_path, issue)
        return AnalysisResult(
            clone_path=clone_path,
            repo_info=repo_info,
            relevant_files=relevant,
            total_files_scanned=scanned,
            keywords=keywords,
        )

    async def analyze_async(
        self,
        repo: RepoRef,
        issue: IssueDetails,
        repo_info: RepoInfo,
        force_refresh: bool = False,
    ) -> AnalysisResult:
        """Async wrapper around analyze."""
        return await asyncio.to_thread(
            self.analyze, repo, issue, repo_info, force_refresh
        )


_STOP_WORDS: frozenset[str] = frozenset(
    {
        "the",
        "and",
        "for",
        "that",
        "this",
        "with",
        "from",
        "have",
        "been",
        "when",
        "what",
        "your",
        "will",
        "should",
        "would",
        "could",
        "there",
        "their",
        "about",
        "into",
        "issue",
        "error",
        "bug",
        "fix",
        "please",
        "thanks",
        "help",
        "need",
        "using",
        "used",
        "also",
        "just",
        "like",
        "make",
        "does",
        "dont",
        "can't",
        "not",
        "are",
        "was",
        "were",
        "has",
        "had",
        "but",
        "can",
        "how",
        "why",
        "who",
        "which",
        "than",
        "then",
        "them",
        "they",
        "you",
        "our",
        "all",
        "any",
        "some",
        "more",
        "most",
        "other",
        "such",
        "only",
        "very",
        "here",
        "where",
        "after",
        "before",
        "because",
        "while",
        "during",
        "through",
        "between",
        "under",
        "over",
        "again",
        "further",
        "once",
    }
)
