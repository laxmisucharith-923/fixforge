"""GitHub API client for fetching issue and repository metadata."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from github import Auth, Github, GithubException

from fixforge.utils import RepoRef, get_env

logger = logging.getLogger(__name__)


@dataclass
class IssueDetails:
    """Normalized GitHub issue or pull request payload."""

    number: int
    title: str
    body: str
    state: str
    labels: list[str]
    html_url: str
    user: str
    repo_full_name: str
    kind: str = "issue"  # "issue" | "pull_request"
    merged: bool | None = None

    @property
    def full_description(self) -> str:
        """Combined title + body for LLM context."""
        kind_label = "Pull request" if self.kind == "pull_request" else "Issue"
        parts = [f"# {self.title}", ""]
        if self.body:
            parts.append(self.body)
        parts.extend(
            [
                "",
                "---",
                f"{kind_label} #{self.number} | State: {self.state}",
                f"Labels: {', '.join(self.labels) or 'none'}",
                f"Author: {self.user}",
                f"URL: {self.html_url}",
            ]
        )
        if self.kind == "pull_request" and self.merged is not None:
            parts.append(f"Merged: {self.merged}")
        return "\n".join(parts)


@dataclass
class RepoInfo:
    """Repository metadata from GitHub."""

    full_name: str
    description: str | None
    default_branch: str
    language: str | None
    clone_url: str
    html_url: str
    open_issues_count: int


class GitHubClient:
    """Thin wrapper around PyGithub for FixForge."""

    def __init__(self, token: str | None = None) -> None:
        self._token = token if token is not None else get_env("GITHUB_TOKEN", "")
        if self._token:
            self._github = Github(auth=Auth.Token(self._token))
            logger.info("GitHub client: authenticated")
        else:
            self._github = Github()
            logger.warning(
                "No GITHUB_TOKEN — unauthenticated API "
                "(60 req/hr for public repos)"
            )

    @property
    def token(self) -> str:
        """GitHub personal access token in use (empty if unauthenticated)."""
        return self._token

    def fetch_issue(self, repo: RepoRef, issue_number: int) -> IssueDetails:
        """Fetch an issue or pull request by number (synchronous)."""
        logger.info("Fetching #%s from %s", issue_number, repo.full_name)
        try:
            gh_repo = self._github.get_repo(repo.full_name)
            item = gh_repo.get_issue(issue_number)
        except GithubException as exc:
            status = getattr(exc, "status", None)
            if status == 404:
                raise ValueError(
                    f"#{issue_number} or repo '{repo.full_name}' not found. "
                    "Check the URL, number, and token permissions."
                ) from exc
            if status == 401:
                raise PermissionError(
                    "GitHub authentication failed. Check GITHUB_TOKEN."
                ) from exc
            if status == 403:
                raise PermissionError(
                    "GitHub API rate limit or forbidden. "
                    "Set GITHUB_TOKEN in .env for higher limits."
                ) from exc
            raise RuntimeError(f"GitHub API error: {exc}") from exc

        kind = "pull_request" if item.pull_request else "issue"
        merged: bool | None = None
        body = item.body or ""
        title = item.title or ""

        if kind == "pull_request":
            try:
                pr = gh_repo.get_pull(issue_number)
                body = pr.body or body
                merged = pr.merged
                title = pr.title or title
            except GithubException as exc:
                logger.warning("Could not load PR details: %s", exc)
            logger.info("Fetched pull request #%s (merged=%s)", issue_number, merged)

        return IssueDetails(
            number=item.number,
            title=title,
            body=body,
            state=item.state or "unknown",
            labels=[lbl.name for lbl in item.labels],
            html_url=item.html_url or "",
            user=item.user.login if item.user else "unknown",
            repo_full_name=repo.full_name,
            kind=kind,
            merged=merged,
        )

    def fetch_repo_info(self, repo: RepoRef) -> RepoInfo:
        """Fetch repository metadata (synchronous)."""
        logger.info("Fetching repo info for %s", repo.full_name)
        try:
            gh_repo = self._github.get_repo(repo.full_name)
        except GithubException as exc:
            raise RuntimeError(
                f"Failed to fetch repo '{repo.full_name}': {exc}"
            ) from exc

        return RepoInfo(
            full_name=gh_repo.full_name,
            description=gh_repo.description,
            default_branch=gh_repo.default_branch or "main",
            language=gh_repo.language,
            clone_url=gh_repo.clone_url or "",
            html_url=gh_repo.html_url or "",
            open_issues_count=gh_repo.open_issues_count,
        )

    async def fetch_issue_async(self, repo: RepoRef, issue_number: int) -> IssueDetails:
        """Async wrapper — runs blocking PyGithub in a thread."""
        return await asyncio.to_thread(self.fetch_issue, repo, issue_number)

    async def fetch_repo_info_async(self, repo: RepoRef) -> RepoInfo:
        """Async wrapper for fetch_repo_info."""
        return await asyncio.to_thread(self.fetch_repo_info, repo)

    def close(self) -> None:
        """Release GitHub client resources."""
        self._github.close()

    async def aclose(self) -> None:
        await asyncio.to_thread(self.close)
