"""FixForge CLI — AI-powered GitHub issue fix suggestions."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import sys
from pathlib import Path

from fixforge.github_client import GitHubClient
from fixforge.llm_fixer import LLMFixer
from fixforge.repo_analyzer import RepoAnalyzer
from fixforge.utils import load_environment, parse_repo_url, setup_logging
from fixforge.validator import FixLoopResult, PatchValidator, ValidationResult

logger = logging.getLogger(__name__)

BANNER = r"""
 _____ _        _____
|  ___| | _____|  ___| __ _ _ __ ___
| |_  | |/ / _ \ |_ / _` | '__/ _ \
|  _| |   <  __/  _| (_| | | |  __/
|_|   |_|\_\___|_|  \__,_|_|  \___|

  AI-powered GitHub issue fixer (MVP)
"""


def build_parser() -> argparse.ArgumentParser:
    """Configure CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="fixforge",
        description="Analyze a GitHub issue and suggest a code fix (manual apply).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m fixforge.main "
            "https://github.com/owner/repo 42\n"
            "  python -m fixforge.main owner/repo 42 --provider groq\n"
            "  python -m fixforge.main owner/repo 42 --manual\n"
        ),
    )
    parser.add_argument(
        "repo",
        help="GitHub repo URL or owner/repo",
    )
    parser.add_argument(
        "issue",
        type=int,
        help="Issue or pull request number",
    )
    parser.add_argument(
        "--provider",
        choices=["groq", "deepseek", "anthropic", "openai"],
        help="LLM provider (default: LLM_PROVIDER env or groq)",
    )
    parser.add_argument(
        "--model",
        help="LLM model name (default: provider-specific)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-clone repository even if cached in temp_repos/",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=7,
        help="Max relevant files to send to LLM (default: 7)",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=Path("./temp_repos"),
        help="Directory for cloned repos (default: ./temp_repos)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM step (fetch + analyze only, for testing)",
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Suggest fix only — skip auto-apply, ruff, and pytest validation loop",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        metavar="N",
        help="Max LLM + validation iterations (default: 3)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write full fix report to this file",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Less console output (errors still shown)",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser


def _format_validation_attempt(v: ValidationResult) -> list[str]:
    lines = [
        f"  Attempt {v.attempt}: "
        f"{'PASS' if v.success else 'FAIL'} | "
        f"patch_applied={v.patch_applied}",
    ]
    if v.changed_files:
        lines.append(f"    Changed: {', '.join(v.changed_files)}")
    if not v.patch_applied:
        lines.append(f"    Apply: {v.apply_output[:200]}...")
    elif not v.ruff_passed:
        lines.append("    Ruff: failed")
    elif v.pytest_ran and v.pytest_passed is False:
        lines.append("    Pytest: failed")
    return lines


def format_report(
    issue_title: str,
    issue_url: str,
    repo_name: str,
    relevant_paths: list[str],
    fix_text: str,
    cost_summary: str,
    clone_path: Path,
    *,
    loop_result: FixLoopResult | None = None,
    manual_mode: bool = False,
) -> str:
    """Build the final human-readable report."""
    lines = [
        "=" * 72,
        "FIXFORGE — FIX REPORT",
        "=" * 72,
        "",
        f"Repository : {repo_name}",
        f"Issue      : {issue_title}",
        f"URL        : {issue_url}",
        f"Clone path : {clone_path.resolve()}",
        "",
        "Relevant files analyzed:",
    ]
    for path in relevant_paths:
        lines.append(f"  - {path}")

    if loop_result is not None:
        status = "SUCCESS" if loop_result.success else "FAILED"
        lines.extend(
            [
                "",
                "-" * 72,
                "VALIDATION LOOP",
                "-" * 72,
                f"Result        : {status}",
                f"Attempts used : {loop_result.attempts_used}",
                "",
                "Attempt history:",
            ]
        )
        for v in loop_result.validation_history:
            lines.extend(_format_validation_attempt(v))

        lines.extend(
            [
                "",
                "-" * 72,
                "FINAL PATCH (unified diff)",
                "-" * 72,
                "",
                loop_result.final_diff or "(no patch extracted)",
            ]
        )

    lines.extend(
        [
            "",
            "-" * 72,
            "LLM OUTPUT",
            "-" * 72,
            "",
            fix_text,
            "",
            "-" * 72,
            cost_summary,
            "-" * 72,
        ]
    )

    if manual_mode:
        lines.append("MODE: Manual review — patch NOT auto-applied to your clone.")
    elif loop_result and loop_result.success:
        lines.append(
            "Validated in isolated work copy under temp_repos/_validate/. "
            "Review and apply the final patch manually to your project."
        )
    else:
        lines.append(
            "Validation did not fully pass. "
            "Review LLM output and errors before applying."
        )

    lines.append("NOTE: No commits or PRs were created.")
    lines.append("=" * 72)
    return "\n".join(lines)


def _configure_stdout_encoding() -> None:
    """Avoid Windows cp1252 crashes on emoji/unicode in issue titles."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            with contextlib.suppress(OSError, ValueError):
                reconfigure(encoding="utf-8", errors="replace")


async def run_pipeline(args: argparse.Namespace) -> int:
    """Execute the full FixForge pipeline."""
    _configure_stdout_encoding()
    load_environment()
    setup_logging(args.log_level)

    if not args.quiet:
        print(BANNER)

    try:
        repo_ref = parse_repo_url(args.repo)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    github = GitHubClient()
    analyzer = RepoAnalyzer(
        temp_root=args.temp_dir,
        github_token=github.token,
    )

    try:
        logger.info("Step 1/4: Fetching #%s", args.issue)
        issue = await github.fetch_issue_async(repo_ref, args.issue)
        repo_info = await github.fetch_repo_info_async(repo_ref)

        if not args.quiet:
            kind = "PR" if issue.kind == "pull_request" else "Issue"
            print(f"\n{kind}: #{issue.number} — {issue.title}")
            print(f"State: {issue.state} | Labels: {', '.join(issue.labels) or 'none'}")
            if issue.merged is not None:
                print(f"Merged: {issue.merged}")
            print(f"URL:   {issue.html_url}\n")

        logger.info("Step 2/4: Cloning and analyzing repository")
        analysis = await analyzer.analyze_async(
            repo_ref,
            issue,
            repo_info,
            force_refresh=args.refresh,
        )
        analysis.relevant_files = analysis.relevant_files[: args.max_files]

        if not args.quiet:
            print(f"Scanned {analysis.total_files_scanned} files")
            print(f"Keywords: {', '.join(analysis.keywords[:12])}...")
            print("Top relevant files:")
            for rf in analysis.relevant_files:
                print(f"  [{rf.score:5.1f}] {rf.path} ({rf.size_bytes} bytes)")

        if args.no_llm:
            print("\n--no-llm set; skipping fix generation.")
            return 0

        fixer = LLMFixer(provider=args.provider, model=args.model)
        loop_result: FixLoopResult | None = None
        fix_text: str

        if args.manual:
            logger.info("Step 3/4: Generating fix (manual mode — no validation)")
            fix_result = await fixer.generate_fix_async(
                issue,
                analysis.relevant_files,
                repo_info.language,
            )
            fix_text = fix_result.raw_response
            cost_summary = fix_result.cost_tracker.summary()
        else:
            logger.info(
                "Step 3/4: Generating fix with validation loop (max %d iterations)",
                args.max_iterations,
            )
            validate_root = args.temp_dir / "_validate"
            validator = PatchValidator(
                source_repo=analysis.clone_path,
                work_root=validate_root,
                issue_number=issue.number,
            )
            loop_result = await fixer.run_validated_fix_loop_async(
                issue,
                analysis.relevant_files,
                validator,
                repo_info.language,
                max_iterations=args.max_iterations,
            )
            fix_text = loop_result.final_raw_response
            cost_summary = loop_result.cost_summary

            if not args.quiet:
                status = "SUCCESS" if loop_result.success else "FAILED"
                n = loop_result.attempts_used
                print(f"\nValidation: {status} after {n} attempt(s)")

        logger.info("Step 4/4: Building report")
        report = format_report(
            issue_title=issue.title,
            issue_url=issue.html_url,
            repo_name=issue.repo_full_name,
            relevant_paths=[f.path for f in analysis.relevant_files],
            fix_text=fix_text,
            cost_summary=cost_summary,
            clone_path=analysis.clone_path,
            loop_result=loop_result,
            manual_mode=args.manual,
        )

        if args.output:
            args.output.write_text(report, encoding="utf-8")
            logger.info("Report written to %s", args.output)

        print("\n" + report)
        if loop_result and not loop_result.success:
            return 2
        return 0

    except (PermissionError, ValueError, RuntimeError) as exc:
        logger.error("FixForge failed: %s", exc)
        if args.quiet:
            print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        await github.aclose()


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()
    exit_code = asyncio.run(run_pipeline(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
