"""
Check evaluation completion status for checkpoints and tasks.

This script inspects the results directory and reports which tasks are
finished / not finished per checkpoint.

Examples:
---------
    * simple status report
    python eval_status.py

    * status report with cutoff date (on or before)
    python eval_status.py --cutoff-date "*:2026-08-12"

    * status report with tasks regex
    python eval_status.py --tasks-regex "*mmlu*" "ifeval_ar" "poetry_*"

    * status report with skips
    python eval_status.py --skip "cpt" "cooldown"

    * status report with cutoff date and tasks regex
    python eval_status.py \
        --cutoff-date "*:2026-08-12" \
        --tasks-regex "*mmlu*" "ifeval_ar" "poetry_*" \
        --out-mkdown-filepath checkpoint_status.md

!DISCLAIMER: Whole script was written and still maintained by AI!
"""

from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser
from fnmatch import fnmatchcase
from functools import lru_cache
from lm_eval.tasks import TaskManager
import os

from helpers import *
from logger import green, red
from configs import (
    AR_TASKS,
    EN_TASKS,
    HF_HUB_CACHE_PATH,
    DEFAULT_BASE_CHECKPOINTS,
    DEFAULT_INSTRUCT_CHECKPOINTS,
    DEFAULT_THINK_CHECKPOINTS,
)


SFT_EVAL_ROOT = Path("/jais3/checkpoints/jais3_experiments/sft_eval")
_SFT_EVAL_SKIP_NAMES = {"config.json", "generation_config.json"}


def _mtime_yyyymmdd(path):
    return datetime.fromtimestamp(os.lstat(path).st_mtime).strftime("%Y-%m-%d")


@lru_cache(maxsize=1)
def _sft_eval_target_index():
    """Map realpath of each sft_eval symlink target -> the sft_eval entry."""
    mapping = {}
    try:
        for entry in SFT_EVAL_ROOT.iterdir():
            if entry.is_symlink():
                mapping[os.path.realpath(entry)] = entry
    except FileNotFoundError:
        pass
    return mapping


def _sft_eval_entry(ckpt_path):
    """Return the canonical ``sft_eval/<name>`` entry for this checkpoint.

    Uses the path component after ``sft_eval`` (not ``Path.name``, which is
    ``config.json`` or ``checkpoint_11831_to_hf`` after the symlink is
    followed). Falls back to a realpath match against sft_eval symlinks.
    """
    parts = ckpt_path.parts
    if "sft_eval" in parts:
        idx = parts.index("sft_eval")
        if idx + 1 < len(parts):
            return SFT_EVAL_ROOT / parts[idx + 1]

    if ckpt_path.name not in _SFT_EVAL_SKIP_NAMES:
        candidate = SFT_EVAL_ROOT / ckpt_path.name
        if candidate.is_symlink() or candidate.exists():
            return candidate

    real = os.path.realpath(ckpt_path)
    if os.path.isfile(real):
        real = os.path.dirname(real)
    return _sft_eval_target_index().get(real)


def get_checkpoint_created_date(ckpt, hf_hub_cache_path=HF_HUB_CACHE_PATH):
    """Return the checkpoint creation date (YYYY-MM-DD).

    For ``sft_eval`` checkpoints, always read the date from
    ``/jais3/checkpoints/jais3_experiments/sft_eval/<name>`` via ``os.lstat``
    (the symlink/entry mtime there, not the CPT target / ``config.json``).
    """
    ckpt_path = Path(str(ckpt).rstrip("/"))

    sft_path = _sft_eval_entry(ckpt_path)
    if sft_path is not None:
        try:
            return _mtime_yyyymmdd(sft_path)
        except FileNotFoundError:
            pass

    try:
        path = ckpt_path.parent if ckpt_path.is_file() else ckpt_path
        return _mtime_yyyymmdd(path)
    except FileNotFoundError:
        pass

    cache_name = str(ckpt).replace("/", "--")
    snap_dir = Path(hf_hub_cache_path) / f"models--{cache_name}" / "snapshots"
    if snap_dir.is_dir():
        subdirs = [p for p in snap_dir.iterdir() if p.is_dir()]
        if subdirs:
            latest = max(subdirs, key=lambda p: p.stat().st_mtime)
            return _mtime_yyyymmdd(latest)
    return "-"


def parse_cutoff_bound(value):
    """Parse a single bound: ``*`` / empty -> None, else YYYY-MM-DD."""
    value = (value or "").strip()
    if not value or value == "*":
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def parse_cutoff_date(value):
    """Parse ``START:END`` cutoff range.

    ``*`` means unbounded on that side. Default ``*:*`` keeps all dates.
    Both ends are inclusive when set.

    * ``*:2026-08-12`` — created on or before 2026-08-12
    * ``2026-08-12:*`` — created on or after 2026-08-12
    * ``2026-08-01:2026-08-12`` — inclusive window
    * ``2026-08-12`` — same as ``*:2026-08-12`` (on or before)
    """
    if value is None:
        return (None, None)
    raw = str(value).strip()
    if ":" not in raw:
        end = parse_cutoff_bound(raw)
        return (None, end)

    start_raw, end_raw = raw.split(":", 1)
    start = parse_cutoff_bound(start_raw)
    end = parse_cutoff_bound(end_raw)
    if start is not None and end is not None and start > end:
        raise ValueError(
            f"Invalid --cutoff-date `{raw}`: start date is after end date."
        )
    return (start, end)


def checkpoint_passes_cutoff(created_date_str, cutoff_range):
    """Keep checkpoints whose created date falls in [start, end] (open bounds allowed)."""
    start, end = cutoff_range
    if start is None and end is None:
        return True
    if not created_date_str or created_date_str == "-":
        return False
    created = datetime.strptime(created_date_str, "%Y-%m-%d").date()
    if start is not None and created < start:
        return False
    if end is not None and created > end:
        return False
    return True


def task_matches_patterns(task, tasks_patterns):
    """Return whether a task matches any fnmatch pattern."""
    return any(fnmatchcase(task, pattern) for pattern in tasks_patterns)


def get_applicable_tasks_for_checkpoint(ckpt_type, tasks_patterns=("*",)):
    """Return all valid tasks for a given checkpoint type, filtered by fnmatch patterns."""
    all_tasks = {**EN_TASKS, **AR_TASKS}
    return [
        task
        for task, task_types in all_tasks.items()
        if ckpt_type in task_types and task_matches_patterns(task, tasks_patterns)
    ]


def checkpoint_matches_skip(checkpoint, skip_patterns):
    """Return whether a checkpoint matches a case-insensitive skip selector."""
    checkpoint = str(checkpoint)
    path = Path(checkpoint)
    candidates = {checkpoint, path.name, *path.parts}
    candidates = {candidate.casefold() for candidate in candidates}

    return any(
        fnmatchcase(candidate, pattern.casefold())
        for pattern in skip_patterns
        for candidate in candidates
    )


def get_status_checkpoints(skip_patterns):
    """Load default checkpoints, excluding skipped roots before expansion."""
    checkpoint_groups = (
        ("base", DEFAULT_BASE_CHECKPOINTS),
        ("instruct", DEFAULT_INSTRUCT_CHECKPOINTS),
        ("think", DEFAULT_THINK_CHECKPOINTS),
    )
    selected = {}
    for ckpt_type, configured_checkpoints in checkpoint_groups:
        selected[ckpt_type] = [
            checkpoint
            for checkpoint in configured_checkpoints
            if not checkpoint_matches_skip(checkpoint, skip_patterns)
        ]

    checkpoints = get_checkpoints(
        selected["base"] or None,
        selected["instruct"] or None,
        selected["think"] or None,
    )
    checkpoints = filter_checkpoints(checkpoints)

    # Also support selectors for individual checkpoints inside an expanded root.
    return [
        (ckpt_type, checkpoint)
        for ckpt_type, checkpoint in checkpoints
        if not checkpoint_matches_skip(checkpoint, skip_patterns)
    ]


def format_row(columns, widths):
    return " | ".join(col.ljust(width) for col, width in zip(columns, widths))


def build_table(rows, headers):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))

    sep = "-+-".join("-" * w for w in widths)
    out = [
        format_row(headers, widths),
        sep,
    ]
    for row in rows:
        out.append(format_row(row, widths))
    return "\n".join(out)


def to_markdown_table(rows, headers):
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def main(args):
    # get lm-evaluation-harness task manager
    print("Loading lm-evaluation-harness task manager...")
    task_manager = TaskManager()
    # Use default checkpoints across all checkpoint types, except explicitly
    # skipped checkpoint roots/groups.
    checkpoints = get_status_checkpoints(args.skip)

    if not checkpoints:
        print(red("No checkpoints left after filtering."))
        return

    cutoff_range = parse_cutoff_date(args.cutoff_date)
    summary_rows = []
    checkpoint_reports = []

    for ckpt_type, ckpt in checkpoints:
        created_date = get_checkpoint_created_date(ckpt)
        if not checkpoint_passes_cutoff(created_date, cutoff_range):
            continue

        ckpt_results_path = get_ckpt_results_path(ckpt_type, ckpt, args.results_path)
        ckpt_results_path.mkdir(parents=True, exist_ok=True)

        applicable_tasks = get_applicable_tasks_for_checkpoint(
            ckpt_type, tasks_patterns=args.tasks_regex
        )
        if not applicable_tasks:
            continue

        remaining_tasks = set(
            get_remaining_tasks(ckpt_type, ckpt, task_manager, ["all"], ckpt_results_path)
        )

        finished_tasks = [t for t in applicable_tasks if t not in remaining_tasks]
        pending_tasks = [t for t in applicable_tasks if t in remaining_tasks]
        progress = f"{len(finished_tasks)}/{len(applicable_tasks)}"
        pct = f"{(100.0 * len(finished_tasks) / len(applicable_tasks)):.1f}%"
        model_name = ckpt_results_path.name

        summary_rows.append(
            [
                ckpt_type,
                model_name,
                created_date,
                progress,
                pct,
                ", ".join(f"`{task}`" for task in pending_tasks) if pending_tasks else "-",
            ]
        )
        checkpoint_reports.append(
            {
                "type": ckpt_type,
                "model_name": model_name,
                "created_date": created_date,
                "finished_tasks": finished_tasks,
                "pending_tasks": pending_tasks,
                "progress": progress,
                "pct": pct,
            }
        )

    if not checkpoint_reports:
        patterns = ", ".join(args.tasks_regex) if args.tasks_regex else "none"
        print(
            red(
                "No checkpoints left after applying "
                f"tasks-regex `{patterns}` and cutoff-date `{args.cutoff_date}`."
            )
        )
        return

    total_finished = sum(len(r["finished_tasks"]) for r in checkpoint_reports)
    total_pending = sum(len(r["pending_tasks"]) for r in checkpoint_reports)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    task_stats = {}
    for report in checkpoint_reports:
        model_name = report["model_name"]
        for task in report["finished_tasks"]:
            task_stats.setdefault(task, {"done_models": [], "pending_models": []})
            task_stats[task]["done_models"].append(model_name)
        for task in report["pending_tasks"]:
            task_stats.setdefault(task, {"done_models": [], "pending_models": []})
            task_stats[task]["pending_models"].append(model_name)

    # Per checkpoint type: count checkpoints (not individual task completions).
    type_ckpt_stats = {}
    for report in checkpoint_reports:
        ckpt_type = report["type"]
        type_ckpt_stats.setdefault(ckpt_type, {"fully_done": 0, "total": 0})
        type_ckpt_stats[ckpt_type]["total"] += 1
        if not report["pending_tasks"]:
            type_ckpt_stats[ckpt_type]["fully_done"] += 1

    checkpoint_type_summary_rows = []
    for ckpt_type in sorted(type_ckpt_stats.keys()):
        fully_done = type_ckpt_stats[ckpt_type]["fully_done"]
        total_ckpts = type_ckpt_stats[ckpt_type]["total"]
        not_fully_done = total_ckpts - fully_done
        done_pct = (
            f"{(100.0 * fully_done / total_ckpts):.1f}%" if total_ckpts else "0.0%"
        )
        checkpoint_type_summary_rows.append(
            [
                ckpt_type,
                f"{fully_done}/{total_ckpts}",
                done_pct,
                str(not_fully_done),
            ]
        )

    task_summary_rows = []
    for task in sorted(task_stats.keys()):
        done = len(task_stats[task]["done_models"])
        pending = len(task_stats[task]["pending_models"])
        total = done + pending
        done_pct = f"{(100.0 * done / total):.1f}%" if total else "0.0%"
        task_summary_rows.append([task, f"{done}/{total}", done_pct, str(pending)])

    model_type_map = {report["model_name"]: report["type"] for report in checkpoint_reports}

    completely_done_by_type_rows = []
    for ckpt_type in ("base", "instruct", "think"):
        s = type_ckpt_stats.get(ckpt_type, {"fully_done": 0, "total": 0})
        completely_done_by_type_rows.append(
            [ckpt_type, f"{s['fully_done']}/{s['total']}"]
        )

    report_lines = [
        "# Checkpoint Evaluation Status",
        "",
        "## Overview",
        "",
        f"- Date: **{generated_at}**",
        f"- Results path: `{args.results_path}`",
        f"- Skipped checkpoint selectors: **{', '.join(args.skip) if args.skip else 'none'}**",
        f"- Tasks regex (fnmatch): **{', '.join(f'`{p}`' for p in args.tasks_regex) if args.tasks_regex else 'none'}**",
        f"- Cutoff date (START:END): **`{args.cutoff_date}`**",
        f"- Total checkpoints: **{len(checkpoint_reports)}**",
        f"- Total finished tasks: **{total_finished}**",
        f"- Total not finished tasks: **{total_pending}**",
        "",
        "## Summary",
        "",
        "### Completely done checkpoints by type",
        "",
        to_markdown_table(
            completely_done_by_type_rows,
            headers=["checkpoint type", "done/total"],
        ),
        "",
        "### Checkpoints",
        "",
        to_markdown_table(
            summary_rows,
            headers=[
                "type",
                "checkpoint name",
                "created",
                "done/total",
                "done_%",
                "not_finished tasks",
            ],
        ),
        "",
        "### Checkpoint Type Summary",
        "",
        to_markdown_table(
            checkpoint_type_summary_rows,
            headers=[
                "checkpoint type",
                "done/total",
                "done_%",
                "not_finished",
            ],
        ),
        "",
        "### Task Summary",
        "",
        to_markdown_table(
            task_summary_rows,
            headers=["task", "done/total checkpoints", "done_%", "not_finished"],
        ),
        "",
        # "## Per-Task Details",
        # "",
    ]

    # for task in sorted(task_stats.keys()):
    #     pending_models = sorted(task_stats[task]["pending_models"])
    #     report_lines.extend([f"## `{task}`", ""])
    #     if pending_models:
    #         for model_name in pending_models:
    #             ckpt_type = model_type_map.get(model_name, "unknown")
    #             report_lines.append(f"- {ckpt_type}: `{model_name}`")
    #     else:
    #         report_lines.append("- All checkpoints done")
    #     report_lines.append("")

    output_file = args.out_mkdown_filepath
    output_file.write_text("\n".join(report_lines), encoding="utf-8")
    print(green(f"Checkpoint status report written to: `{output_file}`"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--results-path",
        type=Path,
        default=Path("/jais3/eval/results_optimized"),
        help="Root path where eval results are stored (default: `/jais3/eval/results_optimized`).",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=["cpt", "cooldown"],
        metavar="CHECKPOINT",
        help=(
            "Checkpoint groups or names to exclude. Defaults to: cpt cooldown. "
            "Pass --skip with no values to check everything."
        ),
    )
    parser.add_argument(
        "--tasks-regex",
        nargs="*",
        default=["*"],
        metavar="PATTERN",
        help=(
            "One or more fnmatch patterns to select which tasks to include "
            '(default: "*" = all tasks). A task matches if any pattern matches. '
            'Example: --tasks-regex "*mmlu*" "ifeval_ar" "poetry_*".'
        ),
    )
    parser.add_argument(
        "--cutoff-date",
        type=str,
        default="*:*",
        metavar="START:END",
        help=(
            "Inclusive created-date window as START:END (YYYY-MM-DD). "
            "Use * for an open bound. Default: *:* (any date). "
            "A single date is treated as *:DATE (on/before). "
            "Examples: *:2026-08-12 or 2026-08-12 (on/before), "
            "2026-08-12:* (on/after), 2026-08-01:2026-08-12 (window)."
        ),
    )
    parser.add_argument(
        "--out-mkdown-filepath",
        type=Path,
        default=Path(__file__).resolve().parent / "checkpoint_status.md",
        help="Filepath to write the markdown report to."
    )
    args = parser.parse_args()
    args.results_path = Path(args.results_path)
    args.out_mkdown_filepath = Path(args.out_mkdown_filepath)
    args.out_mkdown_filepath.parent.mkdir(parents=True, exist_ok=True)
    main(args)
