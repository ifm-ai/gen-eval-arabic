#!/usr/bin/env python3
"""Add recursively flattened weighted averages to lm-eval result groups.

For every ``results_*.json`` directly inside the supplied folder, this script finds
non-empty groups in ``group_subtasks`` and writes derived numeric metrics to:

    results[GROUP_NAME]["weighted_average_keys"]

Every group is flattened to its leaf tasks. A metric is included only when it
is present as a finite numeric scalar on every leaf. Each value is weighted by
the corresponding leaf's ``sample_len``. Existing top-level result metrics and
the JSON's top-level ``groups`` object are never changed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any


OUTPUT_KEY = "weighted_average_keys"
EXCLUDED_KEYS = {
    "name",
    "alias",
    "sample_len",
    "metadata",
    "meta_data",
    OUTPUT_KEY,
}


class GroupResolutionError(ValueError):
    """Raised when a group cannot be flattened safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "folder",
        type=Path,
        help="Folder containing the results_*.json files to process",
    )
    return parser.parse_args()


def warning(path: Path, message: str) -> None:
    print(f"WARNING [{path}]: {message}", file=sys.stderr)


def error(path: Path, message: str) -> None:
    print(f"ERROR [{path}]: {message}", file=sys.stderr)


def is_stderr_key(key: str) -> bool:
    """Recognize both ``metric_stderr`` and ``metric_stderr,filter`` forms."""
    metric_name = key.split(",", 1)[0]
    return key.endswith("_stderr") or metric_name.endswith("_stderr")


def is_finite_number(value: Any) -> bool:
    """Return whether a value is a finite JSON number or numeric string."""
    # bool is an int subclass, but averaging true/false fields is unintended.
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float, str)):
        return False
    if isinstance(value, str) and not value.strip():
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def format_names(names: list[str], limit: int = 5) -> str:
    visible = names[:limit]
    suffix = "" if len(names) <= limit else f"; ... and {len(names) - limit} more"
    return "; ".join(visible) + suffix


def flatten_group(
    group_name: str,
    group_subtasks: dict[str, Any],
    active_path: tuple[str, ...] = (),
) -> list[str]:
    if group_name in active_path:
        cycle_start = active_path.index(group_name)
        cycle = (*active_path[cycle_start:], group_name)
        raise GroupResolutionError(f"cycle detected: {' -> '.join(cycle)}")

    children = group_subtasks.get(group_name, [])
    if children is None:
        children = []
    if not isinstance(children, list) or not all(
        isinstance(child, str) for child in children
    ):
        raise GroupResolutionError("group_subtasks entry must be a list of strings")
    if not children:
        return [group_name]

    leaves: list[str] = []
    next_path = (*active_path, group_name)
    for child in children:
        leaves.extend(flatten_group(child, group_subtasks, next_path))

    seen: set[str] = set()
    duplicates: list[str] = []
    for leaf in leaves:
        if leaf in seen and leaf not in duplicates:
            duplicates.append(leaf)
        seen.add(leaf)
    if duplicates:
        raise GroupResolutionError(
            "duplicate leaves reached through multiple branches: "
            f"{format_names(duplicates)}"
        )
    return leaves


def validate_leaf_results(
    *,
    path: Path,
    group_name: str,
    leaves: list[str],
    results: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[float]] | None:
    missing_results = [leaf for leaf in leaves if leaf not in results]
    if missing_results:
        warning(
            path,
            f"group {group_name!r} skipped; missing leaf results: "
            f"{format_names(missing_results)}",
        )
        return None

    invalid_results = [leaf for leaf in leaves if not isinstance(results[leaf], dict)]
    if invalid_results:
        warning(
            path,
            f"group {group_name!r} skipped; leaf results are not objects: "
            f"{format_names(invalid_results)}",
        )
        return None

    leaf_results: list[dict[str, Any]] = [results[leaf] for leaf in leaves]
    invalid_weights = [
        leaf
        for leaf, leaf_result in zip(leaves, leaf_results)
        if not is_finite_number(leaf_result.get("sample_len"))
        or float(leaf_result["sample_len"]) <= 0
    ]
    if invalid_weights:
        warning(
            path,
            f"group {group_name!r} skipped; leaves have missing, nonnumeric, "
            f"non-finite, or non-positive sample_len: {format_names(invalid_weights)}",
        )
        return None

    weights = [float(leaf_result["sample_len"]) for leaf_result in leaf_results]
    return leaf_results, weights


def calculate_weighted_keys(
    *,
    path: Path,
    group_name: str,
    leaves: list[str],
    leaf_results: list[dict[str, Any]],
    weights: list[float],
) -> dict[str, float]:
    all_keys = set().union(*(leaf_result.keys() for leaf_result in leaf_results))
    common_keys = set.intersection(
        *(set(leaf_result.keys()) for leaf_result in leaf_results)
    )

    eligible_union = {
        key
        for key in all_keys
        if isinstance(key, str)
        and key not in EXCLUDED_KEYS
        and not is_stderr_key(key)
    }
    missing_on_some_leaves = sorted(eligible_union - common_keys)
    for key in missing_on_some_leaves:
        missing = [
            leaf
            for leaf, leaf_result in zip(leaves, leaf_results)
            if key not in leaf_result
        ]
        warning(
            path,
            f"group {group_name!r}: key {key!r} skipped because it is missing "
            f"from {len(missing)}/{len(leaves)} leaves: {format_names(missing)}",
        )

    weighted: dict[str, float] = {}
    total_weight = math.fsum(weights)
    for key in sorted(common_keys):
        if key in EXCLUDED_KEYS or is_stderr_key(key):
            continue

        values = [leaf_result[key] for leaf_result in leaf_results]
        invalid_value_leaves = [
            leaf
            for leaf, value in zip(leaves, values)
            if not is_finite_number(value)
        ]
        if invalid_value_leaves:
            warning(
                path,
                f"group {group_name!r}: key {key!r} skipped because values are "
                f"not finite numeric scalars on: {format_names(invalid_value_leaves)}",
            )
            continue

        numerator = math.fsum(
            float(value) * weight for value, weight in zip(values, weights)
        )
        aggregate = numerator / total_weight
        if not math.isfinite(aggregate):
            warning(
                path,
                f"group {group_name!r}: key {key!r} produced a non-finite "
                "weighted average and was skipped",
            )
            continue
        weighted[key] = aggregate
    return weighted


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    original_mode = stat.S_IMODE(path.stat().st_mode)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, original_mode)
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def process_file(path: Path) -> tuple[int, int, bool]:
    """Return (groups_updated, groups_skipped, file_had_errors)."""
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        error(path, f"could not read JSON: {exc}")
        return 0, 0, True

    if not isinstance(payload, dict):
        error(path, "top-level JSON value must be an object")
        return 0, 0, True
    results = payload.get("results")
    group_subtasks = payload.get("group_subtasks")
    if not isinstance(results, dict):
        error(path, "missing or invalid top-level 'results' object")
        return 0, 0, True
    if not isinstance(group_subtasks, dict):
        error(path, "missing or invalid top-level 'group_subtasks' object")
        return 0, 0, True

    groups: list[str] = []
    malformed_group_entries: list[str] = []
    for name, children in group_subtasks.items():
        if not isinstance(name, str):
            continue
        if children is None or children == []:
            continue
        if not isinstance(children, list):
            malformed_group_entries.append(name)
            continue
        if children:
            groups.append(name)
    for name in malformed_group_entries:
        error(path, f"group {name!r} has a non-list group_subtasks entry; skipped")

    groups_updated = 0
    groups_skipped = len(malformed_group_entries)
    file_had_errors = bool(malformed_group_entries)
    changed = False

    for group_name in sorted(groups):
        try:
            leaves = flatten_group(group_name, group_subtasks)
        except GroupResolutionError as exc:
            error(path, f"group {group_name!r} skipped; {exc}")
            groups_skipped += 1
            file_had_errors = True
            continue

        validated = validate_leaf_results(
            path=path,
            group_name=group_name,
            leaves=leaves,
            results=results,
        )
        if validated is None:
            groups_skipped += 1
            continue
        leaf_results, weights = validated
        weighted = calculate_weighted_keys(
            path=path,
            group_name=group_name,
            leaves=leaves,
            leaf_results=leaf_results,
            weights=weights,
        )

        group_result = results.get(group_name)
        if group_result is None:
            warning(path, f"creating missing results object for group {group_name!r}")
            group_result = {"name": group_name, "alias": group_name}
            results[group_name] = group_result
            changed = True
        elif not isinstance(group_result, dict):
            error(path, f"group {group_name!r} result is not an object; skipped")
            groups_skipped += 1
            file_had_errors = True
            continue

        if group_result.get(OUTPUT_KEY) != weighted:
            if OUTPUT_KEY in group_result and not isinstance(group_result[OUTPUT_KEY], dict):
                warning(path, f"replacing non-object {OUTPUT_KEY!r} for group {group_name!r}")
            group_result[OUTPUT_KEY] = weighted
            changed = True
        groups_updated += 1

    if changed:
        try:
            atomic_write_json(path, payload)
        except (OSError, TypeError, ValueError) as exc:
            error(path, f"could not write updated JSON: {exc}")
            return groups_updated, groups_skipped, True
        print(
            f"UPDATED [{path}]: {groups_updated} groups processed, "
            f"{groups_skipped} skipped"
        )
    else:
        print(
            f"UNCHANGED [{path}]: {groups_updated} groups processed, "
            f"{groups_skipped} skipped"
        )
    return groups_updated, groups_skipped, file_had_errors


def main() -> int:
    args = parse_args()
    folder = args.folder.expanduser()
    if not folder.exists():
        error(folder, "folder does not exist")
        return 2
    if not folder.is_dir():
        error(folder, "path is not a directory")
        return 2

    files = sorted(
        path
        for path in folder.glob("results_*.json")
        if path.is_file() and not path.is_symlink()
    )
    if not files:
        warning(folder, "no files matching results_*.json were found")
        return 0

    total_groups = 0
    total_skipped = 0
    files_with_errors = 0
    for path in files:
        processed, skipped, had_errors = process_file(path)
        total_groups += processed
        total_skipped += skipped
        files_with_errors += int(had_errors)

    print(
        f"SUMMARY: {len(files)} files, {total_groups} groups processed, "
        f"{total_skipped} groups skipped, {files_with_errors} files with errors"
    )
    return 1 if files_with_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
