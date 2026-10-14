import os
import re
import json
import shutil
from pathlib import Path
from copy import deepcopy
from argparse import ArgumentParser

from configs import JUDGE_PREFIX, DEFAULT_JUDGE




def read_json_file(filepath):
    if not filepath.exists():
        raise FileNotFoundError(f"File `{filepath}` is not found!!")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        raise ValueError(f"File `{filepath}` is corrupt!!")


#### George: Added to keep merged filenames from growing on every re-merge
def parse_merged_stem(stem):
    """Strip trailing `_merged` / `_merged_<N>` suffixes from a stem.

    Returns (base_stem, total_count) where total_count is the accumulated
    number of merges, so the output can be named `{base}_merged_{count+1}`
    instead of appending another `_merged` each run.
    """
    count = 0
    while True:
        m = re.search(r"_merged_(\d+)$", stem)
        if m:
            count += int(m.group(1))
            stem = stem[: m.start()]
            continue
        if stem.endswith("_merged"):
            count += 1
            stem = stem[: -len("_merged")]
            continue
        break
    return stem, count
########


#### George: Added to sort results files by the timestamp inside their name so the latest run of each benchmark wins
RESULTS_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}(?:\.\d+)?")


def sort_results_key(filepath):
    """Sort key: timestamp embedded in the filename, then `_judged` last.

    `deep_merge_prefer_second` makes the *last* file in the fold win for scalar
    leaves (e.g. `acc,none`). Sorting by the whole stem put prefixed snapshots
    like `results_recomputed_spbleu_chrfpp_<ts>_*` after every
    `results_<ts>.json` ('r' > '2'), and ranking all `_judged` files after all
    others let an old judged snapshot clobber newer re-runs. Derived files
    (`_merged_*`, `_judged`, `recomputed_*`) keep the timestamp of the run they
    came from, while a re-run gets a new one, so ordering by that timestamp
    makes the latest run win. On equal timestamps `_judged` goes last so the
    judge's scores win over the raw run they were computed from.
    """
    stem = filepath.stem
    m = RESULTS_TIMESTAMP_RE.search(stem)
    timestamp = m.group(0).split(".")[0] if m else ""
    return (timestamp, stem.endswith("_judged"), stem)
########


def write_json_file(data, out_filepath):
    with open(out_filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Results were written successfully in: `{out_filepath}`")


def deep_merge_prefer_second(a, b):
    """Recursively merge two JSON objects (a & b) into one"""
    if isinstance(a, dict) and isinstance(b, dict):
        merged = deepcopy(a)
        for key, b_val in b.items():
            if key in merged:
                merged[key] = deep_merge_prefer_second(merged[key], b_val)
            else:
                merged[key] = deepcopy(b_val)
        return merged

    if isinstance(a, list) and isinstance(b, list):
        merged = deepcopy(a)
        for item in b:
            if item not in merged:
                merged.append(deepcopy(item))
        return merged

    return deepcopy(b)


def remerge_from_old_results(old_results_path):
    """Re-merge from old_results/ if a previous run failed after moving files."""
    old_results_filepaths = sorted(
        old_results_path.glob("results_*.json"), key=sort_results_key
    )
    if len(old_results_filepaths) == 0:
        raise FileNotFoundError(
            f"No files matching 'results_*.json' found in {old_results_path}"
        )
    print(f"\t\tRe-merging {len(old_results_filepaths)}...")
    merged = read_json_file(old_results_filepaths[0])
    for result_file in old_results_filepaths[1:]:
        merged = deep_merge_prefer_second(merged, read_json_file(result_file))
    return merged


def main(args):
    if args.results_path:
        args.model_results_path = [
            args.results_path / folder
            for folder in os.listdir(args.results_path)
        ]

    #### George: Added to process newest folders first — same output, but the
    # latest results (the ones we usually care about) get merged before we spend
    # time on stale folders. Order by the most recent results_*.json inside each
    # folder, falling back to the folder's own mtime; unreadable paths sort last.
    def _folder_recency(path):
        try:
            result_files = list(path.glob("results_*.json"))
            if result_files:
                return max(f.stat().st_mtime for f in result_files)
            return path.stat().st_mtime
        except OSError:
            return float("-inf")

    args.model_results_path = sorted(
        args.model_results_path, key=_folder_recency, reverse=True
    )
    ########

    print(f"Merging results of {len(args.model_results_path)} models:")
    for path in args.model_results_path:
        print(f"  - {path.name}")
    
    # start merging
    for path in args.model_results_path:
        print(f"\n--- Processing: `{path}` ---")
        #### George: Added to skip checkpoint folders whose name starts with `cpt` or `cooldown`
        if path.name.startswith("cpt") or path.name.startswith("cooldown"):
            print(f"\tSkipping checkpoint folder `{path.name}` (starts with 'cpt' or `cooldown`)...")
            continue
        ########
        #### George: Added skip-on-error so one bad path (e.g. permission denied) doesn't abort the whole run
        try:
            if not path.exists():
                raise FileNotFoundError(f"Folder does not exist: {path}")

            if args.remerge:
                old_results_dir = path / "old_results"
                old_results_dir.mkdir(exist_ok=True)
                # move every top-level results file into old_results/ so all
                # files are sorted together by their filename timestamp
                for filepath in path.glob("results_*.json"):
                    dst = old_results_dir / filepath.name
                    if dst.exists():
                        dst = old_results_dir / f"{filepath.stem}_top.json"
                    shutil.move(str(filepath), dst)
                old_filepaths = sorted(old_results_dir.glob("results_*.json"), key=sort_results_key)
                if len(old_filepaths) == 0:
                    print(f'\tWARNING: No files matching "results_*.json" found in {old_results_dir}, skipping...')
                    continue
                merged = remerge_from_old_results(old_results_dir)
                base_stem, prev_count = parse_merged_stem(old_filepaths[-1].stem)
                output_filepath = path / f"{base_stem}_merged_{prev_count + 1}.json"
                write_json_file(merged, output_filepath)
                print(f"Re-merged {len(old_filepaths)} files from `{old_results_dir}` into: {output_filepath}")
                continue

            merged = None
            # merge old results if remerge is True or `old_results/` exists
            old_results_dir = path / "old_results"
            if not old_results_dir.exists():
                print(f"WARNING: No `{str(old_results_dir)}` folder found, skipping...")
            else:
                print(f"\tFound existing `{str(old_results_dir)}`, re-merging from there...")
                merged = remerge_from_old_results(old_results_dir)

            results_filepaths = sorted(path.glob("results_*.json"), key=sort_results_key)
            # skip results files resulted from the judge (except judge-name)
            results_filepaths = [
                filepath
                for filepath in results_filepaths
                if JUDGE_PREFIX not in filepath.stem or args.judge_name in filepath.stem
            ]
            if len(results_filepaths) == 0:
                print(f'\tWARNING: No files matching "results_*.json" found in {path}, skipping...')
                continue
            elif len(results_filepaths) == 1:
                print("\tThere is only one results_*.json file, no merge is needed!")
                continue
            else:
                print(f"Merging {len(results_filepaths)} results_*.json files in {path}")
                if merged is None:
                    merged = read_json_file(results_filepaths[0])
                else:
                    merged = deep_merge_prefer_second(
                        merged,
                        read_json_file(results_filepaths[0])
                    )
                # merge remaining results
                for result_file in results_filepaths[1:]:
                    merged = deep_merge_prefer_second(
                        merged,
                        read_json_file(result_file)
                    )

                # copy all results into a new folder
                old_results_dir.mkdir(exist_ok=True)
                for filepath in results_filepaths:
                    shutil.move(str(filepath), old_results_dir / filepath.name)

                # write merged file with same name as the most recent file
                #### George: Added to keep merged filenames from growing on every re-merge
                base_stem, prev_count = parse_merged_stem(results_filepaths[-1].stem)
                output_filepath = path / f"{base_stem}_merged_{prev_count + 1}.json"
                ########
                write_json_file(merged, output_filepath)
                print(f"Merged {len(results_filepaths)} files into: {output_filepath}")
                print(f"Original files moved to: {old_results_dir}")
        except Exception as e:
            print(f"ERROR: Skipping `{path}`: {e}")
            continue
        ########




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model-results-path", type=Path, nargs="+",
        help='Path(s) to folder(s) containing files like "results_*.json".',
    )
    parser.add_argument(
        "--results-path", type=Path,
        help='Path to the folder where model results exist.',
    )
    parser.add_argument(
        "--judge-name", type=str, default=DEFAULT_JUDGE.split('/')[-1],
        help=(
            "Name of the judge model to consider when merging results "
            f"(default: {DEFAULT_JUDGE.split('/')[-1]})."
        )

    )
    parser.add_argument(
        "--remerge", action="store_true",
        help=(
            "Move all top-level results_*.json into old_results/ and rebuild the "
            "merged file from there (sorted by filename timestamp), even when "
            "there are only 0 or 1 top-level results files."
        ),
    )
    args = parser.parse_args()
    # ensure either model-results-path or results-path is provided
    if not args.model_results_path and not args.results_path:
        parser.error(
            "Either `--model-results-path` or `--results-path` must be provided."
        )
    if args.model_results_path and args.results_path:
        parser.error(
            "Only one of `--model-results-path` or `--results-path` can be provided."
        )
    # ensure results-path exists
    if args.results_path and not args.results_path.exists():
        parser.error(f"`--results-path {args.results_path}` does not exist!")
    
    main(args)
