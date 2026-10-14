"""
This script is adapted from jais_efforts/infer_vllm.py

Example
-------
python judge_vllm.py --results-path /jais3/eval/results/<model_dir> --batch-size 16384 \
    --judge-params "seed=42,max_tokens=8192,temperature=0"

Notes
-----
* Each judge's output is saved under each model's dir as `judge__[JUDGE_NAME]/samples_*.jsonl`
  (resumable per judge). Final scores are written to a single `results_*_judged.json`.
"""
import re
import gc
import math
import json
import torch
import shutil
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from logger import logger
from configs import (
    JUDGE_PREFIX,
    DEFAULT_JUDGE,
    JUDGE_BATCH_SIZE,
    JUDGE_GPU_MEMORY_UTILIZATION,
)
from judge_registry.utils import (
    load_model_samples,
    load_judge_output_samples,
    get_task_entry,
    parse_judge_output,
    free_judge,
)




def batch_infer(
        llm: LLM,
        in_df: pd.DataFrame,
        judge_out_foldername: str,
        args: argparse.Namespace
    ):
    batch_size = args.batch_size
    tokenizer = llm.get_tokenizer()
    # group in_df by taskname
    for taskname, task_df in in_df.groupby("taskname"):
        # load task entry from JUDGE_REGISTRY
        task_entry = get_task_entry(taskname)
        if args.judge_params:
            judge_params = dict(args.judge_params)
        else:
            judge_params = dict(task_entry["judge_params"])
        logger.info(f"Judging `{taskname}` with decoding parameters: {judge_params}")
        enable_thinking = judge_params.pop("enable_thinking", False)
        sample_params = SamplingParams(
            **judge_params,
            structured_outputs=(
                StructuredOutputsParams(json=task_entry["schema"].model_json_schema())
                if task_entry["schema"]
                else None
            ),
        )
        apply_chat_template = tokenizer.chat_template is not None
        if apply_chat_template:
            logger.info("Tokenizer has a chat-template, using it...")
        else:
            logger.info("Tokenizer doesn't have a chat-template, encoding inputs instead...")
        inputs = task_df["prompt"].tolist()
        for i in tqdm(range(0, len(inputs), batch_size), desc="Inferring"):
            if apply_chat_template:
                batched_tokenized_inputs = [
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": msg}],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=enable_thinking,
                    )
                    for msg in inputs[i:i + batch_size]
                ]
            else:
                batched_tokenized_inputs = [
                    tokenizer.encode(msg, add_special_tokens=False)
                    for msg in inputs[i:i + batch_size]
                ]
            # generate judge outputs 
            judge_outputs = llm.generate(
                batched_tokenized_inputs,
                sample_params,
                use_tqdm=not args.hide_subbatch_progress
            )
            # parse judge outputs using the JSON schema and put it in a dataframe
            judge_output_df = parse_judge_output(
                judge_outputs=judge_outputs,
                output_schema_json=(
                    task_entry["schema"].model_json_schema()
                    if task_entry["schema"]
                    else None
                ),
            )
            # sanity check
            assert len(batched_tokenized_inputs) == len(judge_outputs) == len(judge_output_df)
            # concatenate column-wise
            batched_task_df = task_df.iloc[i:i + batch_size].copy()
            batched_task_df = pd.concat(
                [batched_task_df.reset_index(drop=True), judge_output_df],
                axis=1
            )
            # group by model_dir & filename, append to that judge's output folder
            for (filepath, filename), group_df in batched_task_df.groupby(["filepath", "filename"]):
                out_jsonl_filepath = Path(filepath) / judge_out_foldername / f"{filename}"
                # append to the judge's output JSONL file
                group_df.to_json(
                    out_jsonl_filepath,
                    orient="records",
                    lines=True,
                    force_ascii=False,
                    mode='a'
                )
                logger.debug(
                    f"Judge results of {filename} of `{Path(filepath).name}` "
                    f"saved to: {out_jsonl_filepath}"
                )




def main(args):
    args.results_path = list(set(args.results_path))
    logger.info("\n".join([*(str(p) for p in args.results_path), f"Total: {len(args.results_path)}"]))
    default_judge = args.judge_model_name or DEFAULT_JUDGE
    logger.info(f"Using judge model: `{default_judge}` as the default judge!")

    # ── 1. load every judgeable sample across all model dirs ──────────────────
    frames = []
    for model_dir in args.results_path:
        logger.info(f"Started processing: {model_dir.name}")
        df = load_model_samples(model_dir)
        if len(df) == 0:
            continue
        logger.info(f"Loaded {len(df)} utterances from {model_dir.name}")
        frames.append(df)

    if len(frames) == 0:
        logger.warning("No judgeable samples found.")
    
    in_df = pd.concat(frames, ignore_index=True)
    in_df["judge_model"] = in_df["taskname"].apply(
        lambda t: get_task_entry(t, default_judge)["judge"]
    )
    in_df["judge_metadata"] = in_df["taskname"].apply(
        lambda t: json.dumps(get_task_entry(t, default_judge).get("metadata", {}), sort_keys=True)
    )
    if args.limit:
        in_df = in_df.iloc[:args.limit]
        logger.info(
            f"--limit flag set, only judging the first {args.limit} samples "
            f"of each sub-task of each benchmark!"
        )

    # ── 2. judge each distinct judge model in turn ────────────────────────────
    for (judge_model, judge_metadata), sub in in_df.groupby(["judge_model", "judge_metadata"]):
        judge_out_foldername = JUDGE_PREFIX + judge_model.split("/")[-1]
        # override + resumability, per model dir, scoped to current judge's output folder
        todo_frames = []
        for model_dir in args.results_path:
            md_sub = sub[sub["filepath"] == str(model_dir)]
            if len(md_sub) == 0:
                continue
            judge_out_dir = model_dir / judge_out_foldername
            judge_out_dir.mkdir(parents=True, exist_ok=True)
            if args.override and any(judge_out_dir.iterdir()):
                shutil.rmtree(judge_out_dir)
                judge_out_dir.mkdir(parents=True, exist_ok=True)
                logger.warning(f"[OVERRIDE]: cleared {judge_out_dir}")
                for rf in sorted(model_dir.glob("results_*.json")):
                    if judge_out_foldername in rf.stem or rf.stem.endswith("_judged"):
                        rf.unlink()
            if any(judge_out_dir.iterdir()):
                done = load_judge_output_samples(judge_out_dir)
                mask = ~md_sub.set_index(["filepath", "filename", "doc_id"]).index.isin(
                    done.set_index(["filepath", "filename", "doc_id"]).index
                )
                md_sub = md_sub[mask]
                logger.info(f"Skipping {len(done)} samples that have already been judged!")
            todo_frames.append(md_sub)

        todo = (
            pd.concat(todo_frames, ignore_index=True)
            if todo_frames
            else pd.DataFrame()
        )
        if len(todo) == 0:
            logger.info(f"All samples already judged by {judge_model}!")
            continue
        logger.info(f"=== Judging {len(todo)} samples with {judge_model} ===")
        # Load the judge model
        llm = LLM(
            model=judge_model,
            tensor_parallel_size=max(1, args.tensor_parallel_size),
            gpu_memory_utilization=args.gpu_mem_utilization,
            **json.loads(judge_metadata),
        )
        logger.info("LLM loaded successfully")
        batch_infer(
            llm=llm,
            in_df=todo,
            judge_out_foldername=judge_out_foldername,
            args=args
        )
        # Free the judge model's GPU memory
        llm = free_judge(llm)

    # ── 3. score: each subset read from ITS judge's folder; write one combined results file ──
    for model_dir in args.results_path:
        results_filepaths = sorted(model_dir.glob("results_*.json"))
        if len(results_filepaths) == 0:
            logger.warning(f"{model_dir}/results_*.json not found, skipping!")
            continue
        # base = the original (non-judge-derived) results file, else fall back to the latest
        base_results_candidates = [
            f for f in results_filepaths
            if JUDGE_PREFIX not in f.stem and not f.stem.endswith("_judged")
        ]
        merged_results_filepath = (
            base_results_candidates[-1]
            if base_results_candidates
            else results_filepaths[-1]
        )

        with open(merged_results_filepath, "r", encoding="utf-8") as fin:
            file_data = json.load(fin)
        for taskname in tqdm(file_data["results"].keys(), "Scoring"):
            task_entry = get_task_entry(taskname)
            if task_entry is None:
                logger.warning(f"Task `{taskname}` is not in JUDGE_REGISTRY, skipping!")
                continue
            judge_model = task_entry["judge"]
            judge_out_foldername = JUDGE_PREFIX + judge_model.split("/")[-1]
            judge_out_filepaths = list(
                (model_dir / judge_out_foldername).glob(f"samples_{taskname}_[0-9]*.jsonl")
            )
            if len(judge_out_filepaths) == 0:
                logger.info(
                    f"No judge files found for `{taskname}`, most likely "
                    "a benchmark group... skipping!"
                )
                continue
            elif len(judge_out_filepaths) > 1:
                logger.warning(
                    f"Multiple judge files found for `{taskname}` task, "
                    "loading the most recent one!"
                )
            judge_out_df = pd.read_json(judge_out_filepaths[-1], lines=True)
            # loading the scoring function from the task entry
            scorer = task_entry["scoring_fn"]
            file_data["results"][taskname].update(scorer(judge_out_df, model_dir))
            logger.info(f"Scored `{taskname}` with `{scorer.__name__}` scoring function!")

        filename_stem = merged_results_filepath.stem
        # strip any prior [JUDGE-NAME]/merged/judged suffixes
        positions = [
            filename_stem.find(suffix)
            for suffix in (f"_{JUDGE_PREFIX}", "_judged", "_merged")
            if suffix in filename_stem
        ]
        if positions: filename_stem = filename_stem[:min(positions)]
        # write the updated judged results file while adding _judged suffix
        out_results_filepath = merged_results_filepath.parent / f"{filename_stem}_judged.json"
        logger.info(f"Writing results to {out_results_filepath}")
        with open(out_results_filepath, "w", encoding="utf-8") as fout:
            json.dump(file_data, fout, ensure_ascii=False, indent=2)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-path", type=Path, nargs="+", required=True,
        help=("A list of space-separated path(s) to folders containing "
              "samples_*.jsonl & results_**.json files generated from llm-evaluation-harness"),
    )
    parser.add_argument(
        "--judge-model-name", type=str, default=DEFAULT_JUDGE,
        help=(
            "Fallback judge for tasks not in JUDGE_REGISTRY "
            f"(default: {DEFAULT_JUDGE})"
        ),
    )
    parser.add_argument(
        "--batch-size", type=int, default=JUDGE_BATCH_SIZE,
        help=f"Batch size for inference (default: {JUDGE_BATCH_SIZE})"
    )
    parser.add_argument(
        "--judge-params", type=str, default=None,
        help=(
            "JSON string of decoding parameters (e.g. seed=42, temperature=0)"
            f"(default: None)"
        )
    )
    parser.add_argument(
        "--override", action="store_true",
        help="Flag to override judge output files and scores"
    )
    parser.add_argument(
        "--hide-subbatch-progress", action="store_true",
        help="Flag to hide sub-batch progress bar"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit the number of samples to judge (debug mode)"
    )
    parser.add_argument(
        "--gpu-mem-utilization", type=float,
        default=JUDGE_GPU_MEMORY_UTILIZATION,
        help=(
            "Fraction of GPU memory vLLM reserves for the judge model "
            f"(default: {JUDGE_GPU_MEMORY_UTILIZATION})"
        )
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=torch.cuda.device_count(),
        help="Number of GPUs to use for loading the judge model (default: all GPUs)"
    )
    args = parser.parse_args()
    # parse the judge decoding parameters (if provided as a string)
    if args.judge_params and isinstance(args.judge_params, str):
        args.judge_params = {
            param.split('=')[0].strip(): eval(param.split('=')[1].strip())
            for param in args.judge_params.split(',')
        }
    main(args)
