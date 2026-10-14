# Jais 3 Evaluation Pipeline

![Eval pipeline overview](./assets/pipeline.png)

This pipeline has three main stages:

1. `eval.py` runs `lm-eval` for selected checkpoints and tasks.
2. `judge_vllm.py` uses an LLM judge to score generated multiple-choice answers for `_gen` tasks.
3. `merge_results.py` merges multiple `results_*.json` files, including judge-enriched result files, into one consolidated result file per model folder.

## Run Evaluation

`eval.py` is the main evaluation entry point. It runs `lm-eval` for selected checkpoints and tasks, either directly or through Slurm `sbatch`.

```bash
# Run all default base, instruct, and think checkpoints on all tasks
python eval.py --base --instruct --think

# Run one explicit base checkpoint on selected tasks
python eval.py --base /jais3/checkpoints/k2v3-4B --tasks hellaswag mmlu

# Run all default instruct checkpoints on Arabic tasks only
python eval.py --instruct --tasks arabic

# Run all default think checkpoints on English tasks only
python eval.py --think --tasks english

# Smoke test with a small sample limit
python eval.py --instruct google/gemma-4-E2B-it --tasks gsm8k-zeroshot --limit 10

# Submit jobs with sbatch; --num-jobs controls how many sbatch jobs are created
python eval.py --base --instruct --think --tasks all --sbatch --num-jobs 4 --gpu-per-job 1
```

Important note: if you run `python eval.py` with no `--base`, `--instruct`, or `--think`, no checkpoints are selected. Passing a flag with no values, like `--base`, means "use the default checkpoints for this type."

Evaluations of the FLORES, Darija transliteration, and Egyptian transliteration
groups also report `spbleu` and `chrfpp`. `spbleu` is corpus BLEU with SacreBLEU's
`flores200` SentencePiece tokenizer. `chrfpp` is corpus chrF++ with character
1–6-grams, word 1–2-grams, and beta=2; the existing `chrf` metric uses only character
n-grams. Both new metrics use a 0–100 scale. FLORES averages direction scores using
sample counts, as it does for BLEU. The required packages (`sacrebleu` and
`sentencepiece`) are pinned in `requirements.txt`; SacreBLEU downloads the spBLEU
tokenizer model on first use. Use `--override` to rerun tasks with existing results
and obtain the new metrics.

## Run the Judge

`judge_vllm.py` is the LLM-as-a-judge stage. It reads model output folders created by `eval.py`, loads `samples_*.jsonl`, keeps only task/subset names ending in `_gen`, builds judge prompts from each sample, runs the judge model with vLLM structured JSON output, writes judge sample files under a `judge__[JUDGE_MODEL_NAME]` subfolder, and writes a judge-enriched `results_*.json` file with updated `acc,none` scores.

Run the judge for one or more model result folders:

```bash
python judge_vllm.py \
  --results-path /jais3/eval/results/some_model_folder \
  --judge-model-name google/gemma-4-E2B-it \
  --batch-size 16384 \
  --judge-params seed=42,max_tokens=8192,temperature=0
```

Judge output layout for a model folder:

```text
<model_folder>/
├── results_*.json
├── samples_<task>_<timestamp>.jsonl
├── judge__[JUDGE_NAME]/
│   └── samples_<task>_<timestamp>.jsonl
└── results_<timestamp>_judge__[JUDGE_NAME].json
```

## Merge Results

`merge_results.py` consolidates multiple `results_*.json` files inside model result folders. It recursively merges JSON dictionaries, appends unique list items, prefers later files when scalar values conflict, moves original result files into `old_results/`, and writes one merged result file with a `_merged.json` suffix.

Merge one or more explicit model folders:

```bash
python merge_results.py \
  --model-results-path /jais3/eval/results/model_a /jais3/eval/results/model_b
```

Merge all immediate model folders under a results root:

```bash
python merge_results.py --results-path /jais3/eval/results
```

By default, `merge_results.py` ignores judge result files from other judges and keeps judge result files whose stem contains the configured judge name from `configs.py`. Override that selection with:

```bash
python merge_results.py \
  --results-path /jais3/eval/results \
  --judge-name gemma-4-E2B-it
```

## Typical Flow

1. Add or update models and tasks in `configs.py`.
2. Run `eval.py` with the checkpoint type and task group you want.
3. `eval.py` calls `helpers.py` to filter checkpoints, skip finished tasks, and build `lm-eval` commands.
4. `lm-eval` writes `results_*.json` and `samples_*.jsonl` under `--results-path`, one folder per checkpoint.
5. Run `judge_vllm.py` directly, or `judge.sh` through Slurm, to judge `_gen` sample files.
6. The judge writes `judge__<judge_name>/samples_*.jsonl` and a judge-enriched `results_*_judge__<judge_name>.json`.
7. Run `merge_results.py` to combine partial and judge-enriched result files into a final `results_*_merged.json`.


## Multi-turn Benchmarks (mt_eval)

The multi-turn, LLM-judged benchmarks — `mt_bench`, `mt_bench_arabic`, `multichallenge`,
`multi_if`, and the `aragen_*` pair — cannot be expressed as lm-eval tasks: MT-Bench's turn 2
is generated from the model's own turn-1 answer, and prior assistant turns must carry the
model's thinking-tag format in the chat template. They run through the
`mt-evaluation-harness/` submodule instead, which is set up exactly like
`lm-evaluation-harness/`: installed editable into each eval venv and invoked as a console
script (`mt-eval`, mirroring `lm-eval`). See that submodule's README for the install command.

`configs.TASK_TO_HARNESS` is the routing dictionary: any task listed there runs through
mt_eval, every other task routes to lm-eval exactly as before. The routing is applied by
`helpers.create_eval_cmd` when commands are built, so `eval.py`, cron, resume checks, and
the judge/merge stages need no special handling — mt_eval's CLI mirrors lm-eval's and its
`samples_*.jsonl` / `results_*.json` outputs are byte-compatible.

Judging: `multichallenge` uses the strict binary rubric
(`judge_prompts.multichallenge_prompt`); MT-Bench uses FastChat's official single-answer
1-10 grading prompts (`judge_prompts.mtbench_prompt`, with the reference-answer variants
for math/reasoning/coding) through `MTBenchJudgeOutputSchema`. `judge_vllm.py` writes MT-Bench
scores as `judge_score,none` (native 1-10 mean) plus `turn1_score,none`/`turn2_score,none`,
with `acc,none` = mean/10 so dashboards stay on a 0-1 scale. All three route to
`gemma-4-26B-A4B-it` via `BENCHMARK_TO_JUDGE`.


## How the Files Connect

`configs.py` stores the defaults: tasks, checkpoint lists, paths, blacklist, and judge settings. It also defines `build_judge_prompt`, which turns task options, the gold answer, and a model response into the prompt consumed by the judge.

`helpers.py` reads those defaults and turns them into checkpoint lists, task lists, result paths, model arguments, generation arguments, and full `lm-eval` commands.

`eval.py` is the evaluation entry point. It parses CLI arguments, asks `helpers.py` what should run, then runs the commands locally or submits them with `sbatch`.

`judge_vllm.py` is the judge entry point. It reads the sample files written by `lm-eval`, builds judge prompts using `configs.py`, runs the judge model with vLLM, saves judge sample outputs, and writes judge-scored `results_*.json` files.

`judge.sh` is a convenience Slurm wrapper for `judge_vllm.py`. It batches model result folders under a root directory and submits one sbatch job per batch.

`merge_results.py` is the post-processing entry point. It merges multiple result JSON files per model folder, including the selected judge-enriched result file, into a single merged file.

There is no root-level `functions.py` in the current repo. Shared pipeline functions are in `helpers.py`; judge prompt/config functions are in `configs.py`.

# Main Scripts (in-depth)

## `eval.py`

`eval.py` orchestrates the evaluation run.

What it does:

- Reads CLI arguments.
- Gets checkpoints from `--base`, `--instruct`, and `--think`.
- Removes checkpoints listed in `BLACKLIST_CHECKPOINTS`.
- Checks which tasks are still missing for each checkpoint.
- Builds the final `lm-eval` command.
- Runs each command directly, or groups commands into Slurm jobs when `--sbatch` is used.

Main arguments:

| Argument | Meaning |
| --- | --- |
| `--base` | Base checkpoints to evaluate. Use `--base` alone for all default base checkpoints. |
| `--instruct` | Instruct checkpoints to evaluate. Use `--instruct` alone for all default instruct checkpoints. |
| `--think` | Thinking checkpoints to evaluate. Use `--think` alone for all default thinking checkpoints. |
| `--tasks` | Tasks to run. Supports `all`, `english`, `arabic`, `coding`, or task names like `mmlu gsm8k`. |
| `--results-path` | Where result folders are written. Default comes from `configs.py`. |
| `--local-benchmark-path` | Directory where `lm-eval` runs. Default comes from `configs.py`. |
| `--sbatch` | Submit jobs with Slurm instead of running directly. |
| `--num-jobs` | With `--sbatch`, number of sbatch jobs to submit in parallel. |
| `--gpu-per-job` | With `--sbatch`, number of GPUs requested by each job. |
| `--jobname` | Slurm job name. |
| `--sbatch-log-path` | Directory for Slurm logs. |
| `--sbatch-wait-time` | Seconds to wait between Slurm submissions. |
| `--limit` | Limit number of samples per task, useful for testing. |
| `--seed` | Evaluation seed. |
| `--batch-size` | Batch size passed to `lm-eval`. |
| `--default-max-model-length` | Max model length cap. Helpers may lower it based on `config.json`. |
| `--temperature` | Generation temperature. |
| `--dtype` | vLLM dtype. |
| `--attention-backend` | vLLM attention backend. |
| `--gpu-mem-utilization` | vLLM GPU memory utilization. |
| `--thinking-closing-tags` | Tags used to stop/remove thinking traces for thinking models. |
| `--hf-hub-cache-path` | HuggingFace cache path used to inspect model configs. |
| `--lm-eval-verbosity` | Logging level for `lm-eval`. |
| `--override` | Defined, but currently not used by `eval.py`. |

## `configs.py`

`configs.py` is the central place for defaults and allowed values.

What it contains:

- Pipeline defaults: seed, temperature, dtype, max tokens, batch size, GPU memory use, result paths, benchmark path, HF cache path, and sbatch log settings.
- Task maps: `EN_TASKS` and `AR_TASKS`. Each task maps to the checkpoint types that can run it: `base`, `instruct`, and/or `think`.
- Default checkpoint sets: `DEFAULT_BASE_CHECKPOINTS`, `DEFAULT_INSTRUCT_CHECKPOINTS`, and `DEFAULT_THINK_CHECKPOINTS`.
- `BLACKLIST_CHECKPOINTS`: checkpoints skipped before evaluation.
- Judge settings: `JUDGE_PREFIX`, `JUDGE_NAME`, `JUDGE_PARAMS`, `JUDGE_BATCH_SIZE`, `JudgeOutputSchema`, and `build_judge_prompt`.

How to add something:

- Add a benchmark: add it to `EN_TASKS` or `AR_TASKS`, with the checkpoint types it supports. Example: `"new_task": ["base", "instruct"]`. If it needs the multi-turn harness instead of lm-eval, also add it to `TASK_TO_HARNESS` and implement it as an `mt_eval` task module.
- Add a model: add its local path or HuggingFace ID to the right default set: base, instruct, or think.
- Disable a model temporarily: add the exact checkpoint path or ID to `BLACKLIST_CHECKPOINTS`.
- Change default output paths or runtime defaults: edit the constants near the top, like `RESULTS_PATH`, `LOCAL_BENCHMARK_PATH`, `BATCH_SIZE`, or `MAX_TOKENS`.
- Change judge behavior: edit the judge constants, `JudgeOutputSchema`, or `build_judge_prompt`.

## `helpers.py`

`helpers.py` contains the logic used by `eval.py`.

Main functions:

| Function | Main arguments | What it does |
| --- | --- | --- |
| `get_checkpoints` | `base_checkpoints`, `instruct_checkpoints`, `think_checkpoints` | Resolves defaults, expands local checkpoint folders, validates checkpoint type, and returns `(ckpt_type, ckpt)` pairs. |
| `filter_checkpoints` | `checkpoints` | Removes checkpoints listed in `BLACKLIST_CHECKPOINTS`. |
| `get_ckpt_results_path` | `ckpt_type`, `ckpt`, `results_root_path` | Builds the output folder name for a checkpoint. |
| `get_remaining_tasks` | `ckpt_type`, `ckpt`, `task_manager`, `tasks`, `ckpt_results_path` | Resolves task groups, validates task names, checks task compatibility, and skips already-finished tasks. |
| `get_model_args` | `ckpt_type`, `ckpt`, `args` | Builds vLLM model args like `pretrained`, `dtype`, memory use, max length, and thinking settings. |
| `get_gen_args` | `ckpt_type`, `ckpt`, `args` | Builds generation args like temperature and max generation tokens. |
| `create_lm_eval_cmd` | `tasks`, model args, gen args, template flag, output path | Builds the raw `lm-eval --model vllm ...` command. |
| `get_evaluation_cmd` | `ckpt_type`, `ckpt`, `task_manager`, `args`, `sbatch` | Builds the full shell command, including env vars and virtualenv activation. |
| `create_lm_eval_sbatch_cmd` | sbatch settings, `wrap_cmd` | Wraps one or more eval commands in an `sbatch` command. |
| `get_max_model_length` | `ckpt_path`, default max length, HF cache path | Reads `config.json` when available and caps the max length safely. |

How to add or change helper behavior:

- Add a new model family virtualenv: edit `get_ckpt_venv`.
- Add environment variables for every run: edit `set_environment_variables`.
- Change how result folders are named: edit `get_ckpt_results_path`.
- Change how completed tasks are detected: edit `get_remaining_tasks`.
- Change vLLM or generation arguments: edit `get_model_args` or `get_gen_args`.

## `judge_vllm.py`

`judge_vllm.py` scores generated multiple-choice tasks after evaluation.

Main flow:

1. Receives one or more model result folders through `--results-path`.
2. Reads each folder's `samples_*.jsonl` files.
3. Keeps only tasks/subsets whose names end with `_gen`.
4. For every sample, builds a prompt with `build_judge_prompt(options, gold_label, answer)`.
5. Loads the judge model with vLLM.
6. Applies the judge model chat template and runs batched generation.
7. Forces structured JSON output with `JudgeOutputSchema`.
8. Writes judge sample files under `judge__[judge_model_short_name]/`.
9. Reads the original result JSON, replaces `acc,none` for judged subsets with judge accuracy, and writes a new result JSON with the judge suffix.

Main arguments:

| Argument | Meaning |
| --- | --- |
| `--results-path` | One or more model result folders created by `eval.py`. |
| `--judge-model-name` | Judge model name/path. Default comes from `configs.py`. |
| `--batch-size` | Judge inference batch size. |
| `--judge-params` | Comma-separated vLLM sampling params, such as `seed=42,max_tokens=8192,temperature=0`. |
| `--override` | Delete previous judge outputs and judge result files for the selected judge before rerunning. |
| `--limit` | Limit number of samples to judge, useful for debugging. |

## `merge_results.py`

`merge_results.py` consolidates result JSON files after evaluation and judging.

What it does:

- Accepts either explicit model folders via `--model-results-path` or a root folder via `--results-path`.
- If `old_results/` already exists, re-merges the files inside it first.
- Finds current `results_*.json` files.
- Skips judge result files from other judges, while keeping files for `--judge-name`.
- Deep-merges result JSON content.
- Moves original result files into `old_results/`.
- Writes a final `results_*_merged.json` file in the model result folder.

Use `merge_results.py` after `eval.py` and `judge_vllm.py` when a model folder has several partial result files or judge-enriched result files that need to become one final artifact.
