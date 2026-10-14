import re
import yaml
import time
import math
import subprocess
from pathlib import Path
# from collections import Counter
from argparse import ArgumentParser
from lm_eval.tasks import TaskManager

from helpers import *
from m4_helpers import *
from logger import logger
from configs import (
    SEED,
    TEMPERATURE,
    DTYPE,
    DEFAULT_MAX_MODEL_LEN,
    DEFAULT_PROFILE,
    BATCH_SIZE,
    GPU_MEM_UTIL,
    ATTENTION_BACKEND,
    LM_EVAL_VERBOSITY,
    THINKING_CLOSING_TAGS,
    RESULTS_PATH,
    LOCAL_BENCHMARK_PATH,
    HF_HUB_CACHE_PATH,
    SBATCH_WAIT_TIME,
    SBATCH_LOG_PATH,
)

def main(args):
    # get lm-evaluation-harness task manager
    logger.info("Loading lm-evaluation-harness task manager...")
    task_manager = TaskManager()
    # load hyperparams.yaml found in same folder
    logger.info("Loading hyperparams.yaml...")
    hyperparams_yaml_filepath = Path(__file__).parent / "hyperparams.yaml"
    try:
        with open(hyperparams_yaml_filepath, "r") as f:
            hyperparams = yaml.load(f, Loader=yaml.FullLoader)["configurations"]
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Hyperparams file not found at: `{hyperparams_yaml_filepath}`"
        )
    profile_hyperparams = hyperparams[args.profile]
    # expand checkpoints
    logger.info("Loading checkpoints for evaluation...")
    checkpoints = get_checkpoints(
        args.base,
        args.instruct,
        args.think
    )
    # filter black-listed checkpoints
    checkpoints = filter_checkpoints(checkpoints)
    if not checkpoints:
        logger.error("No checkpoints left to evaluate!")
        return
    logger.info(f"Total checkpoints to evaluate: {len(checkpoints)}")
    logger.debug('\n'.join([f"\t - {ckpt}" for ckpt in checkpoints]))

    # get command to export environment variables
    export_env_variable_cmd = set_environment_variables()

    sbatch_jobs = [] # used only when --sbatch is passed
    for ckpt_type, ckpt in checkpoints:
        logger.info(f"========================================================")
        logger.info(f"Evaluating `{ckpt_type}` checkpoint: `{ckpt}`")
        logger.info(f"========================================================")
        # get output path for this checkpoint's results
        ckpt_results_path = get_ckpt_results_path(
            ckpt_type, ckpt, args.results_path,
            results_suffix=getattr(args, "results_suffix", ""), #! ANWAR: REDUNDANT FEATURE, REMOVE LATER
        )
        ckpt_results_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Results will be saved to: `{ckpt_results_path}`")
        # get remaining tasks for this checkpoint
        remaining_tasks = get_remaining_tasks(
            ckpt_type, ckpt, task_manager, args.tasks, ckpt_results_path,
            override=args.override
        )
        if not remaining_tasks:
            logger.info(f"All tasks already evaluated for this checkpoint! Skipping...\n\n")
            continue
        # group tasks by their common hyperparameters (venv, harness, model args, gen args)
        grouped_remaining_tasks = group_tasks_by_hyperparameters(
            tasks=remaining_tasks,
            ckpt_type=ckpt_type,
            ckpt=ckpt,
            profile_hyperparams=profile_hyperparams,
            args=args,
        )
        # !ANWAR: this condition should be removed later when we fix the 100B venv.
        if "100b" in ckpt.lower():
            export_env_variable_cmd += ' && export CPATH=/scratch/ahmed.frikha/pyinclude'
        
        # ===== evaluate checkpoints sequentially (without sbatch) =====
        if not args.sbatch:
            # get evaluation commands for this checkpoint
            eval_commands = get_evaluation_commands(
                ckpt_type,
                ckpt,
                grouped_remaining_tasks,
                args
            )
            # evaluate commands sequentially
            for eval_command in eval_commands:
                eval_command = ' && '.join([
                    export_env_variable_cmd,
                    eval_command,
                ])
                logger.debug(f"\n# Run the following command in your terminal:\n{eval_command}\n")
                try:
                    subprocess.run(
                        eval_command,
                        check=True,
                        shell=True,
                        cwd=str(args.local_benchmark_path)
                    )
                except subprocess.CalledProcessError as e:
                    logger.error(f"Error evaluating {ckpt_type} {ckpt}: {e.stderr}")
                    if args.fail_on_error:
                        raise e
        # ===== evaluate checkpoints using sbatch =====
        else:
            log_directory = Path(args.sbatch_log_path) / ckpt_results_path.name
            log_directory.mkdir(parents=True, exist_ok=True)
            if args.task_per_job:
                # flatten the grouped_remaining_tasks into one task per entry
                grouped_remaining_tasks = [
                    (task_metadata, [task])
                    for task_metadata, tasks in grouped_remaining_tasks
                    for task in tasks
                ]
                logger.info(
                    "Since --task-per-job is passed, flattening grouped tasks "
                    f"into: {len(grouped_remaining_tasks)} tasks..."
                )
            task_groups = [task_group for _, task_group in grouped_remaining_tasks]
            # get evaluation commands for this checkpoint
            eval_commands = get_evaluation_commands(
                ckpt_type,
                ckpt,
                grouped_remaining_tasks,
                args
            )
            for eval_command, task_group in zip(eval_commands, task_groups):
                curr_log_filepath = log_directory / f"%j__{task_group}.log"
                job_name = f"{args.job_name_prefix}__{ckpt_type}__{Path(ckpt).name}__{task_group}"
                # filter out commands that are currently running
                if job_name in get_active_slurm_job_names():
                    logger.warning(
                        f"Skipping `{ckpt_type}` checkpoint: `{ckpt}` on task: `{task_group}` "
                        f" because Slurm job `{job_name}` is already running!"
                    )
                    continue
                # create the wrap command needed for sbatch submission
                wrap_cmd = ' && '.join([
                    export_env_variable_cmd,
                    f"cd {shlex.quote(str(args.local_benchmark_path))}",
                    f"echo '=== Evaluating {ckpt} [[{ckpt_type}]] ==='",
                    'echo "Machine name: $(hostname)"',
                    f"mkdir -p {shlex.quote(str(ckpt_results_path))}",
                    (
                        f"( {eval_command}; exit_code=$?; "
                        f"if [ $exit_code -eq 0 ]; then "
                        f"echo '=== Finished {ckpt} ==='; "
                        f"else echo '=== FAILED {ckpt} (exit $exit_code) ==='; fi; "
                        f"exit $exit_code )"
                    )
                ])
                # create the actual sbatch command
                sbatch_cmd = create_lm_eval_sbatch_cmd(
                    partition = "defq",
                    nodes = args.num_nodes_per_job,
                    ntasks = args.num_nodes_per_job,
                    cpus_per_task = args.num_nodes_per_job * 16, # 16 CPUs per node
                    gpu = f"gpu:{math.ceil(args.gpu_per_job / args.num_nodes_per_job)}",
                    job_name = job_name,
                    log_filepath = curr_log_filepath,
                    nodelist = args.nodelist,
                    wrap_cmd = wrap_cmd,
                )
                sbatch_jobs.append(sbatch_cmd)
        
    # start submitting sbatch jobs
    if not args.sbatch: return
    if len(sbatch_jobs) == 0:
        logger.info("No sbatch jobs to submit (all checkpoints already evaluated)🎉")
        return
    # submit jobs to sbatch
    logger.info(f"Submitting {len(sbatch_jobs)} jobs to sbatch...")
    for idx, sbatch_cmd in enumerate(sbatch_jobs):
        result = subprocess.run(
            sbatch_cmd,
            check=True,
            capture_output=True,
            text=True
        )
        job_id = result.stdout.strip()
        logger.info(f"Submitted job #{idx + 1}: {job_id}")
        time.sleep(args.sbatch_wait_time) # wait before submitting the next job





if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--base", nargs="*", default=None,
        help="List of base models to evaluate"
    )
    parser.add_argument(
        "--instruct", nargs="*", default=None,
        help="List of instruct models to evaluate"
    )
    parser.add_argument(
        "--think", nargs="*", default=None,
        help="List of think models to evaluate"
    )
    parser.add_argument(
        "--tasks", nargs="+", default=["all"],
        help=(
            "List of tasks to evaluate (space-separated). "
            "Special values:\n"
            "  - all      : run all available tasks (default) \n"
            "  - english  : run only English tasks\n"
            "  - arabic   : run only Arabic tasks\n"
        ),
    )
    parser.add_argument(
        "--task-per-job", action="store_true",
        help=(
            "Submit one sbatch job per (checkpoint, task) pair instead of "
            "one sbatch job per checkpoint. Each (ckpt, task) runs on its own GPU."
        )
    )

    parser.add_argument(
        "--results-path", type=Path, default=RESULTS_PATH,
        help=(
            f"Path to store evaluation results (default: {RESULTS_PATH})."
        )
    )
    parser.add_argument( #! ANWAR: REDUNDANT FEATURE, REMOVE LATER
        "--results-suffix", default="",
        help="Suffix appended to each model result directory (e.g. __temp_0.7)."
    )
    parser.add_argument(
        "--local-benchmark-path", type=Path, default=LOCAL_BENCHMARK_PATH,
        help=(
            f"Path to the local benchmark datasets used by evaluation tasks " +
            f"(default: {LOCAL_BENCHMARK_PATH})."
        )
    )
    parser.add_argument(
        "--fail-on-error", action="store_true",
        help="Fail the evaluation if any error occurs."
    )
    parser.add_argument(
        "--override", action="store_true",
        help="Override existing results if they exist."
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help="Random seed for evaluation (default: 42)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help="Batch size for evaluation (default: 16384)"
    )
    parser.add_argument(
        "--lm-eval-verbosity", type=str, default=LM_EVAL_VERBOSITY,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Verbosity level for lm-eval (default: DEBUG)"
    )
    parser.add_argument(
        "--max-model-length", type=int,
        help= (
            "Max model length to use with the model. If not set, it falls back "
            "to the model's max-model-length. If both are set, the smaller value "
            "wins."
        )
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help=(
            "Generation temperature. If omitted, it is resolved from --profile "
            f"(deterministic falls back to {TEMPERATURE} -> greedy). If passed, "
            "it overrides the profile value."
        )
    )
    parser.add_argument(
        "--system-instruction", type=str, default=None,
        help=(
            "A flag to enable system prompts prepended to benchmark prompts.",
        )
    )
    parser.add_argument(
        "--gen-args", type=str, default=None,
        help=(
            "Extra generation hyperparameters passed to lm-eval --gen_kwargs, "
            "excluding temperature and max-model-length.\n"
            "Example: --gen-args 'top_p=0.95, repetition_penalty=1.1'"
        ),
    )
    parser.add_argument(
        "--merge-gen-args", action="store_true", #! ANWAR: REDUNDANT FEATURE, REMOVE LATER
        help="Keep profile generation settings except keys explicitly set by --gen-args."
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Number of samples to evaluate per task (default: all samples)"
    )
    parser.add_argument(
        "--dtype", type=str, default=DTYPE,
        help="Default data type for evaluation"
    )
    parser.add_argument(
        "--attention-backend", type=str, default=ATTENTION_BACKEND,
        help="Default attention backend for evaluation"
    )
    parser.add_argument(
        "--gpu-mem-utilization", type=float, default=GPU_MEM_UTIL,
        help="Default GPU memory utilization for evaluation"
    )
    parser.add_argument(
        "--thinking-closing-tags", type=str, default=THINKING_CLOSING_TAGS,
        help=(
            "A string of closing tags for thinking of various models separated by `|` "
            "(default: THINKING_CLOSING_TAGS from configs.py). "
        )
    )
    parser.add_argument(
        "--hf-hub-cache-path", type=Path, default=HF_HUB_CACHE_PATH,
        help="Path to cache HuggingFace models and datasets"
    )
    parser.add_argument(
        "--ignore-completion-metadata", action="store_true",
        help="ignore completion metadata."
    )
    parser.add_argument(
        "--sbatch", action="store_true",
        help="Run the evaluation using sbatch instead of locally."
    )
    parser.add_argument(
        "--max-nodes", type=int,
        help=(
            "Number of sbatch jobs to submit in parallel"
        )
    )
    parser.add_argument(
        "--gpu-per-job", type=int, default=1,
        help=(
            "Number of GPUs to use for each sbatch job (default: 1) "
        )
    )
    parser.add_argument(
        "--sbatch-log-path", type=Path, default=SBATCH_LOG_PATH,
        help="Path to store sbatch logs"
    )
    parser.add_argument(
        "--sbatch-wait-time", type=int, default=SBATCH_WAIT_TIME,
        help="Time to wait between sbatch jobs (default: 10 seconds)"
    )
    parser.add_argument(
        "--job-name-prefix", type=str, default=DEFAULT_PROFILE,
        help=f"A prefix for the job name (default: {DEFAULT_PROFILE})"
    )

    parser.add_argument(
        "--profile", type=str, default=DEFAULT_PROFILE,
        choices=["optimized", "deterministic"],
        help=(
            "Hyperparameter profile. 'optimized' (default) resolves per-task "
            "tuned hyperparameters from configs.py; 'deterministic' runs "
            "greedy/deterministic. Explicitly-passed --temperature/--gen-args "
            "always override the profile."
        )
    )

    args = parser.parse_args()
    if args.results_suffix and not re.fullmatch(r"__[A-Za-z0-9_.-]+", args.results_suffix):
        #! ANWAR: REDUNDANT FEATURE, REMOVE LATER
        parser.error("--results-suffix must start with __ and contain only letters, digits, _, ., or -")
    # no mode flag at all -> run all modes (each [] resolves to its defaults)
    if args.base is None and args.instruct is None and args.think is None:
        args.base = args.instruct = args.think = []

    # make sure every path argument is a Path object
    args.results_path = Path(args.results_path)
    args.local_benchmark_path = Path(args.local_benchmark_path)
    args.sbatch_log_path = Path(args.sbatch_log_path)
    args.hf_hub_cache_path = Path(args.hf_hub_cache_path)
    
    # sanity checks
    if args.max_nodes is not None and args.max_nodes <= 0:
        raise ValueError("`--max-nodes` must be positive")
    if not args.sbatch:
        # if override flag is set, double-check with the user (fool-proofing)
        if args.override:
            confirm = input(
                "`--override` flag is set, want to override existing results? (y/n): "
            )
            if confirm.lower() != "y":
                logger.error("Removing `--override` flag...")
                args.override = False
        if args.max_nodes:
            logger.warning(
                "`--max-nodes` argument will be ignored (interactive mode)!"
            )
        if args.sbatch_log_path:
            logger.warning(
                "`--sbatch-log-path` argument will be ignored (interactive mode)!"
            )
        if args.job_name_prefix:
            logger.warning(
                "`--job-name-prefix` argument will be ignored (interactive mode)!"
            )
        if args.sbatch_wait_time:
            logger.warning(
                "`--sbatch-wait-time` argument will be ignored (interactive mode)!"
            )
        if args.task_per_job:
            logger.warning(
                "`--task-per-job` argument will be ignored (interactive mode)!"
            )
    else:
        if args.gpu_per_job and args.gpu_per_job <= 0:
            raise ValueError("`--gpu-per-job` must be positive")
        args.num_nodes_per_job = get_num_nodes_per_job(args.gpu_per_job)
        args.nodelist = get_m4_available_nodes()
        available_nodes_count = args.nodelist.count(',') + 1
        if args.max_nodes and available_nodes_count > args.max_nodes:
            args.nodelist = ','.join(args.nodelist.split(',')[:args.max_nodes])
            logger.info(
                f"Using only --max-nodes={args.max_nodes} nodes: {args.nodelist}"
            )

    #! ANWAR: commented this condition on 20th of August 2026
    # if args.profile == "optimized" and not args.task_per_job:
    #     if args.sbatch:
    #         args.task_per_job = True
    #         logger.warning(
    #             "`--profile optimized` needs per-task granularity; "
    #             "auto-enabling `--task-per-job`."
    #         )
    
    main(args)
