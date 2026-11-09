"""
Helper functions for the evaluation scripts.
"""
import re
import json
import shlex
from pathlib import Path
from collections import defaultdict
from lm_eval.tasks.manager import Kind
from fnmatch import fnmatch, fnmatchcase

from configs import *
from logger import logger


def set_environment_variables(args):
    logger.info("Setting environment variables...")
    env_variables = [
        'umask 002', # to avoid permission issues
        "export HF_ALLOW_CODE_EVAL=1" # to allow code evaluation in HuggingFace models
        ]
    if not args.online_mode:
        logger.info("Running in offline mode to avoid unecessary updates.")
        env_variables.append("export HF_HUB_OFFLINE=1")
    return ' && '.join(env_variables)


def resolve_checkpoints(checkpoints, default, msg):
    if checkpoints is None:
        return []
    if checkpoints == []:
        logger.info(msg)
        return default
    return checkpoints


def get_checkpoints(instruct_checkpoints, think_checkpoints):
    """
    Parses and combines the base, instruct, and think checkpoints into a
    single list of checkpoints to evaluate.
    """
    total_ckpts = []
    # handle instruct checkpoints
    instruct_checkpoints = resolve_checkpoints(
        instruct_checkpoints,
        INSTRUCT_CHECKPOINTS,
        "No instruct checkpoints were provided, loading all default instruct checkpoints."
    )
    # handle think checkpoints
    think_checkpoints = resolve_checkpoints(
        think_checkpoints,
        THINK_CHECKPOINTS,
        "No thinking checkpoints were provided, loading all default thinking checkpoints."
    )

    # groups to help organize the parsing and validation of checkpoints
    groups = [
        ("instruct", instruct_checkpoints, INSTRUCT_CHECKPOINTS),
        ("think", think_checkpoints, THINK_CHECKPOINTS),
    ]
    # iterate over checkpoints and validate them
    for ckpt_type, input_ckpts, allowed_set in groups:
        for ckpt in input_ckpts:
            if ckpt not in allowed_set:
                logger.error(
                    f"Checkpoint '{ckpt}' is not a valid `{ckpt_type}` checkpoint, skipping!"
                )
            else:
                total_ckpts.append((ckpt_type, ckpt))
    return total_ckpts


def get_ckpt_results_path(ckpt_type, ckpt, results_root_path):
    logger.debug(
        f"Resolving results path for `{ckpt_type}` checkpoint `{ckpt}`..."
    )
    if ckpt_type == "instruct":
        if ckpt in THINK_CHECKPOINTS:
            suffix = "_thinking_low"
            logger.debug(
                "\tCheckpoint is also a think checkpoint; using suffix `_thinking_low`."
            )
        else:
            suffix = "_instruct"
            logger.debug("\tInstruct checkpoint; using suffix `_instruct`.")
    elif ckpt_type == "think":
        suffix = "_thinking_high"
        logger.debug("\tThink checkpoint; using suffix `_thinking_high`.")
    ckpt_out_foldername = ckpt.split("/")[-1] + suffix
    ckpt_results_path = results_root_path / ckpt_out_foldername
    logger.debug(f"\tResults folder: `{ckpt_results_path}`")
    return ckpt_results_path


def fix_task_naming(task, all_tasks):
    all_tasks_map = {k.lower(): k for k in all_tasks.keys()}
    return all_tasks_map.get(task.lower(), None)


def get_task_metadata(task):
    logger.debug(f"Getting metadata for `{task}`...")
    # get harness
    harness = TASK_TO_HARNESS.get(task, DEFAULT_HARNESS)
    # get language
    if task in AR_TASKS:
        language = "arabic"
    elif task in EN_TASKS:
        language = "english"
    else:
        raise ValueError(f"Task `{task}` is not in Arabic or English tasks!")
    # get domain
    domain = "coding" if task in EN_CODING else None
    task_requested_model_length = TASK_MAX_MODEL_LEN.get(task, DEFAULT_MAX_MODEL_LEN)
    # check if task is a multiturn task
    if TASK_NUM_TURNS.get(task, 1) > 1:
        task_requested_model_length *= TASK_NUM_TURNS[task] # adjust for multiturn tasks
    logger.debug(f"\tDomain: {domain}")
    logger.debug(f"\tLanguage: {language}")
    logger.debug(f"\tTask requested model length: {task_requested_model_length}")
    return harness, domain, language, task_requested_model_length


def get_subtasks(task_manager, task):
    harness = TASK_TO_HARNESS.get(task, DEFAULT_HARNESS)
    if harness == "minihelmet_eval":
        # miniHELMET writes samples per dataset/context length, not per group.
        return list(HELMET_MINI_SUBTASKS[task])
    if harness != DEFAULT_HARNESS:
        return [task]
    subtasks = []
    def _get_subtasks_recursively(taskname):
        task_type = task_manager.task_index[taskname].kind
        if task_type == Kind.TAG:
            subtasks.extend(task_manager.task_index[taskname].tags)
        elif task_type == Kind.TASK:
            subtasks.append(taskname)
        elif task_type == Kind.GROUP:
            for subtask in task_manager.task_index[taskname].cfg["task"]:
                _get_subtasks_recursively(subtask)
        else:
            pass # ignore other task types
    _get_subtasks_recursively(task)
    return list(set(subtasks))


def get_remaining_tasks(
        ckpt_type, ckpt, task_manager, tasks, ckpt_results_path, override=False
    ):
    # if no tasks were provided, use all available tasks
    if tasks == ["all"]:
        logger.info("No tasks were provided, loading `all` available tasks.")
        tasks = list(EN_TASKS.keys()) + list(AR_TASKS.keys())
    elif tasks == ["english"]:
        logger.info("Loading all `English` tasks.")
        tasks = list(EN_TASKS.keys())
    elif tasks == ["arabic"]:
        logger.info("Loading all `Arabic` tasks.")
        tasks = list(AR_TASKS.keys())
    else:
        logger.info(f"Loading user-specified tasks: {tasks}")
        if type(tasks) == str:
            tasks = [tasks]
    # filter already-evaluated tasks for the given checkpoint
    logger.debug(f"Getting remaining tasks for `{ckpt_type}` checkpoint: `{ckpt}`")
    remaining_tasks = []
    all_tasks = {**EN_TASKS, **AR_TASKS}
    ckpt_output_samples = list(ckpt_results_path.glob("*.jsonl"))
    for task in tasks:
        # check if task exists
        if fix_task_naming(task, all_tasks) is None:
            logger.warning(
                f"\t[WARNING] Task `{task}` was not found, skipping for now "
                "but you should double-check!"
            )
            continue
        # get the correct task name (case-sensitive)
        task = fix_task_naming(task, all_tasks)
        # check if task is valid for this checkpoint type
        if ckpt_type not in all_tasks[task]:
            logger.error(
                f"\tTask `{task}` is NOT valid for checkpoint type `{ckpt_type}`, skipping!"
            )
            continue
        # check if any of the subtasks of the task group exists already
        subtasks = get_subtasks(task_manager, task)
        present = [
            subtask for subtask in subtasks
            if any(
                re.match(rf"samples_{re.escape(subtask)}_\d+.*\.jsonl$", filepath.name)
                for filepath in ckpt_output_samples
            )
        ]

        require_all = TASK_TO_HARNESS.get(task, DEFAULT_HARNESS) == "minihelmet_eval"
        done = (len(present) == len(subtasks)) if require_all else bool(present)
        if not override and done:
            logger.info(
                f"\tTask `{task}` has been already evaluated, skipping!"
            )
        else:
            remaining_tasks.append(task)
    logger.info(f"\tThere are {len(remaining_tasks)} remaining tasks:")
    for task in remaining_tasks:
        logger.info(f"\t\t- {task}")
    return remaining_tasks


def get_model_args(ckpt_type, ckpt, model_length, args):
    logger.debug(
        f"Building model args for `{ckpt_type}` checkpoint `{ckpt}`..."
    )
    model_args = {
        "pretrained": ckpt,
        "dtype": args.dtype,
        "seed" : args.seed,
        "gpu_memory_utilization": args.gpu_mem_utilization,
        "tensor_parallel_size": args.gpu_per_job,
        # "max_model_len": model_length, #! the following argument was added to handle `bbq-8b-mid-3-final` ckpt
        "log_completion_metadata": not args.ignore_completion_metadata,
    }

    # set model arguments for `instruct` models
    if ckpt_type == "instruct":
        logger.debug("\tInstruct checkpoint: disabling thinking.")
        model_args["enable_thinking"] = False
        if "Fanar-2-27B-Instruct".lower() in ckpt.lower():
            logger.debug("\tFanar instruct checkpoint: setting no_thinking=True.")
            model_args["chat_template_args"] = {
                "no_thinking": True
            }
        elif "minimax-m3" in ckpt.lower():
            logger.debug("\tMiniMax-M3 instruct checkpoint: disabling thinking_mode.")
            model_args["chat_template_args"] = {
                "thinking_mode": "disabled"
            }
    
    # set model arguments for `think` models
    elif ckpt_type == "think":
        logger.debug("\tThink checkpoint: enabling thinking.")
        model_args["enable_thinking"] = True
        model_args["think_end_token"] = args.thinking_closing_tags
        if "Qwen3.8-27B".lower() in ckpt.lower():
            logger.debug("\tQwen3.8-27B: setting reasoning_effort=xhigh.")
            model_args["chat_template_args"] = {
                "reasoning_effort": "xhigh"
            }
        elif "Muse-Glimmer-30B".lower() in ckpt.lower():
            logger.debug("\tMuse-Glimmer-30B: setting reasoning_strength=xhigh.")
            model_args["chat_template_args"] = {
                "reasoning_strength": "xhigh"
            }
        elif "minimax-m3" in ckpt.lower():
            logger.debug("\tMiniMax-M3 think checkpoint: enabling thinking_mode.")
            model_args["chat_template_args"] = {
                "thinking_mode": "enabled"
            }
        else:
            logger.debug("\tSetting default think chat template: reasoning_effort=high.")
            model_args["chat_template_args"] = {
                "reasoning_effort": "high"
            }
    
    # handle "jais 3" or "k2-horizon" checkpoints
    if "jais-3" in ckpt.lower() or "k2-horizon" in ckpt.lower():
        reasoning_effort = "low" if ckpt_type == "instruct" else "high"
        logger.debug(
            f"\tJais-3/K2-Horizon checkpoint: overriding reasoning_effort={reasoning_effort}."
        )
        model_args["enable_thinking"] = True
        model_args["think_end_token"] = args.thinking_closing_tags
        model_args["chat_template_args"] = {
            "reasoning_effort": reasoning_effort
        }
    
    # handle multi-node evaluation
    if getattr(args, "num_nodes_per_job", 1) > 1:
        logger.debug(
            f"\tMulti-node evaluation ({args.num_nodes_per_job} nodes): "
            "using Ray as the distributed executor."
        )
        model_args["distributed_executor_backend"] = "ray"

    # handle special cases
    if "nemotron" in ckpt.lower():
        # Nemotron-H has eight Mamba groups, so tensor parallelism must divide
        # eight. Use pipeline parallelism for allocations larger than TP8.
        if args.gpu_per_job >= 8 and args.gpu_per_job % 8 == 0:
            nemotron_tp_size = 8
        elif 8 % args.gpu_per_job == 0:
            nemotron_tp_size = args.gpu_per_job
        else:
            logger.error(
                "Nemotron-H requires --gpu-per-job to be a divisor or "
                f"multiple of its 8 Mamba groups; got {args.gpu_per_job}."
            )
            raise ValueError(
                "Nemotron-H requires --gpu-per-job to be a divisor or "
                "multiple of its 8 Mamba groups"
            )
        model_args["tensor_parallel_size"] = nemotron_tp_size
        nemotron_pp_size = args.gpu_per_job // nemotron_tp_size
        if nemotron_pp_size > 1:
            model_args["pipeline_parallel_size"] = nemotron_pp_size
            logger.debug(
                f"\tNemotron-H: tensor_parallel_size={nemotron_tp_size}, "
                f"pipeline_parallel_size={nemotron_pp_size}."
            )
        else:
            logger.debug(
                f"\tNemotron-H: tensor_parallel_size={nemotron_tp_size}."
            )
    if "minimax-m3" in ckpt.lower():
        # MiniMax-M3's sparse-attention backend requires 128-token KV-cache
        # blocks. These evals are text-only, so skip loading its vision tower.
        logger.debug(
            "\tMiniMax-M3: setting block_size=128 and language_model_only=True."
        )
        model_args["block_size"] = 128
        model_args["language_model_only"] = True

        if args.gpu_per_job > 8:
            if args.gpu_per_job % 8 != 0:
                logger.error(
                    "MiniMax-M3 multi-node needs --gpu-per-job to be a multiple "
                    f"of 8 (TP=8 per node); got {args.gpu_per_job}."
                )
                raise ValueError(
                    "MiniMax-M3 multi-node needs --gpu-per-job to be a multiple "
                    f"of 8 (TP=8 per node); got {args.gpu_per_job}"
                )
            model_args["tensor_parallel_size"] = 8
            model_args["pipeline_parallel_size"] = args.gpu_per_job // 8
            logger.debug(
                "\tMiniMax-M3 multi-node: tensor_parallel_size=8, "
                f"pipeline_parallel_size={args.gpu_per_job // 8}."
            )
    
    if "qwen3.5-397b" in ckpt.lower():
      logger.debug("\tQwen3.5-397B: setting max_num_seqs=128.")
      model_args["max_num_seqs"] = 128
    
    if "falcon-h1-34b-instruct" in ckpt.lower():
      logger.debug(
          "\tFalcon-H1-34B: setting max_num_seqs=128 and tensor_parallel_size=2."
      )
      model_args["max_num_seqs"] = 128
      model_args["tensor_parallel_size"] = 2   # mamba_n_groups=2
      
    # print model args
    logger.debug("\tModel args:")
    logger.debug(f"{'\n'.join([f'\t\t- {k}={v}' for k, v in model_args.items()])}")
    # return model args as string
    return json.dumps(model_args)


def resolve_max_model_length(ckpt, task_requested_model_length, args):
    # an explicit --max-model-length wins; otherwise fall back to the
    # benchmark's requested length (task_requested_model_length).
    # Either is then capped by the model's possible context length (cfg_len).
    logger.debug(f"Resolving max model length for `{ckpt}`...")
    def _read_config_max_length(ckpt_path, hf_hub_cache_path):
        """Return the max context length from the model's config.json, or None if unavailable."""
        def _find_keys_recursive(node, keys):
            """
            Yield int values for any of `keys` found at any depth in a nested
            dict/list config.
            """
            if isinstance(node, dict):
                for k, v in node.items():
                    if k in keys and isinstance(v, (int, float)):
                        yield int(v)
                    else:
                        yield from _find_keys_recursive(v, keys)
            elif isinstance(node, list):
                for item in node:
                    yield from _find_keys_recursive(item, keys)
                    
        config_file = None
        if (ckpt_path / "config.json").is_file():
            config_file = ckpt_path / "config.json"
            logger.debug(f"\tFound config.json next to the checkpoint: `{config_file}`.")
        elif Path(hf_hub_cache_path).is_dir():
            cache_name = str(ckpt_path).replace("/", "--")
            snap_dir = Path(hf_hub_cache_path) / f"models--{cache_name}" / "snapshots"
            if snap_dir.is_dir():
                subdirs = [p for p in snap_dir.iterdir() if p.is_dir()]
                if subdirs:
                    latest = max(subdirs, key=lambda p: p.stat().st_mtime)
                    candidate = latest / "config.json"
                    if candidate.is_file():
                        config_file = candidate
                        logger.debug(
                            f"\tFound config.json in the HF cache snapshot: `{config_file}`."
                        )

        if config_file is None:
            logger.debug(
                "\tNo config.json found; the model's context length is unknown."
            )
            return None
        try:
            with config_file.open("r", encoding="utf-8") as f:
                cfg = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                f"\tCould not read `{config_file}` ({exc}); "
                "the model length will not be capped."
            )
            return None

        keys = ["max_position_embeddings", "max_sequence_length", "n_positions", "seq_length"]
        found = list(_find_keys_recursive(cfg, keys))
        if found:
            cfg_len = max(found)
            logger.debug(f"\tConfig reports a max context length of {cfg_len}.")
            return cfg_len
        logger.debug(
            f"\tConfig `{config_file}` has no recognized context-length keys."
        )
        return None
    if args.max_model_length is not None:
        requested = args.max_model_length
        logger.debug(f"\tUsing explicit --max-model-length: {requested}.")
    else:
        requested = task_requested_model_length
        logger.debug(f"\tUsing the task-requested model length: {requested}.")
    cfg_len = _read_config_max_length(Path(ckpt), args.hf_hub_cache_path)
    if cfg_len is not None and requested > cfg_len:
        logger.warning(
            f"\tRequested model length {requested} exceeds the model's "
            f"context length {cfg_len}; capping to {cfg_len}."
        )
        return cfg_len
    if cfg_len is not None:
        logger.debug(
            f"\tResolved max model length: {requested} "
            f"(within the model context length {cfg_len})."
        )
        return requested
    logger.debug(f"\tResolved max model length: {requested} (uncapped).")
    return requested


def get_model_hyperparams(
        ckpt,
        profile_hyperparams,
        mode=None,
        domain=None,
        language=None
    ):
    logger.debug(f"Getting profile hyperparams for `{ckpt}`...")
    function_arguments = {"mode": mode, "domain": domain, "language": language}
    # sanity checks
    if mode and mode not in ("instruct", "think"):
        raise ValueError(
            f"Couldn't load model hyperparams for `{mode}` mode! "
            "Available modes are: instruct, think"
        )
    if domain and domain not in ("coding"):
        raise ValueError(
            f"Couldn't load model hyperparams for `{domain}` domain! "
            "Available domains are: coding"
        )
    if language and language not in ("arabic", "english"):
        raise ValueError(
            f"Couldn't load model hyperparams for `{language}` language! "
            "Available languages are: arabic, english"
        )
    
    # get the checkpoint's hyperparams
    model_hyperparams = profile_hyperparams.get("defaults", {})
    # match the model with the all keys in profile's models
    for modelname_pattern, hyperparams in profile_hyperparams.get("models", {}).items():
        if fnmatchcase(ckpt, modelname_pattern):
            logger.debug(f"\tFound a matching pattern: {modelname_pattern}")
            model_hyperparams.update(hyperparams["defaults"])
            for rule in hyperparams.get("rules", []):
                conditions = rule.get("when", {})
                # match rule conditions with given function arguments
                if all(
                    function_arguments[key] == value
                    for key, value in conditions.items()
                ):
                    model_hyperparams.update(rule.get("set", {}))
                    logger.debug(f"\tMatched a rule: {conditions}, updating model hyperparams.")
                    break
            else:
                logger.debug(f"\tNo rule was matched!")
            break
    else:
        logger.debug(f"\tNo model was matched!")
    
    # set the temperature to the default value if not already set
    model_hyperparams["temperature"] = model_hyperparams.get(
        "temperature",
        DEFAULT_TEMPERATURE
    )
    return model_hyperparams


def get_gen_args(ckpt_model_length, model_hyperparams, args):
    gen_args = []
    # an explicit --temperature overrides the profile's value
    temperature = args.temperature or model_hyperparams["temperature"]
    gen_args.append(f"temperature={temperature}")
    gen_args.append(f"do_sample={temperature != 0}")

    # --gen-args replaces the profile bundle
    if getattr(args, "gen_args", None):
        for gen_arg in args.gen_args.split(","):
            gen_arg = gen_arg.strip()
            if gen_arg:
                gen_args.append(gen_arg)
    else:
        for key, value in model_hyperparams.items():
            if key not in ("temperature", "do_sample",):
                gen_args.append(f"{key}={value}")

    gen_args.append(f"max_gen_toks={ckpt_model_length}")
    logger.debug("\tGeneration args:")
    logger.debug(f"{'\n'.join([f'\t\t- {arg}' for arg in gen_args])}")
    return ','.join(gen_args)


def apply_chat_template(ckpt_type, ckpt):
    return ckpt_type in {"instruct", "think"}


def create_scorer_cmds(tasks, output_path):
    """Post-generation scoring commands, NOT USED FOR NOW"""
    return [
        shlex.join(["python", str(scorer), "--results-path", str(output_path)])
        for scorer in (TASK_TO_SCORER.get(task) for task in tasks)
        if scorer is not None
    ]


def get_harness_cmd(harness: str) -> list[str]:
    # sanity check
    all_harnesses = set(TASK_TO_HARNESS.values()) | {DEFAULT_HARNESS}
    if harness not in all_harnesses: raise ValueError(f"Invalid harness: {harness}")
    # get the harness command
    if harness == "lm_eval":
        # return ["lm-eval"]
        return ["python", "-m", "lm_eval", "run"]
    elif harness == "mt_eval":
        return ["mt-eval"]
    else:
        raise ValueError(f"Invalid harness: {harness}")


def group_tasks_by_hyperparameters(tasks, ckpt_type, ckpt, profile_hyperparams, args):
    logger.debug(f"Grouping remaining tasks by hyperparameters...")
    grouped_tasks = defaultdict(list)
    for task in tasks:
        # get venv path
        venv_path = get_venv(task, ckpt)
        # get task metadata
        harness, domain, language, task_requested_model_length = get_task_metadata(task)
        # get model hyperparameters from profile
        model_hyperparams = get_model_hyperparams(
            ckpt,
            profile_hyperparams,
            mode=ckpt_type,
            domain=domain,
            language=language,
        )
        # get model arguments
        ckpt_model_length = resolve_max_model_length(ckpt, task_requested_model_length, args)
        ckpt_model_args = get_model_args(ckpt_type, ckpt, ckpt_model_length, args)
        # get generation arguments (per-task)
        ckpt_gen_args = get_gen_args(ckpt_model_length, model_hyperparams, args)
        # group tasks by their harness, model arguments, and generation arguments
        grouped_tasks[(venv_path, harness, ckpt_model_args, ckpt_gen_args)].append(task)
    logger.info(f"\tGrouped remaining tasks into: {len(grouped_tasks)} groups:")
    logger.info(f"{'\n'.join([f'\t\t- {tasks}' for tasks in grouped_tasks.values()])}")
    return grouped_tasks.items()


def create_eval_cmd(
        tasks,
        harness,
        per_model_args,
        per_gen_args,
        enable_chat_template,
        output_path,
        **kwargs
    ):
    # sanity check, all tasks should have the same harness
    harnesses = {TASK_TO_HARNESS.get(task, DEFAULT_HARNESS) for task in tasks}
    assert len(harnesses) == 1, f"Tasks {tasks} mix harnesses {harnesses} in one command"

    # get the harness command
    cmd = get_harness_cmd(harness)
    # add needed flags to the harness command
    cmd.extend([
        "--model", "vllm",
        "--seed", str(kwargs["seed"]),
        "--model_args", per_model_args,
        "--gen_kwargs", per_gen_args,
        "--tasks", ','.join(tasks),
        "--output_path", str(Path(output_path) / "results.json"),
        "--log_samples",
        "--verbosity", kwargs["lm_eval_verbosity"],
        "--trust_remote_code",
        "--device", "cuda:0"
        "--confirm_run_unsafe_code",  # necessary for code tasks (e.g. humaneval)
        "--apply_chat_template",
    ])
    if kwargs.get("limit") is not None:
        cmd.extend(["--limit", str(kwargs["limit"])])
    if kwargs.get("system_instruction") is not None:
        cmd.extend(["--system_instruction", str(kwargs["system_instruction"])])
    return cmd


def get_venv(task, ckpt):
    """virtual environment might change based on the task and/or checkpoint"""
    venv_path = "/scratch/ad_shared/venv/k2h_native_venv" #! CHANGE BEFORE RELEASE
    logger.debug(f"Checkpoint `{ckpt}` is using virtual environment: `{venv_path}`")
    return venv_path


def create_lm_eval_sbatch_cmd(
        partition, nodes, ntasks, cpus_per_task, gpu, job_name, log_filepath,
        nodelist, wrap_cmd
    ):
    sbatch_cmd = [
        "sbatch",
        "--parsable",
        f"--partition={partition}",
        f"--nodes={nodes}",
        f"--ntasks={ntasks}",
        f"--cpus-per-task={cpus_per_task}",
        f"--gres={gpu}",
        f"--job-name={job_name}",
        f"--output={log_filepath}",
        f"--error={log_filepath}",
        f"--nodelist={nodelist}",
        f"--wrap={wrap_cmd}",
    ]
    return sbatch_cmd


def filter_checkpoints(checkpoints):
    filtered_checkpoints = []
    for ckpt_type, ckpt in checkpoints:
        if any(fnmatch(ckpt, pat) for pat in BLACKLIST_CHECKPOINTS):
            logger.warning(f"Skipping checkpoint: `{ckpt}`!")
            continue
        filtered_checkpoints.append((ckpt_type, ckpt))
    return filtered_checkpoints


def get_evaluation_commands(ckpt_type, ckpt, grouped_tasks, args):
    # highlight hyper-parameters used in evaluation
    logger.info("\tOther hyperparameters:")
    logger.info(f"\t\t- Seed: {args.seed}")
    logger.info(f"\t\t- Applying chat-template")

    ckpt_results_path = get_ckpt_results_path(ckpt_type, ckpt, args.results_path)
    logger.info(f"\t\t- Results will be saved to: {ckpt_results_path}")

    commands = []
    for (venv_path, harness, model_args, gen_args), tasks in grouped_tasks:
        harness_eval_cmd = create_eval_cmd(
            tasks=tasks,
            harness=harness,
            per_model_args=model_args,
            per_gen_args=gen_args,
            enable_chat_template=apply_chat_template(ckpt_type, ckpt),
            output_path=ckpt_results_path,
            seed=args.seed,
            limit=args.limit,
            lm_eval_verbosity=args.lm_eval_verbosity,
            system_instruction=args.system_instruction
        )
        # check if we need to use Ray for multi-node inference
        if getattr(args, "num_nodes_per_job", 1) > 1:
            num_nodes = args.num_nodes_per_job
            ray_setup_dir = shlex.quote(str(Path(__file__).resolve().parent))
            if harness == "lm_eval":
                python_executable = "python"
                run_harness = 'runpy.run_module("lm_eval", run_name="__main__", alter_sys=True)'
                harness_args = harness_eval_cmd[3:]  # Keep "run" and all flags.
            elif harness == "mt_eval":
                python_executable = "python"
                run_harness = 'runpy.run_module("mt_eval", run_name="__main__", alter_sys=True)'
                harness_args = harness_eval_cmd[1:]  # Remove only "mt-eval".
            else:
                raise ValueError(f"Invalid harness: {harness}")
            # Initialize the Ray job with CUDA's library path before vLLM creates actors.
            ray_driver_cmd = [
                python_executable,
                "-c",
                (
                    "import os, sys, ray, runpy; "
                    'ray.init(address=os.environ["RAY_ADDRESS"], runtime_env={"env_vars": '
                    '{"PYTHONPATH": os.environ["PYTHONPATH"], '
                    '"LD_PRELOAD": "/cm/shared/apps/cuda/12.8/lib64/libcudart.so.12"'
                    ' + (" " + os.environ["LD_PRELOAD"] if os.environ.get("LD_PRELOAD") else "")}}); '
                    + run_harness
                ),
                *harness_args,
            ]
            harness_cmd = (
                'ray_head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1) && '
                # Avoid cross-session collisions when multiple jobs share a
                # head node by deriving a stable, job-specific Ray port.
                'ray_port=$((20000 + ${SLURM_JOB_ID:-0} % 20000)) && '
                'ray_address="$ray_head_node:$ray_port" && '
                f'srun --nodes={num_nodes} '
                f'--ntasks={num_nodes} '
                '--ntasks-per-node=1 '
                'env '
                'RAY_ADDRESS="$ray_address" '
                'RAY_TMPDIR="/tmp/ray-${SLURM_JOB_ID}" '
                'RAY_gcs_rpc_server_reconnect_timeout_s=600 '
                'RAY_gcs_server_request_timeout_seconds=600 '
                f'PYTHONPATH={ray_setup_dir}"${{PYTHONPATH:+:$PYTHONPATH}}" '
                'VLLM_WORKER_MULTIPROC_METHOD=spawn '
                'ray symmetric-run '
                '--address "$ray_address" '
                f'--min-nodes={num_nodes} '
                f'-- {shlex.join(ray_driver_cmd)}'
            )
        else:
            harness_cmd = shlex.join(harness_eval_cmd)
        # use run harness_cmd whether with ray or without (single node inference)
        eval_cmd = ' && '.join([
            # activate virtual environment
            f"source {shlex.quote(f'{venv_path}/bin/activate')}",
            # run evaluation command
            harness_cmd,
            # run scorer commands (if needed)
            *create_scorer_cmds(tasks, ckpt_results_path)
        ])
        # wrap the commands in a subshell and check the exit code
        wrapped_eval_cmd = (
            f"( {eval_cmd}; exit_code=$?; "
            f"if [ $exit_code -eq 0 ]; then "
            f"echo '=== Finished `{tasks}` using `{harness}` harness ==='; "
            f"else echo '=== FAILED `{tasks}` using `{harness}` harness (exit $exit_code) ==='; fi; "
            f"exit $exit_code )"
        )
        commands.append(wrapped_eval_cmd)
    # sanity check
    assert len(commands) == len(grouped_tasks),\
        f"{len(commands)}) commands doesn't match ({len(grouped_tasks)}) grouped-tasks"
    return commands
