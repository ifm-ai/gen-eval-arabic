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


CUDA_RUNTIME_LIBRARY = "/cm/shared/apps/cuda/12.8/lib64/libcudart.so.12"
def preload_cuda_runtime():
    """Make CUDA runtime symbols available before a Ray worker imports vLLM."""
    import ctypes
    ctypes.CDLL(CUDA_RUNTIME_LIBRARY, mode=ctypes.RTLD_GLOBAL)


def set_environment_variables():
    logger.info("Setting environment variables...")
    return ' && '.join([
        'umask 002',
        # so that floresspmmodel is not downloaded again, as downnload url is blocked
        'export SACREBLEU="/scratch/ad_shared/data"',
        # to handle the M4 bandwidth issue
        "export HF_HUB_OFFLINE=1",
        "export VLLM_ATTENTION_BACKEND=TRITON_ATTN",
        # to help with reproducibility
        # "export VLLM_BATCH_INVARIANT=1", #NOTE: Removed as of 14/05/2026
        # to allow code evaluation in HuggingFace models
        "export HF_ALLOW_CODE_EVAL=1",
        # to handle HuggingFace metrics cache issue
        'export HF_METRICS_CACHE="$HF_HOME/metrics_unique/metrics_$(openssl rand -hex 8)"; mkdir -p "$HF_METRICS_CACHE"',
        # redirect torch/triton compile caches to local user-specific disk to avoid NFS stale file handle errors
        #### George: Added to fix PermissionError on /tmp//torchinductor_cache: when $USER is empty (cron/non-login shells) the path collapses to a shared /tmp//torchinductor_cache owned by another user. Use $(id -un) (never empty), redirect both the inductor and triton caches (the autotune cache reads TORCHINDUCTOR_CACHE_DIR via cache_dir()), and pre-create the dirs so they are owned by us.
        'export _EVAL_TMP="/tmp/${SLURM_JOB_ID:-manual}"',
        'export TORCHINDUCTOR_CACHE_DIR="$_EVAL_TMP/torchinductor_cache"',
        'export TRITON_CACHE_DIR="$_EVAL_TMP/triton_cache"',
        'export FLASHINFER_WORKSPACE_BASE="$_EVAL_TMP/flashinfer"',
        'export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY="${VLLM_RAY_EXTRA_ENV_VARS_TO_COPY:+${VLLM_RAY_EXTRA_ENV_VARS_TO_COPY},}FLASHINFER_WORKSPACE_BASE"',
        'export NCCL_DEBUG=WARN',
        # 'export NCCL_NVLS_ENABLE=0',
        # # flashinfer JIT lock deadlock/ESTALE on NFS (~/.cache/flashinfer)
        # 'export FLASHINFER_CACHE_DIR="$_EVAL_TMP/flashinfer_cache"',
        # # default temp dir for everything else (e.g. fallback inductor paths)
        'export TMPDIR="$_EVAL_TMP"',
        'mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"',
        ########
    ])


def expand_checkpoint(checkpoint_str):
    # check if it's a local checkpoint
    if Path(checkpoint_str).exists():
        # check if it's a single checkpoint
        if (Path(checkpoint_str) / "config.json").is_file():
            return [checkpoint_str]
        # parse all checkpoints in the directory
        checkpoint_paths = [
            str(p.parent)
            for p in Path(checkpoint_str).rglob("*/config.json")
        ]
        return checkpoint_paths
    else:
        # it's a HuggingFace checkpoint, return as is
        return [checkpoint_str]


def resolve_checkpoints(checkpoints, default, msg):
    if checkpoints is None:
        return []
    if checkpoints == []:
        logger.info(msg)
        return default
    return checkpoints


def get_checkpoints(base_checkpoints, instruct_checkpoints, think_checkpoints):
    """
    Parses and combines the base, instruct, and think checkpoints into a
    single list of checkpoints to evaluate.
    """
    total_ckpts = []
    # handle base checkpoints
    base_checkpoints = resolve_checkpoints(
        base_checkpoints,
        DEFAULT_BASE_CHECKPOINTS,
        "No base checkpoints were provided, loading all default base checkpoints."
    )
    # handle instruct checkpoints
    instruct_checkpoints = resolve_checkpoints(
        instruct_checkpoints,
        DEFAULT_INSTRUCT_CHECKPOINTS,
        "No instruct checkpoints were provided, loading all default instruct checkpoints."
    )
    # handle think checkpoints
    think_checkpoints = resolve_checkpoints(
        think_checkpoints,
        DEFAULT_THINK_CHECKPOINTS,
        "No thinking checkpoints were provided, loading all default thinking checkpoints."
    )

    # groups to help organize the parsing and validation of checkpoints
    groups = [
        ("base", base_checkpoints, DEFAULT_BASE_CHECKPOINTS),
        ("instruct", instruct_checkpoints, DEFAULT_INSTRUCT_CHECKPOINTS),
        ("think", think_checkpoints, DEFAULT_THINK_CHECKPOINTS),
    ]
    # iterate over checkpoints and validate them
    for ckpt_type, input_ckpts, allowed_set in groups:
        # expand allowed_set checkpoints (if possible)
        allowed_set = {
            expanded
            for ckpt in allowed_set
            for expanded in expand_checkpoint(ckpt)
        }
        # expand user-defined checkpoints (if possible)
        input_ckpts_expanded = []
        for ckpt in input_ckpts:
            expanded_ckpts = expand_checkpoint(ckpt)
            if len(expanded_ckpts) > 1:
                logger.info(
                    f"Expanding local checkpoint: {ckpt} into {len(expanded_ckpts)} checkpoints."
                )
            for expanded in expanded_ckpts:
                input_ckpts_expanded.append(expanded)
        # iterate over expanded checkpoints and check validity
        for ckpt in input_ckpts_expanded:
            if ckpt not in allowed_set:
                logger.error(
                    f"Checkpoint '{ckpt}' is not a valid `{ckpt_type}` checkpoint, skipping!"
                )
            else:
                total_ckpts.append((ckpt_type, ckpt))
    return total_ckpts


def get_ckpt_results_path(
        ckpt_type, ckpt, results_root_path,
        ckpt_root_path=Path("/jais3/checkpoints"), results_suffix=""
    ):
    # local checkpoint
    if Path(ckpt).exists():
        ckpt_relative_path = Path(ckpt).relative_to(ckpt_root_path).as_posix()
        if ckpt_type == "base":
            suffix = ''
        elif ckpt_type == "instruct":
            suffix = "_thinking_low"
        elif ckpt_type == "think":
            suffix = "_thinking_high"
        else:
            raise ValueError(f"Invalid checkpoint type: {ckpt_type}")
        # remove "jais3_experiments" when found (#! BUG-Prune?)
        ckpt_relative_path = ckpt_relative_path.replace('jais3_experiments/', '')
        ckpt_out_foldername = str(ckpt_relative_path).replace('/', '_') + suffix
        ckpt_results_path = results_root_path / ckpt_out_foldername
    # HuggingFace checkpoint
    else:
        suffix = "_thinking_high" if ckpt_type == "think" else ''
        ckpt_out_foldername = ckpt.split("/")[-1] + suffix
        ckpt_results_path = results_root_path / ckpt_out_foldername
    return ckpt_results_path.with_name(ckpt_results_path.name + results_suffix)


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
        
        
    #     if not override and any(
    #         re.match(rf"samples_{re.escape(subtask)}_\d+.*\.jsonl$", filepath.name)
    #         for subtask in subtasks
    #         for filepath in ckpt_output_samples
    #     ):
    #         logger.info(
    #             f"\tTask `{task}` has been already evaluated, skipping!"
    #         )
    #     else:
    #         remaining_tasks.append(task)
    # logger.info(f"\tThere are {len(remaining_tasks)} remaining tasks:")
    # for task in remaining_tasks:
    #     logger.info(f"\t\t- {task}")
    # return remaining_tasks


def get_model_args(ckpt_type, ckpt, model_length, args):
    model_args = {
        "pretrained": ckpt,
        "dtype": args.dtype,
        "seed" : args.seed,
        "gpu_memory_utilization": args.gpu_mem_utilization,
        "tensor_parallel_size": args.gpu_per_job,
        #! the following argument was added to handle `bbq-8b-mid-3-final` ckpt
        "max_model_len": model_length,
        "log_completion_metadata": not args.ignore_completion_metadata,
    }
    if getattr(args, "num_nodes_per_job", 1) > 1:
        model_args["distributed_executor_backend"] = "ray"
    models_require_expert_parallel = [
        "IFM/Jais-3-375B-A23B-FP8",
    ]
    if ckpt in models_require_expert_parallel:
        model_args["enable_expert_parallel"] = True
        model_args["enable_flashinfer_autotune"] = False


    # set chat_template_args
    if ckpt_type == "instruct":
        if "Fanar-2-27B-Instruct".lower() in ckpt.lower():
            model_args["chat_template_args"] = {
                "no_thinking": True
            }
        else:
            model_args["enable_thinking"] = False
    elif ckpt_type == "think":
        model_args["enable_thinking"] = True
        model_args["think_end_token"] = args.thinking_closing_tags
        if "Qwen3.8-27B".lower() in ckpt.lower():
            model_args["chat_template_args"] = {
                "reasoning_effort": "xhigh"
            }
        elif "Muse-Glimmer-30B".lower() in ckpt.lower():
            model_args["chat_template_args"] = {
                "reasoning_strength": "xhigh"
            }
    # handle "jais 3" checkpoints
    if ckpt_type != "base" and ("sft" in ckpt.lower() or "bbq" in ckpt.lower() or "k2-horizon" in ckpt.lower()):
        model_args["enable_thinking"] = True
        model_args["think_end_token"] = args.thinking_closing_tags
        model_args["chat_template_args"] = {
            "reasoning_effort": "low" if ckpt_type == "instruct" else "high"
        }
    if ckpt_type in {"instruct", "think"} and "nemotron" in ckpt.lower():
        model_args["chat_template_args"] = {
            "enable_thinking": ckpt_type == "think"
        }
        # Nemotron-H has eight Mamba groups, so tensor parallelism must divide
        # eight. Use pipeline parallelism for allocations larger than TP8.
        if args.gpu_per_job >= 8 and args.gpu_per_job % 8 == 0:
            nemotron_tp_size = 8
        elif 8 % args.gpu_per_job == 0:
            nemotron_tp_size = args.gpu_per_job
        else:
            raise ValueError(
                "Nemotron-H requires --gpu-per-job to be a divisor or "
                "multiple of its 8 Mamba groups"
            )
        model_args["tensor_parallel_size"] = nemotron_tp_size
        nemotron_pp_size = args.gpu_per_job // nemotron_tp_size
        if nemotron_pp_size > 1:
            model_args["pipeline_parallel_size"] = nemotron_pp_size
    if "minimax-m3" in ckpt.lower():
        # MiniMax-M3's sparse-attention backend requires 128-token KV-cache
        # blocks. These evals are text-only, so skip loading its vision tower.
        model_args["block_size"] = 128
        model_args["language_model_only"] = True

        if args.gpu_per_job > 8:
            if args.gpu_per_job % 8 != 0:
                raise ValueError(
                    "MiniMax-M3 multi-node needs --gpu-per-job to be a multiple "
                    f"of 8 (TP=8 per node); got {args.gpu_per_job}"
                )
            model_args["tensor_parallel_size"] = 8
            model_args["pipeline_parallel_size"] = args.gpu_per_job // 8

        # MiniMax-M3 controls reasoning through its official chat-template
        # argument. Keep the generic flag aligned for harness integrations that
        # also inspect it.
        if ckpt_type == "instruct":
            model_args["enable_thinking"] = False
            model_args["chat_template_args"] = {"thinking_mode": "disabled"}
        elif ckpt_type == "think":
            model_args["enable_thinking"] = True
            model_args["chat_template_args"] = {"thinking_mode": "enabled"}
    
    if "mistral-small-4-119b-2603" in ckpt.lower():
        if args.gpu_per_job > 8:
            if args.gpu_per_job % 8 != 0:
                raise ValueError(
                    "mistral-small-4-119b-2603 multi-node needs --gpu-per-job to be a multiple "
                    f"of 8 (TP=8 per node); got {args.gpu_per_job}"
                )
            model_args["tensor_parallel_size"] = 8
            model_args["pipeline_parallel_size"] = args.gpu_per_job // 8
        
        if ckpt_type == "instruct":
            model_args["enable_thinking"] = False
            model_args["chat_template_args"] = {"reasoning_effort": "none"}
        elif ckpt_type == "think":
            model_args["enable_thinking"] = True
            model_args["chat_template_args"] = {"reasoning_effort": "high"}

    # #!Testing the 100B checkpoint:
    # if "100b" in ckpt.lower():
    #   model_args["enforce_eager"] = True
    if "qwen3.5-397b" in ckpt.lower():
      model_args["max_num_seqs"] = 128
    
    if "falcon-h1-34b-instruct" in ckpt.lower():
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

        if config_file is None:
            return None
        try:
            with config_file.open("r", encoding="utf-8") as f:
                cfg = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

        keys = ["max_position_embeddings", "max_sequence_length", "n_positions", "seq_length"]
        found = list(_find_keys_recursive(cfg, keys))
        if found:
            return max(found)
        return None
    requested = args.max_model_length
    if requested is None:
        requested = task_requested_model_length
    cfg_len = _read_config_max_length(Path(ckpt), args.hf_hub_cache_path)
    if cfg_len is not None:
        return min(requested, cfg_len)
    else:
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
    model_hyperparams["temperature"] = model_hyperparams.get("temperature", TEMPERATURE)
    return model_hyperparams


def get_gen_args(ckpt_model_length, model_hyperparams, args):
    gen_args = []
    # an explicit --temperature overrides the profile's value
    temperature = args.temperature if args.temperature is not None else model_hyperparams["temperature"]
    gen_args.append(f"temperature={temperature}")
    gen_args.append(f"do_sample={temperature != 0}")

    # --gen-args replaces the profile bundle unless --merge-gen-args is set.
    if getattr(args, "gen_args", None):
        if getattr(args, "merge_gen_args", False):
            override_keys = {arg.split("=", 1)[0].strip() for arg in args.gen_args.split(",")}
            for key, value in model_hyperparams.items():
                if key not in {"temperature", "do_sample"} | override_keys:
                    gen_args.append(f"{key}={value}")
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
    elif harness == "minihelmet_eval":
        return [
            "python", str(MINIHELMET_WRAPPER),
        ]
    elif harness == "humanpref":
        return [
            str(Path(HUMANPREF_VENV) / "bin" / "python"),
            str(HUMANPREF_WRAPPER),
        ]
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
        "--batch_size", str(kwargs["batch_size"]),
        "--output_path", str(Path(output_path) / "results.json"),
        "--log_samples",
        "--verbosity", kwargs["lm_eval_verbosity"],
        "--trust_remote_code",
        "--device", "cuda:0"
    ])
    if harness != "humanpref":
        cmd.append("--confirm_run_unsafe_code")  # necessary for code tasks (e.g. humaneval)
    if enable_chat_template:
        cmd.append("--apply_chat_template")
    if kwargs.get("limit") is not None:
        cmd.extend(["--limit", str(kwargs["limit"])])
    if kwargs.get("system_instruction") is not None:
        cmd.extend(["--system_instruction", str(kwargs["system_instruction"])])
    return cmd


def get_venv(task, ckpt):
    """virtual environment might change based on the task and/or checkpoint"""
    #     logger.debug(f"Using miniHELMET Virtual Environment: `{MINIHELMET_VENV}`")
    #     return MINIHELMET_VENV
    # elif TASK_TO_HARNESS.get(task, DEFAULT_HARNESS) == "humanpref":
        # Native-arch checkpoints (K2-Horizon / Nemotron / MiniMax-M3) only load
        # in the k2h vLLM-0.28 venv; humanpref runs in-process on one node, so it
        # runs there too. Everything else uses the default HumanPref venv.
    #     logger.debug(f"Using HumanPref Virtual Environment: `{HUMANPREF_VENV}`")
    #     return HUMANPREF_VENV

    # venv_path = "/scratch/ad_shared/.US_venv/.vllm-bbq-0518_py3.12"
    # ! Anwar: replace following part with one general venv (prior to the release)
    venv_path = "/scratch/ad_shared/venv/k2h_native_venv"

    models_with_xllm = [
        # xml rl 1B
        "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-1Bcv5c189_sft_reasoningv9.5.2k2horCT_FTPO_es035_RLVRIfI3rabTashkeelMathCode16k_8n_checkpoint_199_to_hf",
        # xml rl 4B
        "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-4Bc10248_sft_reasoningv9.5.1k2horCT_FTPO_es035_RLVRIfI3rabTashkeelMathCode16k_8n_checkpoint_199_to_hf",
        # xml rl 8B
        "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-8Bc10248_sft_reasoningv9.5.2k2horCT_FTPO_es035_RLVRIfI3rabTashkeelMathCode16k_8n_checkpoint_199_to_hf",
        # xml ftpo 36B
        "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-MoVA36B_sft_ifm_20260916_FTPO_es035_hf_FTPO_es035",
        # json rl 32B
        "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-32Bc10248_sft_reasoningv9.5.2_FTPO_es035_RLVRIfI3rabTashkeelMathCode16k_checkpoint_199_to_hf",
        # json rl 8B
        "/jais3/checkpoints/jais3_experiments/sft/A13/A13cs_A13x1-8Bc10248_sft_reasoningv9.5.2_FTPO_es035_RLVRIfI3rabTashkeelMathCode16k_8n_checkpoint_199_to_hf",
    ]
    if any(x.lower() in ckpt.lower() for x in models_with_xllm):
        venv_path = "/scratch/ad_shared/.US_venv/.vllm-bbq-0518_py3.12_updated"
    elif any(x in ckpt.lower() for x in ("jais-2", "qwen3.5")):
        venv_path = "/scratch/ad_shared/.jais2_venv_eval_harness"
    elif "gemma-4-12b-it" in ckpt.lower():
        venv_path = "/scratch/ad_shared/.gemma4_12b_venv"
    elif "falcon-h1-1.5b" in ckpt.lower() or "granite" in ckpt.lower():
        venv_path = "/scratch/ad_shared/.US_venv/.vllm-bbq-0518_py3.12/" 
    elif "falcon-h1-3b-instruct" in ckpt.lower() or "falcon-h1-7b-instruct" in ckpt.lower() or "falcon-h1-34b-instruct" in ckpt.lower() or "fanar" in ckpt.lower():
        venv_path = "/scratch/ad_shared/.gemma4_venv"
    elif "gemma-4" in ckpt.lower():
        venv_path = "/scratch/ad_shared/.gemma4_venv"
    elif "qwen3.8-flash-next" in ckpt.lower():
        venv_path = "/scratch/ad_shared/.eval-muse-glimmer-30b"
    elif ("qwen3.6" in ckpt.lower()) or ("qwen3.8" in ckpt.lower()):
        venv_path = "/scratch/ad_shared/.qwen36"
    #BBQ Mid-1
    elif ("bbq" in ckpt.lower() and "mid1" in ckpt.lower()):
        venv_path = "/scratch/ad_shared/.US_venv/.vllm-patched_xllm-moe_cu128_py3.12"
    #BBQ Mid-2
    elif ("bbq" in ckpt.lower() and "mid2" in ckpt.lower()):
        venv_path = "/scratch/ad_shared/.US_venv/.vllm-bbq-0511_py3.12"
    #BBQ Mid-3
    elif ("bbq" in ckpt.lower() and "mid3" in ckpt.lower()):
        venv_path = "/scratch/ad_shared/.US_venv/.vllm-bbq-0518_py3.12"
    #Any other BBQ checkpoint (e.g. mid4)
    elif ("bbq" in ckpt.lower()):
        venv_path = "/scratch/ad_shared/.US_venv/.vllm-bbq-0518_py3.12"
    # elif ("100b" in ckpt.lower()):
    #     venv_path = "/scratch/ahmed.frikha/jais3_a0_eval/venv-vllm"
    elif ("muse-glimmer" in ckpt.lower()) or ("Mistral-Small-4-119B-2603".lower() in ckpt.lower()):
        venv_path = "/scratch/ad_shared/.eval-muse-glimmer-30b"
    
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
    logger.info(f"\t\t- Batch size: {args.batch_size}")
    logger.info(f"\t\t- Applying chat-template: {apply_chat_template(ckpt_type, ckpt)}")

    ckpt_results_path = get_ckpt_results_path(
        ckpt_type, ckpt, args.results_path,
        results_suffix=getattr(args, "results_suffix", ""), #! ANWAR: REDUNDANT FEATURE, REMOVE LATER
    )

    commands = []
    for (venv_path, harness, model_args, gen_args), tasks in grouped_tasks:
        harness_eval_cmd = create_eval_cmd(
            tasks=tasks,
            harness=harness,
            per_model_args=model_args,
            per_gen_args=gen_args,
            enable_chat_template=apply_chat_template(ckpt_type, ckpt),
            output_path=ckpt_results_path,
            batch_size=args.batch_size,
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
                # minihelmet/humanpref use: python /path/to/wrapper.py ...
                python_executable = harness_eval_cmd[0]
                script_path = harness_eval_cmd[1]
                run_harness = (
                    f"sys.path.insert(0, {str(Path(script_path).resolve().parent)!r}); "
                    f"runpy.run_path({script_path!r}, run_name='__main__')"
                )
                harness_args = harness_eval_cmd[2:]
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
