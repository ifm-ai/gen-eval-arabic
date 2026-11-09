"""
This file contains the configuration for the evaluation tasks and checkpoints
used in our jais-3-eval suite.
"""
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent

# ═════════════════════════════════════════════════════════════════════════════
# Pipeline Configuration
# ═════════════════════════════════════════════════════════════════════════════
SEED = 42
DTYPE="bfloat16"
GPU_MEM_UTIL = 0.90
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_MODEL_LEN = 1_048_576 #NOTE: set to the maximm
# ATTENTION_BACKEND = "TRITON_ATTN"
LM_EVAL_VERBOSITY = "DEBUG"
LOGGER_LEVEL = "DEBUG"
DEFAULT_PROFILE = "optimized"
# closing tags for the thinking traces for all models we support
THINKING_CLOSING_TAGS = (
    r"</think>|" # default thinking closing tag
    r"</think_faster>|" # Jais 3 low reasoning (old name)
    r"</think_fast>|" # Jais 3 medium reasoning (old name)
    r"</ifm\|think>|"  # Jais 3 high reasoning
    r"</ifm\|think_faster>" # Jais 3 low reasoning
    r"<channel\|>|" # Gemma 4 thinking closing tag
    r"<\|eom\|>|"
    r"</mm:think>|"
    r"to=user<\|message\|>|"
)
# location where the benchmark datasets are stored
LOCAL_BENCHMARK_PATH = ROOT_DIR / "local_benchmarks"
# path to the HuggingFace cache
HF_HUB_CACHE_PATH = Path("/jais3/.cache/huggingface/hub") #! CHANGE BEFORE RELEASE
# SBATCH Configuration (#? DELETE BEFORE RELEASE)
SBATCH_WAIT_TIME = 5
SBATCH_LOG_PATH = ROOT_DIR / "logs" / "sbatch"

# ═════════════════════════════════════════════════════════════════════════════
# TASKS
# ═════════════════════════════════════════════════════════════════════════════
# variable to set the max model length for each task
TASK_MAX_MODEL_LEN = {}

# Generations a task chains into one growing conversation; the window is sized to hold them all
# while each turn stays capped at the per-turn budget, so turn 1 cannot starve the rest.
TASK_NUM_TURNS = {
    "mt_bench": 2,          # answer Q1, then answer Q2 given its own A1
    "mt_bench_arabic": 2,   # answer Q1, then answer Q2 given its own A1
    "multi_if": 3,          # 3 chained turns, instructions cumulative through them
    "aragen_12_24": 2,      # Round 2 answers Q1, then Q2 given its own A1
    "aragen_03_25": 2,      # Round 2 answers Q1, then Q2 given its own A1
}

EN_TASKS = {
    ###################### EN_MCQ_TO_GEN ######################
    "mmlu_gen": ["instruct", "think"],
    "mmlu_pro_zero_shot": ["instruct", "think"],

    ###################### EN_IF ######################
    "ifeval": ["instruct", "think"],

    ###################### EN_MATH_GEN ######################
    "gsm8k-zeroshot": ["instruct", "think"],
    "gsm8k_cot_zeroshot": ["instruct", "think"],
    "gpqa_diamond_zeroshot_gen": ["instruct", "think"],
    "aime25_modified_repeats_8": ["instruct", "think"],
    "aime26_modified_repeats_8": ["instruct", "think"],
    "minerva_math500_reasoning": ["instruct", "think"],

    ###################### EN_FACTUALITY_GEN ######################
    "simpleqa_verified": ["instruct", "think"],
    "aa_omniscience_gen": ["instruct", "think"],
    "freshqa": ["instruct", "think"],

    ###################### EN_LONG ######################
    "aa_lcr_v1.1_en": ["instruct", "think"],

    ###################### EN_MULTITURN ######################
    "multichallenge": ["instruct", "think"],
    "mt_bench": ["instruct", "think"],
    "multi_if": ["instruct", "think"],

    ###################### EN_SAFETY ######################
    "harmbench": ["instruct", "think"],
    "harmbench_contextual": ["instruct", "think"],
    "harmbench_copyright": ["instruct", "think"],
    "xstest": ["instruct", "think"],
}

EN_CODING = {
    "humaneval_instruct": ["instruct", "think"],
    "mbpp_instruct": ["instruct", "think"],
}

EN_TASKS = (
    EN_TASKS
    | EN_CODING
)

AR_TASKS = {
    ###################### AR_MATH_GEN ######################
    "arabic_gsm8k-zeroshot": ["instruct", "think"],

    ###################### AR_IF ######################
    "ara_ifbench": ["instruct", "think"],
    "ifeval_ar": ["instruct", "think"],

    ###################### AR_TRANSLATION_GEN ######################
    "trans_flores200_plus": ["instruct", "think"],

    ###################### AR_TRANSLITERATION_GEN ######################
    "darija_transliteration": ["instruct", "think"],
    "egyptian_transliteration": ["instruct", "think"],

    ###################### AR_MCQ_TO_GEN ######################
    "arabicmmlu_gen": ["instruct", "think"],
    "arabicmmlu_with_diacritics_gen": ["instruct", "think"],
    "mmlu_ar_dialects_only_gen": ["instruct","think"],
    "aradice_culture_gen": ["instruct", "think"],
    "arab_culture_gen": ["instruct", "think"],
    "belebele_arabic_gen": ["instruct", "think"],
    "gat_math_gen": ["instruct", "think"],
    # "3LM_gen": ["instruct", "think"], # ! SATURATED!!!
    "arastem_gen": ["instruct", "think"],
    "islamic_qa_QASI_gen": ["instruct", "think"],
    "islamic_qa_IslamicEval2015_1B_gen": ["instruct", "think"],
    "islamic_qa_palmX2025_2_gen": ["instruct", "think"],
    "poetry_analysis_gen": ["instruct", "think"],
    # "dialect_id_MADAR_gen": ["instruct","think"],
    # "dialect_id_MADAR_gen_light": ["instruct","think"],
    "dialect_id_QADI_gen": ["instruct","think"],
    "alyah-emirati-benchmark_gen": ["instruct", "think"],
    "arabic_cuisine_gen": ["instruct", "think"],
    "medarabench_gen": ["instruct", "think"],
    "arabic_linguistic_gen": ["instruct", "think"],
    # "mmlu_pro_ar_zeroshot": ["instruct", "think"], #!Recently added
    
    ###################### AR_FACTUALITY_GEN ######################
    "simpleqa_verified_ar": ["instruct", "think"],
    "halluscore": ["instruct", "think"],
    # "freshqa_ar": ["instruct", "think"], #? Should be added

    ###################### AR_MULTITURN ######################
    "mt_bench_arabic": ["instruct", "think"],

    ###################### AR_HELM ######################
    "arabic_legal_helm_gen": ["instruct", "think"],
    "arabic_finance_helm_gen": ["instruct", "think"],

    ###################### AR_LONG ######################
    "aa_lcr_v1.1_ar": ["instruct", "think"],

    ###################### AR_SAFETY ######################
    "arasafe_harmful": ["instruct", "think"],
    "arab_safety_regional": ["instruct", "think"],

    ###################### Leaderboards ######################
    # "aragen_12_24": ["instruct", "think"],
    # "aragen_03_25": ["instruct", "think"],
}


# ═════════════════════════════════════════════════════════════════════════════
# Harness routing
# ═════════════════════════════════════════════════════════════════════════════
DEFAULT_HARNESS = "lm_eval"
TASK_TO_HARNESS = {
    ######################### mt-evaluation-harness ############################
    "multichallenge": "mt_eval",
    "mt_bench": "mt_eval",
    "mt_bench_arabic": "mt_eval",
    "multi_if": "mt_eval",
    "aragen_12_24": "mt_eval",
    "aragen_03_25": "mt_eval",
}


TASK_TO_SCORER = {} #? SHOULD BE REMOVED

# ═════════════════════════════════════════════════════════════════════════════
# Default checkpoints
# ═════════════════════════════════════════════════════════════════════════════
JAIS_CHECKPOINTS = {
    # Quantized checkpoints
    "IFM/Jais-3-3.7B-FP8",
    "IFM/Jais-3-7B-FP8",
    "IFM/Jais-3-32B-FP8",
    "IFM/Jais-3-MoVA-36B-A4B-FP8",
    "IFM/Jais-3-375B-A23B-FP8",

    # Main checkpoints
    "inception42/Jais-3-0.9B",
    "inception42/Jais-3-3.7B",
    "inception42/Jais-3-7B",
    "inception42/Jais-3-32B",
    "inception42/Jais-3-MoVA-36B-A4B",
    "inception42/Jais-3-375B-A23B",
}

INSTRUCT_CHECKPOINTS={
    # 1B models
    "CohereLabs/tiny-aya-global",
    "meta-llama/Llama-3.2-1B-Instruct",
    "tiiuae/Falcon-H1-1.5B-Deep-Instruct",
    "tiiuae/Falcon-H1-0.5B-Instruct",
    "hammh0a/Hala-1.2B",
    
    # 4B models
    "ibm-research/granite-4.0-h-3b-ar",
    "tiiuae/Falcon-H1-3B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    
    # 7B models
    "google/gemma-4-E4B-it",
    "google/gemma-4-12B-it",
    "tiiuae/Falcon-H1-7B-Instruct",
    "inception42/Jais-2-8B-Chat",
    "humain-ai/ALLaM-7B-Instruct-preview",
    "QCRI/Fanar-1-9B-Instruct",
    "silma-ai/SILMA-9B-Instruct-v1.0",
    "Navid-AI/Yehia-7B-preview",
    "CohereLabs/aya-expanse-8b",
    "Applied-Innovation-Center/Karnak-6B-v1.0",
    "hammh0a/Hala-9B", #! TAKES TOO LONG!!!
    "FreedomIntelligence/AceGPT-v2-8B-Chat",

    # 32B models
    "CohereLabs/aya-23-8B",
    "QCRI/Fanar-2-27B-Instruct",
    "inception42/Jais-2-70B-Chat",
    "tiiuae/Falcon-H1-34B-Instruct",
    "Applied-Innovation-Center/Karnak-40B-v1.0",
}

THINK_CHECKPOINTS = {
    # 1B models
    "Qwen/Qwen3.5-2B"
    
    # 4B models
    "Qwen/Qwen3.5-4B",
    "google/gemma-4-E2B-it",
    "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",

    # 7B models
    "google/gemma-4-E4B-it",
    "google/gemma-4-12B-it",
    "Qwen/Qwen3.5-9B",

    # 32B models
    "google/gemma-4-31B-it",
    "meta-models/Muse-Glimmer-30B",
    "Qwen/Qwen3.8-27B",
    "QCRI/Fanar-2-27B-Instruct",
    "IFM/K2-Horizon-32B",

    # Large Models
    "Qwen/Qwen3.8-Flash-Next",
    "Qwen/Qwen3.5-397B-A17B",
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
    "MiniMaxAI/MiniMax-M3",
    "mistralai/Mistral-Small-4-119B-2603"
}

# All JAIS checkpoints are THINK models
THINK_CHECKPOINTS = (
    THINK_CHECKPOINTS |
    JAIS_CHECKPOINTS
)

# All THINK models are INSTRUCT models
INSTRUCT_CHECKPOINTS = (
    INSTRUCT_CHECKPOINTS |
    THINK_CHECKPOINTS
)

CHECKPOINTS= (
    INSTRUCT_CHECKPOINTS |
    THINK_CHECKPOINTS
)

# ═════════════════════════════════════════════════════════════════════════════
# Blacklisting
# ═════════════════════════════════════════════════════════════════════════════
BLACKLIST_CHECKPOINTS = {} # add checkpoints to blacklist here

# ═════════════════════════════════════════════════════════════════════════════
# JUDGE Configurations
# ═════════════════════════════════════════════════════════════════════════════
JUDGE_PREFIX = "judge__" # prefix to be used for judge-related files
DEFAULT_JUDGE = "google/gemma-4-E2B-it"
JUDGE_PARAMS = {
    "seed": 42,
    "max_tokens": 8192,
    "temperature": 0,
    "enable_thinking": False,
}
JUDGE_BATCH_SIZE = 16384
JUDGE_GPU_MEMORY_UTILIZATION = 0.9
#REASONING_TAIL_NUM_CHARS_TO_KEEP = 2000 # tail chars to keep from reasoning when no response is provided
MAX_RESPONSE_LENGTH = 32768 # number of characters to keep from too long responses
