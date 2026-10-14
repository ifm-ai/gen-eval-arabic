"""
This file contains the configuration for the evaluation tasks and checkpoints
used in our jais-3-eval suite.
"""
from pathlib import Path
REPO_DIR = Path(__file__).resolve().parent

# ═════════════════════════════════════════════════════════════════════════════
# Pipeline Configuration
# ═════════════════════════════════════════════════════════════════════════════
SEED = 42
TEMPERATURE = 0
DTYPE="bfloat16"
BATCH_SIZE = 16384
GPU_MEM_UTIL = 0.90
DEFAULT_MAX_MODEL_LEN = 1_048_576 # 32768 (changed on 12th of October 2026)
ATTENTION_BACKEND = "TRITON_ATTN"
LM_EVAL_VERBOSITY = "DEBUG"
LOGGER_LEVEL = "DEBUG"
DEFAULT_PROFILE = "optimized"
# closing tags for the thinking traces for all models we support
THINKING_CLOSING_TAGS = "</think>|</think_fast>|</think_faster>|<channel\|>|</ifm\|think>|<\|eom\|>|</mm:think>|to=user<\|message\|>|</ifm\|think_faster>"
# location where the evaluation results will be stored
RESULTS_PATH = Path("/jais3/eval/results")
# location where the benchmark datasets are stored
LOCAL_BENCHMARK_PATH = Path("/jais3/data/jais_2/benchmarks")
# path to the HuggingFace cache
HF_HUB_CACHE_PATH = Path("/jais3/.cache/huggingface/hub")
# SBATCH Configuration
SBATCH_WAIT_TIME = 5
SBATCH_LOG_PATH = REPO_DIR / "logs" / "sbatch"
# Nodes to exclude (used by the train team)
# EXCLUDED_NODES = (
#     "dgx-[01,05,13,14,15,17,32,49,50,54,55,56,57,60,61,63,64,65,66,67,68,"
#     "70,71,72,73,74,75,76,77,79,80,81,82,83,84,85,86,87,88,92,93,94,99,"
#     "102,105,106,107,110,115,135,136,137,139,144,146,147,148,149,151,152,153,154,"
#     "156,157,158,159]"
# )
# NOTE: This is the list of nodes that are excluded from the evaluation since
# the train team is using them for training as of 16/09/2026.
# EXCLUDED_NODES = (
#     "dgx-[01,02,03,04,05,08,09,10,11,13,14,17,18,20,21,22,23,25,26,27,28,30,32,"
#     "35,36,37,38,39,41,42,43,44,46,48,49,50,54,55,56,57,60,61,63,64,65,66,67,68,"
#     "69,70,71,72,73,74,75,76,77,79,80,81,82,83,84,85,86,87,89,90,92,93,95,96,97,"
#     "98,99,100,102,103,104,105,106,110,112,113,115,117,118,119,121,125,126,127,"
#     "129,130,132,133,134,135,136,137,142,144,145,146,147,149,151,152,155,156,"
#     "157,158,159]"
# )
EXCLUDED_NODES = ''
TEMPORARY_EXCLUDED_NODES = (
    "dgx-[06,12,34,36,39,40,46,50,58,59,69,91,97,100,108,155,152]"
)

# ═════════════════════════════════════════════════════════════════════════════
# TASKS
# ═════════════════════════════════════════════════════════════════════════════
TASK_MAX_MODEL_LEN = {}
# ! commented out on 12/10/2026
# TASK_MAX_MODEL_LEN = {
#     # ── 8K (8192) ────────────────────────────────────────────────────────────
#     "3LM": 8192,
#     "3LM_gen": 8192,
#     "arab_culture": 8192,
#     "arab_culture_gen": 8192,
#     "aradice_culture": 8192,
#     "aradice_culture_gen": 8192,
#     "arabicmmlu": 8192,
#     "arabicmmlu_gen": 8192,
#     "arabicmmlu_with_diacritics": 8192,
#     "arabicmmlu_with_diacritics_gen": 8192,
#     "arastem": 8192,
#     "arastem_gen": 8192,
#     "belebele_arabic": 8192,
#     "belebele_arabic_gen": 8192,
#     "dialect_id_QADI_gen": 8192,
#     "gat_math": 8192,
#     "gat_math_gen": 8192,
#     "islamic_qa": 8192,
#     "islamic_qa_gen": 8192,
#     "islamic_qa_QASI": 8192,
#     "islamic_qa_QASI_gen": 8192,
#     "islamic_qa_IslamicEval2015_1B": 8192,
#     "islamic_qa_IslamicEval2015_1B_gen": 8192,
#     "islamic_qa_palmX2025_2": 8192,
#     "islamic_qa_palmX2025_2_gen": 8192,
#     "mmlu_ar_dialects_only": 8192,
#     "mmlu_ar_dialects_only_gen": 8192,
#     "mmlu": 8192,
#     "mmlu_gen": 8192,
#     "poetry_analysis_gen": 8192,
#     "trans_flores200_plus": 8192,
#     "darija_transliteration": 8192,
#     "egyptian_transliteration": 8192,
#     "arabic_cuisine_gen": 8192,
#     "MedArabiQ_mcq_gen": 8192,
#     "medarabench_gen": 8192,
#     "arabic_linguistic_gen": 8192,
#     "uae_knowledge_all_gen": 8192,
#     "alghafa_native": 8192,
#     "alghafa_native_gen": 8192,
#     "exams_ar": 8192,
#     "exams_ar_gen": 8192,
#     "aratrust": 8192,
#     "aratrust_gen": 8192,
#     "alrage": 8192,
#     "arabic_mmlu_ht_gen": 8192,
#     "madinah_qa_gen": 8192,

#     # ── 32K (32768) ──────────────────────────────────────────────────────────
#     "arabic_gsm8k": 32768,            # can be lower after the fix
#     "arabic_gsm8k-zeroshot": 32768,   # can be lower after the fix
#     "gsm8k": 32768,                   # can be lower after the fix
#     "gsm8k_cot": 32768,               # can be lower after the fix
#     "gsm8k_reasoning_base": 32768,    # can be lower after the fix
#     "gsm8k-zeroshot": 32768,          # can be lower after the fix
#     "gsm8k_cot_zeroshot": 32768,      # can be lower after the fix
#     "mmlu_pro_zero_shot": 32768,
#     "mmlu_pro_ar_zeroshot": 32768,
#     "halluscore": 32768,
#     "multichallenge": 32768,
#     "mt_bench": 32768,
#     "mt_bench_arabic": 32768,
#     "humaneval": 32768,
#     "humaneval_instruct": 32768,
#     "ifeval": 32768,
#     "ifeval_ar": 32768,
#     "simpleqa_verified": 32768,
#     "simpleqa_verified_ar": 32768,
#     "freshqa": 32768,
#     "freshqa_ar": 32768,
#     "ara_ifbench": 32768,
#     "alyah-emirati-benchmark_gen": 32768,
#     "minerva_math500": 32768,
#     "minerva_math500_reasoning": 32768,
#     "harmbench": 32768,
#     "harmbench_contextual": 32768,
#     "harmbench_copyright": 32768,
#     "multi_if": 32768,
#     "arasafe_harmful": 32768,
#     "xstest": 32768,
#     "abb": 32768,
#     "aragen_12_24": 32768,
#     "aragen_03_25": 32768,
#     "mmmlu_ar": 32768,

#     # ── 64K (65536) ──────────────────────────────────────────────────────────
#     "aime24_modified": 1_048_576,
#     "aime25_modified": 1_048_576,
#     "aime26_modified": 1_048_576,
    
#     "aime26_modified_repeats_8": 1_048_576,
#     "aime25_modified_repeats_8": 1_048_576,
    
#     "hmmt_2025": 65536,
#     "mbpp": 65536,
#     "mbpp_instruct": 65536,
#     "aa_omniscience_gen": 65536,
#     "arab_safety_regional": 65536,

#     # ── 81K (81920) ──────────────────────────────────────────────────────────
#     "gpqa_diamond_zeroshot_gen": 81920,
    
#     # ── 131K (131072) ──────────────────────────────────────────────────────────
#     "longbench2_arabic_64k_gen": 131072,
#     "longbench2_english_64k_gen": 131072,
    
#     # ── 256K (262144) ──────────────────────────────────────────────────────────
#     "longbench2_arabic_128k_gen": 262144,
#     "longbench2_english_128k_gen": 262144,
#     "aa_lcr_v1.1_ar": 262144,
#     "aa_lcr_v1.1_en": 262144,

#     # ── miniHELMET (max input 131072 + 65536 thinking budget = 196608 window) ────
#     "recall_mini": 196608,
#     "rag_mini":    196608,
#     "longqa_mini": 196608,
#     "summ_mini":   196608,
#     "icl_mini":    196608,
#     "rerank_mini": 196608,
#     "cite_mini":   196608,
# }

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
    ###################### EN_MCQ ######################
    "hellaswag": ["base"],
    "mmlu": ["base"],
    "mmlu_pro": ["base"],
    # "gpqa_diamond_zeroshot": ["base"]

    ###################### EN_MCQ_TO_GEN ######################
    "mmlu_gen": ["instruct", "think"],
    "mmlu_pro_zero_shot": ["instruct", "think"],
    # "mmlu_us_gen": ["instruct", "think"] #NOTE: USED FOR DEBUGGING

    ###################### EN_IF ######################
    "ifeval": ["base", "instruct", "think"],

    ###################### EN_MATH_GEN ######################
    "gsm8k": ["base"],
    "gsm8k_cot": ["base"],
    "minerva_math500": ["base"],
    "gsm8k_reasoning_base": ["base"],
    "gsm8k-zeroshot": ["instruct", "think"],
    "gsm8k_cot_zeroshot": ["instruct", "think"],
    "gpqa_diamond_zeroshot_gen": ["instruct", "think"],
    # "aime24_modified": ["instruct", "think"],
    # "aime25_modified": ["instruct", "think"],
    # "aime26_modified": ["instruct", "think"],
    # "aime25_modified_repeats_8": ["instruct", "think"],
    # "aime26_modified_repeats_8": ["instruct", "think"],
    # "hmmt_2025": ["instruct", "think"], #! <--- needs a judge.
    "minerva_math500_reasoning": ["instruct", "think"],

    ###################### EN_REASONING ######################
    # "bbeh": ["instruct", "think"], #! Added on 21st of August 2026

    ###################### EN_FACTUALITY_GEN ######################
    "simpleqa_verified": ["instruct", "think"],
    "aa_omniscience_gen": ["instruct", "think"],
    "freshqa": ["instruct", "think"],

    ###################### EN_LONG ######################
    # "longbench2_english_64k_gen": ["instruct", "think"],
    # "longbench2_english_128k_gen": ["instruct", "think"],
    # "aa_lcr_v1.1_en": ["instruct", "think"],

    ###################### EN_MULTITURN ######################
    "multichallenge": ["instruct", "think"],
    "mt_bench": ["instruct", "think"],
    "multi_if": ["instruct", "think"],

    ###################### EN_SAFETY ######################
    "harmbench": ["instruct", "think"],
    "harmbench_contextual": ["instruct", "think"],
    "harmbench_copyright": ["instruct", "think"],
    "xstest": ["instruct", "think"], #!Recently added

    ###################### EN_miniHELMET ######################
    # "recall_mini": ["instruct", "think"],
    # "rag_mini": ["instruct", "think"],
    # "longqa_mini": ["instruct", "think"],
    # "summ_mini": ["instruct", "think"],
    # "icl_mini": ["instruct", "think"],
    # "rerank_mini": ["instruct", "think"],
    # "cite_mini": ["instruct", "think"],

    ###################### EN_HUMANPREF ######################
    # "representative_en": ["instruct", "think"],
    # "hard_en": ["instruct", "think"],
}

EN_CODING = {
    "humaneval": ["base"],
    "humaneval_instruct": ["instruct", "think"],
    "mbpp": ["base"],
    "mbpp_instruct": ["instruct", "think"],
}

EN_TASKS = (
    EN_TASKS
    | EN_CODING
)


AR_TASKS = {
    ###################### AR_MCQ ######################
    "arabicmmlu": ["base"],
    "arabicmmlu_with_diacritics": ["base"],
    "mmlu_ar_dialects_only": ["base"],
    "aradice_culture": ["base"],
    "arab_culture": ["base"],
    # "dialect_id_MADAR": ["base"],
    # "dialect_id_QADI": ["base"],
    # "poetry_analysis": ["base"],
    "hellaswag_ar": ["base"],
    "belebele_arabic": ["base"],
    "3LM": ["base"],
    "arastem": ["base"],
    # "gat": ["base"],
    "gat_math": ["base"],
    "islamic_qa": ["base"],
    "islamic_qa_QASI": ["base"],
    "islamic_qa_IslamicEval2015_1B": ["base"],
    "islamic_qa_palmX2025_2": ["base"],
    "alyah-emirati-benchmark": ["base"],
    "alghafa_native": ["base"],
    "exams_ar": ["base"],
    "aratrust": ["base"],
    "arabic_mmlu_ht": ["base"],
    "madinah_qa": ["base"],
    # "MedArabiQ_mcq": ["base"],
    # "medarabench": ["base"],
    # "arabic_linguistic_mcq": ["base"],

    ###################### AR_MATH_GEN ######################
    "arabic_gsm8k": ["base"],
    "arabic_gsm8k-zeroshot": ["instruct", "think"],

    ###################### AR_IF ######################
    "ara_ifbench": ["instruct", "think"],
    "ifeval_ar": ["instruct", "think"],

    ###################### AR_TRANSLATION_GEN ######################
    # "trans_MADAR": ["instruct", "think"],
    "trans_flores200_plus": ["instruct", "think"],
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
    # "3LM_gen": ["instruct", "think"],
    "arastem_gen": ["instruct", "think"],
    "islamic_qa_gen": ["instruct", "think"],
    "islamic_qa_QASI_gen": ["instruct", "think"],
    "islamic_qa_IslamicEval2015_1B_gen": ["instruct", "think"],
    "islamic_qa_palmX2025_2_gen": ["instruct", "think"],
    "poetry_analysis_gen": ["instruct", "think"],
    # "dialect_id_MADAR_gen": ["instruct","think"],
    # "dialect_id_MADAR_gen_light": ["instruct","think"],
    "dialect_id_QADI_gen": ["instruct","think"],
    "alyah-emirati-benchmark_gen": ["instruct", "think"],
    "arabic_cuisine_gen": ["instruct", "think"],
    # "MedArabiQ_mcq_gen": ["instruct", "think"], #! Removed on 3rd of September because of quality issues.
    "medarabench_gen": ["instruct", "think"],
    "arabic_linguistic_gen": ["instruct", "think"],
    #?=== Below are benchmarks found in OALLv2===
    # "alghafa_native_gen": ["instruct", "think"],
    # "exams_ar_gen": ["instruct", "think"],
    # "aratrust_gen": ["instruct", "think"],
    # "alrage": ["instruct", "think"],
    # "arabic_mmlu_ht_gen": ["instruct", "think"],
    # "madinah_qa_gen": ["instruct", "think"],
    # "mmlu_pro_ar_zeroshot": ["instruct", "think"], #!Recently added
    # "uae_knowledge_all_gen": ["instruct", "think"],
    
    ###################### AR_FACTUALITY_GEN ######################
    "simpleqa_verified_ar": ["instruct", "think"],
    "halluscore": ["instruct", "think"],
    # "freshqa_ar": ["instruct", "think"],

    ###################### AR_MULTITURN ######################
    "mt_bench_arabic": ["instruct", "think"],

    ###################### AR_HELM ######################
    "arabic_legal_helm_gen": ["instruct", "think"],
    "arabic_finance_helm_gen": ["instruct", "think"],

    ###################### AR_LONG ######################
    # "longbench2_arabic_64k_gen": ["instruct", "think"],
    # "longbench2_arabic_128k_gen": ["instruct", "think"],
    # "aa_lcr_v1.1_ar": ["instruct", "think"],

    ###################### AR_SAFETY ######################
    # "arasafe_harmful": ["instruct", "think"], #!Recently added
    # "arab_safety_regional": ["instruct", "think"], #!Recently added

    ###################### Leaderboards ######################
    # "abb": ["instruct", "think"], #!Recently added
    # "mmmlu_ar": ["instruct", "think"],
    # "aragen_12_24": ["instruct", "think"],
    # "aragen_03_25": ["instruct", "think"],

    ###################### AR_HUMANPREF ######################
    # "representative_ar": ["instruct", "think"],
    # "hard_ar": ["instruct", "think"],
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
    ###################### minihelmet-evaluation-harness #######################
    "recall_mini": "minihelmet_eval",
    "rag_mini": "minihelmet_eval",
    "longqa_mini": "minihelmet_eval",
    "summ_mini": "minihelmet_eval",
    "icl_mini": "minihelmet_eval",
    "rerank_mini": "minihelmet_eval",
    "cite_mini": "minihelmet_eval",
    ###################### Human-Pref-evaluation-harness #######################
    "representative_en": "humanpref",
    "representative_ar": "humanpref",
    "hard_en": "humanpref",
    "hard_ar": "humanpref",
}

MINIHELMET_PATH = REPO_DIR / "helmet-evaluation-harness"
MINIHELMET_WRAPPER = MINIHELMET_PATH / "minihelmet_wrapper.py"
MINIHELMET_VENV = "/scratch/ad_shared/venv/helmet-xllm023-jais3evals"
HELMET_DATA_ROOT = "/scratch/abhishek.maiti/HELMET/data"

HELMET_MINI_SUBTASKS = {
    "recall_mini": ["ruler_niah_mk_2_in65536", "ruler_niah_mk_2_in131072", "ruler_niah_mk_3_in65536", "ruler_niah_mk_3_in131072", "ruler_niah_mv_in65536", "ruler_niah_mv_in131072", "json_kv_in65536", "json_kv_in131072"],
    "rag_mini": ["kilt_nq_in65536", "kilt_nq_in131072", "kilt_triviaqa_in65536", "kilt_triviaqa_in131072", "kilt_hotpotqa_in65536", "kilt_hotpotqa_in131072", "kilt_popqa_3_in65536", "kilt_popqa_3_in131072"],
    "longqa_mini": ["narrativeqa_65236_in65536", "narrativeqa_130772_in131072", "infbench_qa_eng_65326_in65536", "infbench_qa_eng_130862_in131072", "infbench_choice_eng_65326_in65536", "infbench_choice_eng_130862_in131072"],
    "summ_mini": ["infbench_sum_eng_64136_in65536", "infbench_sum_eng_129672_in131072", "multi_lexsum_64836_in65536", "multi_lexsum_130372_in131072"],
    "icl_mini": ["icl_trec_coarse_3300shot_balance_in65536", "icl_trec_coarse_6600shot_balance_in131072", "icl_trec_fine_3200shot_balance_in65536", "icl_trec_fine_6400shot_balance_in131072", "icl_banking77_2900shot_balance_in65536", "icl_banking77_5900shot_balance_in131072", "icl_clinic150_3525shot_balance_in65536", "icl_clinic150_7050shot_balance_in131072", "icl_nlu_4080shot_balance_in65536", "icl_nlu_8296shot_balance_in131072"],
    "rerank_mini": ["msmarco_rerank_psg_in65536", "msmarco_rerank_psg_in131072"],
    "cite_mini": ["alce_asqa_345_in65536", "alce_asqa_700_in131072", "alce_qampari_345_in65536", "alce_qampari_700_in131072"],
}

TASK_TO_SCORER = {
}

# ── HumanPref ────────────────────────────────────────────────────────────────
HUMANPREF_PATH = REPO_DIR
HUMANPREF_WRAPPER = HUMANPREF_PATH / "humanpref_evals" / "humanpref_wrapper.py"
HUMANPREF_VENV = "/scratch/ad_shared/venv/helmet-xllm023-jais3evals"  # copy of helmet-xllm023 (vLLM 0.23 + xllm plugin) carrying this repo's lm_eval


# --tasks multichallenge mt_bench_arabic multi_if longbench2_arabic_128k_gen
# ═════════════════════════════════════════════════════════════════════════════
# Default checkp§oints
# ═════════════════════════════════════════════════════════════════════════════
DEFAULT_BASE_CHECKPOINTS = {
        # Jais 3 checkpoints
        "/jais3/checkpoints/k2v3-4B",
        "/jais3/checkpoints/k2v3-9B",

        # Cooldown and CPT
        "/jais3/checkpoints/jais3_experiments/cooldown",
        "/jais3/checkpoints/jais3_experiments/cpt",
        
	    # HuggingFace Checkpoints 
        "tiiuae/Falcon-H1-3B-Base",
        "meta-llama/Llama-3.2-3B",
        "tiiuae/Falcon-H1-7B-Base",
        "QCRI/Fanar-1-9B",
        "meta-llama/Llama-3.1-8B",
        "Qwen/Qwen3.5-4B-Base",
        "Qwen/Qwen3.5-9B-Base",
        "google/gemma-4-E2B",
        "google/gemma-4-E4B",

        # "FreedomIntelligence/AceGPT-v2-8B", #! FAILS
}

DEFAULT_INSTRUCT_CHECKPOINTS={
    "IFM/Jais-3-32B-Uno",
    "IFM/Jais-3-7B-Uno",
    "IFM/Jais-3-3.7B-Uno",
    "IFM/Jais-3-0.9B-Uno",
    "IFM/Jais-3-7B-FP8",
    "IFM/Jais-3-3.7B-FP8",
    "IFM/Jais-3-32B-FP8",
    "IFM/Jais-3-MoVA-36B-A4B-FP8",
    "IFM/Jais-3-375B-A23B-FP8",
    "google/gemma-4-E2B-it-custom",
    "google/gemma-4-E4B-it-custom",
    "google/gemma-4-12B-it-custom",
    "google/gemma-4-31B-it-custom",
    # 1B models
    "CohereLabs/tiny-aya-global",
    "Qwen/Qwen3.5-2B", #has thinking mode.
    # "meta-llama/Llama-3.2-1B-Instruct",
    "tiiuae/Falcon-H1-1.5B-Deep-Instruct",
    # "tiiuae/Falcon-H1-0.5B-Instruct",
    "hammh0a/Hala-1.2B",
    # "utter-project/EuroMoE-2.6B-A0.6B-Instruct-2512",
    "IFM/K2-Horizon-0.9B",
    
    # 4B models
    "Qwen/Qwen3.5-4B",
    "google/gemma-4-E2B-it",
    "ibm-research/granite-4.0-h-3b-ar",
    "tiiuae/Falcon-H1-3B-Instruct",
    # "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
    # "mistralai/Ministral-3-3B-Reasoning-2512",
    # "Applied-Innovation-Center/Karnak-6B-v1.0",
    # "meta-llama/Llama-3.2-3B-Instruct",
    "IFM/K2-Horizon-3.7B",
    "/jais3/checkpoints/K2-Horizon-0.9B-mid_2_47684",
    # 7B models
    "google/gemma-4-E4B-it",
    "google/gemma-4-12B-it",
    "Qwen/Qwen3.5-9B",
    "tiiuae/Falcon-H1-7B-Instruct",
    # "inceptionai/Jais-2-8B-Chat",
    "inception42/Jais-2-8B-Chat",
    "humain-ai/ALLaM-7B-Instruct-preview",
    "QCRI/Fanar-1-9B-Instruct",
    # "hammh0a/Hala-9B",
    "IFM/K2-Horizon-7B",

    # 32B models
    "google/gemma-4-31B-it",
    # "meta-models/Muse-Glimmer-30B",
    "Qwen/Qwen3.8-27B",
    "QCRI/Fanar-2-27B-Instruct",
    "inception42/Jais-2-70B-Chat",
    "IFM/K2-Horizon-32B",
    "IFM/K2-Horizon-MoVA-36B-A4B",
    "IFM/K2-Horizon-375B-A23B",


    # SFT Checkpoints
    "/jais3/checkpoints/jais3_experiments/sft", #! DEPRECATED
    "/jais3/checkpoints/jais3_experiments/sft_eval",
    
    "/jais3/checkpoints/jais3_experiments/release_checkpoints/1b/cpt/hf_checkpoints/artifacts/checkpoint_11141_to_hf/",
    "/jais3/checkpoints/jais3_experiments/release_checkpoints/1b/cpt_sft/hf_checkpoints/artifacts/checkpoint_11141_to_hf/",
    "/jais3/checkpoints/jais3_experiments/release_checkpoints/1b_xml/sft/hf_checkpoints/artifacts/checkpoint_10711_to_hf/",
    "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-release-375Bxml_FTPO",
    "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-release-375Bxml_sft",
    "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-release-375Bxml_RLVR",
    "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-release-375Bxml_RLVR_postSFT",
    "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-release-1Bxml_FTPO_fixed",

    # "/jais3/checkpoints/jais3_experiments/sft/A0/A0_k2v3_100B_base_Jais3SFTv7",  #! TEST
    
    # # HuggingFace Checkpoints
    # "CohereLabs/tiny-aya-global",
    # "tiiuae/Falcon-H1-3B-Instruct",
    # "meta-llama/Llama-3.2-3B-Instruct",
    # "inceptionai/Jais-2-8B-Chat",
    # "meta-llama/Llama-3.1-8B-Instruct",
    # "inceptionai/Jais-2-70B-Chat",

    # # #Arabic Centric Models
    # "ibm-research/granite-4.0-h-3b-ar",
    # "humain-ai/ALLaM-7B-Instruct-preview",
    # "QCRI/Fanar-1-9B-Instruct",
    # # "QCRI/Fanar-2-27B-Instruct",
    # "CohereLabs/aya-23-8B",
    # "tiiuae/Falcon-H1-7B-Instruct",
    # "silma-ai/SILMA-9B-Instruct-v1.0",
    # "Navid-AI/Yehia-7B-preview",
    # "FreedomIntelligence/AceGPT-v2-8B-Chat",
    # "CohereLabs/aya-expanse-8b",
    # # # "hammh0a/Hala-9B", #! TAKES TOO LONG!!!

    # "Qwen/Qwen3.5-4B",
    # "Qwen/Qwen3.5-9B",
    # "google/gemma-4-E2B-it",
    # "google/gemma-4-E4B-it",
    # # "Qwen/Qwen3.6-27B", #! Might be replaced by Qwen3.8-27B in the future
    # "Qwen/Qwen3.8-27B",
    # "google/gemma-4-12B-it",
    # "google/gemma-4-31B-it", #!Commented out on 7th of August because it freezes on some tasks
    "tiiuae/Falcon-H1-34B-Instruct",

    # "Applied-Innovation-Center/Karnak-40B-v1.0",
    # "Applied-Innovation-Center/Karnak-6B-v1.0",

    # "/jais3/checkpoints/bbq-4b-mid3_v3-final",
    # "/jais3/checkpoints/bbq-4b-mid4-v2-final",

    # "/jais3/checkpoints/bbq-8b-mid1-final",
    # "/jais3/checkpoints/bbq-8b-mid2",
    # "/jais3/checkpoints/bbq-8b-mid2_v3-cp25000",
    # "/jais3/checkpoints/bbq-8b-mid3-final",
    # "/jais3/checkpoints/bbq-8b-mid3_v3-cp5500",
    # "/jais3/checkpoints/bbq-8b-mid4-v2-final",
    
    # "/jais3/checkpoints/bbq-32b-mid3_v3",
    "/jais3/checkpoints/bbq-8b-mid5-phase2",
    "/jais3/checkpoints/bbq-8b-mid5-phase1",
    "/jais3/checkpoints/bbq-4b-mid5-phase1",
    # "/jais3/checkpoints/bbq-36b-mova-mid4-v2", #replaced by K2-Horizon-MoVA-36B-A4B-mid_4
    "/jais3/checkpoints/K2-Horizon-MoVA-36B-A4B-mid_4",

    "Qwen/Qwen3.8-Flash-Next",
    "Qwen/Qwen3.5-397B-A17B",
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
    "MiniMaxAI/MiniMax-M3",

    # # xml 1B
    # # M1 (no-drop)
    "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-1Bfinalcv5c189_sft_reasoningv9.5.2k2horCT_FTPO_es035_RLVRIfI3rabTashkeelMathCode16k_M1nodrop_4n_checkpoint_199_to_hf",
    # # Main
    "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-1Bfinalcv5c189_sft_reasoningv9.5.2k2horCT_FTPO_es035_RLVRIfI3rabTashkeelMathCode16k_8n_checkpoint_199_to_hf",
    # # UniB
    "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-1Bfinalcv5c189_sft_reasoningv9.5.2k2horCT_FTPO_es035_RLVRIfI3rabTashkeelMathCode16k_uniformB_signal_4n_checkpoint_199_to_hf",
    # # M3 (no-drop + token-level loss + KL)
    "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-1Bfinalcv5c189_sft_reasoningv9.5.2k2horCT_FTPO_es035_RLVRIfI3rabTashkeelMathCode16k_M3nodrop_toklvl_kl_4n_checkpoint_199_to_hf",
    # # M2 (no-drop + token-level loss)
    "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-1Bfinalcv5c189_sft_reasoningv9.5.2k2horCT_FTPO_es035_RLVRIfI3rabTashkeelMathCode16k_M2nodrop_toklvl_4n_checkpoint_199_to_hf",
    # # Main s43
    "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-1Bfinalcv5c189_sft_reasoningv9.5.2k2horCT_FTPO_es035_RLVRIfI3rabTashkeelMathCode16k_seed43_8n_checkpoint_199_to_hf",
    # # Main s44
    "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-1Bfinalcv5c189_sft_reasoningv9.5.2k2horCT_FTPO_es035_RLVRIfI3rabTashkeelMathCode16k_seed44_8n_checkpoint_199_to_hf",
    # # UniA
    "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-1Bfinalcv5c189_sft_reasoningv9.5.2k2horCT_FTPO_es035_RLVRIfI3rabTashkeelMathCode16k_uniformA_4n_checkpoint_199_to_hf",

    # 1B further experiments
    "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-1Bfinalcv5c189_sft_reasoningv9.5.2k2horCT_FTPO_es035_RLVRIfI3rabTashkeelMathCode16k_8n_checkpoint_199_to_hf_FTPO_es035",
    "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-1Bfinalcv5c189_sft_reasoningv9.5.2k2horCT_FTPO_es035_RLVRIfI3rabTashkeelMathCode16k_8n_checkpoint_199_to_hf_FTPO_es065",
    "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-1Bfinalcv5c189_sft_reasoningv9.5.2k2horCT_FTPO_es035_RLVRIfI3rabTashkeelMathCode16k_8n_checkpoint_199_to_hf_FTPO_es080",
    "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-1Bfinalcv5c189_sft_reasoningv9.5.2k2horCT_FTPO_es035_RLVRIfI3rabTashkeelMathCode16k_8n_checkpoint_199_to_hf_FTPO_ep050",

    "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-1Bfinalcv5c189_sft_reasoningv9.5.2k2horCT_FTPO_es035_RLVRIfI3rabTashkeelMathCode16k_M1nodrop_4n_checkpoint_199_to_hf_FTPO_es035",
    "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-1Bfinalcv5c189_sft_reasoningv9.5.2k2horCT_FTPO_es035_RLVRIfI3rabTashkeelMathCode16k_M1nodrop_4n_checkpoint_199_to_hf_FTPO_es065",
    "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-1Bfinalcv5c189_sft_reasoningv9.5.2k2horCT_FTPO_es035_RLVRIfI3rabTashkeelMathCode16k_M1nodrop_4n_checkpoint_199_to_hf_FTPO_es080",
    "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-1Bfinalcv5c189_sft_reasoningv9.5.2k2horCT_FTPO_es035_RLVRIfI3rabTashkeelMathCode16k_M1nodrop_4n_checkpoint_199_to_hf_FTPO_ep050",

    "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-32Bxml-rlvr199loopfix_sft_reasoningv9.5.2k2horCT_2ep_checkpoint_2800_to_hf",
    "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-release-375Bxml_RLVR_IFRLVR_checkpoint_79_to_hf",
    "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-release-375Bxml_FTPO_RLVRIfI3rabTashkeelMathCode16k_64n_if375_checkpoint_89_to_hf",
    "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-release-375Bxml_FTPO_RLVRIfI3rabTashkeelMathCode16k_64n_if375_checkpoint_139_to_hf",
}

DEFAULT_THINK_CHECKPOINTS = {
    "IFM/Jais-3-32B-Uno",
    "IFM/Jais-3-7B-Uno",
    "IFM/Jais-3-3.7B-Uno",
    "IFM/Jais-3-0.9B-Uno",
    "IFM/Jais-3-7B-FP8",
    "IFM/Jais-3-3.7B-FP8",
    "IFM/Jais-3-32B-FP8",
    "IFM/Jais-3-MoVA-36B-A4B-FP8",
    "IFM/Jais-3-375B-A23B-FP8",
    "google/gemma-4-E2B-it-custom",
    "google/gemma-4-E4B-it-custom",
    "google/gemma-4-12B-it-custom",
    "google/gemma-4-31B-it-custom",

    #SFT CHECKPOINTS
    "/jais3/checkpoints/jais3_experiments/sft", #! DEPRECATED FOR NOW
    "/jais3/checkpoints/jais3_experiments/sft_eval",
    # "/jais3/checkpoints/jais3_experiments/sft/A0/A0_k2v3_100B_base_Jais3SFTv7",  #! TEST
    # "/jais3/checkpoints/bbq-8b-mid1-final", #! TEST
    # "/jais3/checkpoints/bbq-8b-mid2", #! TEST
    # "/jais3/checkpoints/bbq-8b-mid3-final", #! TEST
    # "/jais3/checkpoints/jais3_experiments/sft_eval/A7_A7x5cs17191lv5RLIFAr_checkpoint_160",
    # "/jais3/checkpoints/jais3_experiments/sft_eval/A7_A7x5cs17191lv6RewrittenAraTxt_RLIFEnAr_1206_rewardpenalty_lang_c0c0383_checkpoint_160",


    "/jais3/checkpoints/jais3_experiments/release_checkpoints/1b/cpt/hf_checkpoints/artifacts/checkpoint_11141_to_hf/",
    "/jais3/checkpoints/jais3_experiments/release_checkpoints/1b/cpt_sft/hf_checkpoints/artifacts/checkpoint_11141_to_hf/",
    "/jais3/checkpoints/jais3_experiments/release_checkpoints/1b_xml/sft/hf_checkpoints/artifacts/checkpoint_10711_to_hf/",
    "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-release-375Bxml_FTPO",
    "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-release-375Bxml_sft",
    "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-release-375Bxml_RLVR",
    "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-release-375Bxml_RLVR_postSFT",
    "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-release-1Bxml_FTPO_fixed",

    # 1B models
    "Qwen/Qwen3.5-2B", #has thinking mode.
    "IFM/K2-Horizon-0.9B",
    "/jais3/checkpoints/K2-Horizon-0.9B-mid_2_47684",
    # 4B models
    "Qwen/Qwen3.5-4B",
    "google/gemma-4-E2B-it",
    # "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
    # "mistralai/Ministral-3-3B-Reasoning-2512",
    "IFM/K2-Horizon-3.7B",

    # 7B models
    "google/gemma-4-E4B-it",
    "google/gemma-4-12B-it",
    "Qwen/Qwen3.5-9B",
    "IFM/K2-Horizon-7B",
    "IFM/K2-Horizon-375B-A23B",

    "IFM/K2-Horizon-MoVA-36B-A4B",

    # 32B models
    "google/gemma-4-31B-it",
    "meta-models/Muse-Glimmer-30B",
    "Qwen/Qwen3.8-27B",
    "QCRI/Fanar-2-27B-Instruct",
    "IFM/K2-Horizon-32B",

    # # xml 1B
    # # M1 (no-drop)
    "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-1Bfinalcv5c189_sft_reasoningv9.5.2k2horCT_FTPO_es035_RLVRIfI3rabTashkeelMathCode16k_M1nodrop_4n_checkpoint_199_to_hf",
    # # Main
    "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-1Bfinalcv5c189_sft_reasoningv9.5.2k2horCT_FTPO_es035_RLVRIfI3rabTashkeelMathCode16k_8n_checkpoint_199_to_hf",
    # # UniB
    "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-1Bfinalcv5c189_sft_reasoningv9.5.2k2horCT_FTPO_es035_RLVRIfI3rabTashkeelMathCode16k_uniformB_signal_4n_checkpoint_199_to_hf",
    # # M3 (no-drop + token-level loss + KL)
    "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-1Bfinalcv5c189_sft_reasoningv9.5.2k2horCT_FTPO_es035_RLVRIfI3rabTashkeelMathCode16k_M3nodrop_toklvl_kl_4n_checkpoint_199_to_hf",
    # # M2 (no-drop + token-level loss)
    "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-1Bfinalcv5c189_sft_reasoningv9.5.2k2horCT_FTPO_es035_RLVRIfI3rabTashkeelMathCode16k_M2nodrop_toklvl_4n_checkpoint_199_to_hf",
    # # Main s43
    "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-1Bfinalcv5c189_sft_reasoningv9.5.2k2horCT_FTPO_es035_RLVRIfI3rabTashkeelMathCode16k_seed43_8n_checkpoint_199_to_hf",
    # # Main s44
    "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-1Bfinalcv5c189_sft_reasoningv9.5.2k2horCT_FTPO_es035_RLVRIfI3rabTashkeelMathCode16k_seed44_8n_checkpoint_199_to_hf",
    # # UniA
    "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-1Bfinalcv5c189_sft_reasoningv9.5.2k2horCT_FTPO_es035_RLVRIfI3rabTashkeelMathCode16k_uniformA_4n_checkpoint_199_to_hf",


    # #HuggingFace Checkpoints 
    # "Qwen/Qwen3.5-4B",
    # "Qwen/Qwen3.5-9B",
    # "Qwen/Qwen3.6-27B",
    # "Qwen/Qwen3.8-27B", #! Should replace Qwen3.6-27B in the future
    # "google/gemma-4-E2B-it",
    # "google/gemma-4-E4B-it",
    # "google/gemma-4-12B-it",
    # "google/gemma-4-31B-it",  #!Commented out on 7th of August because it freezes on some tasks
    # "meta-models/Muse-Glimmer-30B", #doesn't support none-thinking mode (the supported thinking efforts are low, medium, high, and xhigh).
    # "/jais3/checkpoints/bbq-32b-mid3_v3",
    # "/jais3/checkpoints/bbq-4b-mid3_v3-final",
    # "/jais3/checkpoints/bbq-4b-mid4-v2-final",
    # "/jais3/checkpoints/bbq-8b-mid1-final",
    # "/jais3/checkpoints/bbq-8b-mid2",
    # "/jais3/checkpoints/bbq-8b-mid2_v3-cp25000",
    # "/jais3/checkpoints/bbq-8b-mid3-final",
    # "/jais3/checkpoints/bbq-8b-mid3_v3-cp5500",
    # "/jais3/checkpoints/bbq-8b-mid4-v2-final",
    "/jais3/checkpoints/bbq-8b-mid5-phase2",
    "/jais3/checkpoints/bbq-8b-mid5-phase1",
    "/jais3/checkpoints/bbq-4b-mid5-phase1",
    # "/jais3/checkpoints/bbq-36b-mova-mid4-v2", #replaced by K2-Horizon-MoVA-36B-A4B-mid_4
    "/jais3/checkpoints/K2-Horizon-MoVA-36B-A4B-mid_4",
    "Qwen/Qwen3.8-Flash-Next",
    "Qwen/Qwen3.5-397B-A17B",
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
    "MiniMaxAI/MiniMax-M3",
    "mistralai/Mistral-Small-4-119B-2603"
    
    # 1B further experiments
    "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-1Bfinalcv5c189_sft_reasoningv9.5.2k2horCT_FTPO_es035_RLVRIfI3rabTashkeelMathCode16k_8n_checkpoint_199_to_hf_FTPO_es035",
    "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-1Bfinalcv5c189_sft_reasoningv9.5.2k2horCT_FTPO_es035_RLVRIfI3rabTashkeelMathCode16k_8n_checkpoint_199_to_hf_FTPO_es065",
    "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-1Bfinalcv5c189_sft_reasoningv9.5.2k2horCT_FTPO_es035_RLVRIfI3rabTashkeelMathCode16k_8n_checkpoint_199_to_hf_FTPO_es080",
    "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-1Bfinalcv5c189_sft_reasoningv9.5.2k2horCT_FTPO_es035_RLVRIfI3rabTashkeelMathCode16k_8n_checkpoint_199_to_hf_FTPO_ep050",

    "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-1Bfinalcv5c189_sft_reasoningv9.5.2k2horCT_FTPO_es035_RLVRIfI3rabTashkeelMathCode16k_M1nodrop_4n_checkpoint_199_to_hf_FTPO_es035",
    "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-1Bfinalcv5c189_sft_reasoningv9.5.2k2horCT_FTPO_es035_RLVRIfI3rabTashkeelMathCode16k_M1nodrop_4n_checkpoint_199_to_hf_FTPO_es065",
    "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-1Bfinalcv5c189_sft_reasoningv9.5.2k2horCT_FTPO_es035_RLVRIfI3rabTashkeelMathCode16k_M1nodrop_4n_checkpoint_199_to_hf_FTPO_es080",
    "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-1Bfinalcv5c189_sft_reasoningv9.5.2k2horCT_FTPO_es035_RLVRIfI3rabTashkeelMathCode16k_M1nodrop_4n_checkpoint_199_to_hf_FTPO_ep050",

    "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-32Bxml-rlvr199loopfix_sft_reasoningv9.5.2k2horCT_2ep_checkpoint_2800_to_hf",
    "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-release-375Bxml_RLVR_IFRLVR_checkpoint_79_to_hf",
    "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-release-375Bxml_FTPO_RLVRIfI3rabTashkeelMathCode16k_64n_if375_checkpoint_89_to_hf",
    "/jais3/checkpoints/jais3_experiments/sft_eval/A13cs_A13x1-release-375Bxml_FTPO_RLVRIfI3rabTashkeelMathCode16k_64n_if375_checkpoint_139_to_hf",
}

# BASE_CHECKPOINTS + INSTRUCT_CHECKPOINTS + THINK_CHECKPOINTS
CHECKPOINTS= (
    DEFAULT_BASE_CHECKPOINTS |
    DEFAULT_INSTRUCT_CHECKPOINTS |
    DEFAULT_THINK_CHECKPOINTS
)


# ═════════════════════════════════════════════════════════════════════════════
# Blacklisting
# ═════════════════════════════════════════════════════════════════════════════
BLACKLIST_CHECKPOINTS = {
    "/jais3/checkpoints/jais3_experiments/sft_eval/A0_*",
    "/jais3/checkpoints/jais3_experiments/sft_eval/A3_*",
    "/jais3/checkpoints/jais3_experiments/sft_eval/A3.1_*",
    "/jais3/checkpoints/jais3_experiments/sft_eval/A04_*",
    "/jais3/checkpoints/jais3_experiments/sft_eval/A5_*",
    "/jais3/checkpoints/jais3_experiments/sft_eval/A7_*",
    "/jais3/checkpoints/jais3_experiments/sft_eval/A8_*",
    "/jais3/checkpoints/jais3_experiments/sft_eval/A8x*",
    "/jais3/checkpoints/jais3_experiments/sft_eval/A9_*",
    "/jais3/checkpoints/jais3_experiments/sft_eval/A10_*",
    "/jais3/checkpoints/jais3_experiments/sft_eval/A11_*",
    "/jais3/checkpoints/jais3_experiments/sft_eval/S3_*",
    "/jais3/checkpoints/jais3_experiments/sft_eval/S4_*",
    "/jais3/checkpoints/jais3_experiments/sft_eval/S5_*",
    "/jais3/checkpoints/jais3_experiments/sft_eval/S6_*",
    "/jais3/checkpoints/jais3_experiments/sft_eval/S7_*",
    "/jais3/checkpoints/jais3_experiments/sft_eval/S8_*",
    "/jais3/checkpoints/jais3_experiments/sft_eval/S9_*",
    "*deltaMid*",
}
# filter base checkpoints as well
BLACKLIST_CHECKPOINTS = BLACKLIST_CHECKPOINTS | DEFAULT_BASE_CHECKPOINTS

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
REASONING_TAIL_NUM_CHARS_TO_KEEP = 2000 # tail chars to keep from reasoning when no response is provided
MAX_RESPONSE_LENGTH = 32768 # number of characters to keep from too long responses
