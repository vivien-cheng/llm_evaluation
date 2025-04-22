#!/usr/bin/env bash
# Removed 'set -e' to allow continuation after individual model failures
# Errors will still be reported for each failed step.

echo "--- Workflow B - Step 2c: Evaluating Logprob Filtering Strategy ---" # <-- Renamed Step

# --- Basic Setup ---
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
EXAMPLE_DIR=$(realpath "$SCRIPT_DIR/..")
SRC_DIR="$EXAMPLE_DIR/src"
RESPONSES_DIR="$EXAMPLE_DIR/outputs/model_responses" # Input from Step 1
ORIGINAL_DATA_FILE="$EXAMPLE_DIR/data/processed/longhealth_cleaned.jsonl" # Input from Step 0
OUTPUT_EVAL_DIR="$EXAMPLE_DIR/outputs/logprob_evaluation" # Output dir for this step
# OUTPUT_SCORED_RESPONSES_DIR="$EXAMPLE_DIR/outputs/scored_responses_detail" # Optional

# --- Configuration ---
# Choose which setting to run: "fixed_verifier" or "fixed_generator"
EVALUATION_SETTING="fixed_generator" # <<< EDIT THIS: "fixed_verifier" or "fixed_generator"

# --- Setting 1: Fixed Verifier, Varying Generators ---
# Define the SINGLE verifier model to use (REPLACE Llama if access issue persists)
FIXED_VERIFIER="tiiuae/falcon-7b-instruct" # Example: Use Falcon
# FIXED_VERIFIER="meta-llama/Llama-3.1-8B-Instruct" # <<< Your requested model (NEEDS ACCESS)

# --- Setting 2: Fixed Generator, Varying Verifiers ---
# Define the SINGLE generator model's response file to evaluate
# (Filename should match output from Step 1)
FIXED_GENERATOR_RESP_FILE="${RESPONSES_DIR}/Qwen_Qwen2.5-7B-Instruct_responses.jsonl" # Example
# Define the LIST of verifier models to test (REPLACE Llama if access issue persists)
VARYING_VERIFIERS=(
    "tiiuae/falcon-7b-instruct"
    "Qwen/Qwen1.5-1.8B-Chat"
    "mistralai/Mistral-7B-Instruct-v0.3" # Gated
    # "meta-llama/Llama-3.1-8B-Instruct" # <<< Your requested model (NEEDS ACCESS)
    # Add Llama 3.2 variants here if accessible
)

# --- Common Parameters ---
MODEL_MAX_LEN=4096
CACHE_FLAG="--use_cache"
MAX_CONTEXT_CHARS=10000
PASS_K=3 # Value for Pass@K calculation
PROJECT_ROOT=$(realpath "$EXAMPLE_DIR/../..")

# --- Prerequisites Reminder ---
echo "######################################################################"
echo "# Ensure prerequisites are met for selected models & setting.      #"
echo "# 1. Verifier model access requested/approved if gated.            #"
echo "# 2. Logged in via 'huggingface-cli login' with READ token.        #"
echo "# 3. 'bitsandbytes' installed if using quantization.               #"
echo "# 4. Response files exist in '$RESPONSES_DIR' (run Step 1).        #"
echo "# 5. Original data file exists '$ORIGINAL_DATA_FILE' (run Step 0). #"
echo "######################################################################"
echo "Starting runs in 5 seconds..."
sleep 5

# --- Ensure directories exist ---
mkdir -p "$OUTPUT_EVAL_DIR"
# mkdir -p "$OUTPUT_SCORED_RESPONSES_DIR" # Uncomment if using optional dir

if [ ! -f "$ORIGINAL_DATA_FILE" ]; then
    echo "Error: Original data file not found: $ORIGINAL_DATA_FILE"
    exit 1
fi

# =============================================================
# --- Execute Based on Setting ---
# =============================================================

if [ "$EVALUATION_SETTING" == "fixed_verifier" ]; then
    # --- Setting 1: Fixed Verifier, Varying Generators ---
    echo ""
    echo "*********** RUNNING SETTING 1: FIXED VERIFIER ***********"
    echo "Verifier Model: $FIXED_VERIFIER"
    echo "Responses Dir: $RESPONSES_DIR (Contains responses from multiple generators)"
    echo "*********************************************************"

    if [ ! -d "$RESPONSES_DIR" ] || [ -z "$(ls -A $RESPONSES_DIR/*.jsonl 2>/dev/null)" ]; then
        echo "Error: Responses directory '$RESPONSES_DIR' not found or empty for Setting 1. Run Step 1 with multiple generators."
        exit 1
    fi

    model_id="$FIXED_VERIFIER"
    safe_model_name=$(echo "$model_id" | tr '/' '_')

    # --- Run WITHOUT Quantization ---
    echo "--- Attempting run WITHOUT Quantization ---"
    output_eval_file_nq="${OUTPUT_EVAL_DIR}/eval_FIXED_VERIFIER_${safe_model_name}_NoQuant.json"
    detail_output_arg_nq=""
    # if [ -n "$OUTPUT_SCORED_RESPONSES_DIR" ]; then ... fi # Optional detail dir logic
    quant_flag_run=""
    echo "Output file: $output_eval_file_nq"

    CMD="PYTHONPATH=\"$PROJECT_ROOT:$EXAMPLE_DIR:$PYTHONPATH\" python \"$SRC_DIR/evaluate_logprob_filter.py\" \
        --responses_dir \"$RESPONSES_DIR\" \
        --original_data_file \"$ORIGINAL_DATA_FILE\" \
        --output_eval_file \"$output_eval_file_nq\" \
        $detail_output_arg_nq \
        --verifier_model_name \"$model_id\" \
        --max_context_len \"$MAX_CONTEXT_CHARS\" \
        --model_max_length \"$MODEL_MAX_LEN\" \
        --pass_k $PASS_K \
        $CACHE_FLAG \
        $quant_flag_run"
    echo "Executing: $CMD"
    eval "$CMD"
    if [ $? -eq 0 ]; then echo "--- Evaluation finished successfully. ---"; else echo "--- Evaluation FAILED for $model_id (Check Logs) ---"; fi
    echo ""

    # --- Run WITH Quantization (Optional) ---
    run_quant=true
    case "$model_id" in # Add models to skip quantization if needed
         Qwen/Qwen1.5-1.8B-Chat|TinyLlama/TinyLlama-1.1B-Chat-v1.0|microsoft/phi-2|Qwen/Qwen2.5-3B-Instruct) run_quant=false ;;
    esac
    if [ "$run_quant" = true ]; then
       echo "--- Attempting run WITH Quantization ---"
       output_eval_file_q="${OUTPUT_EVAL_DIR}/eval_FIXED_VERIFIER_${safe_model_name}_Quant.json"
       quant_flag_run="--use_quantization"
       echo "Output file: $output_eval_file_q"
       CMD_Q="PYTHONPATH=\"$PROJECT_ROOT:$EXAMPLE_DIR:$PYTHONPATH\" python \"$SRC_DIR/evaluate_logprob_filter.py\" \
           --responses_dir \"$RESPONSES_DIR\" \
           --original_data_file \"$ORIGINAL_DATA_FILE\" \
           --output_eval_file \"$output_eval_file_q\" \
           --verifier_model_name \"$model_id\" \
           --max_context_len \"$MAX_CONTEXT_CHARS\" \
           --model_max_length \"$MODEL_MAX_LEN\" \
           --pass_k $PASS_K \
           $CACHE_FLAG \
           $quant_flag_run"
       echo "Executing: $CMD_Q"
       eval "$CMD_Q"
       if [ $? -eq 0 ]; then echo "--- Evaluation finished successfully. ---"; else echo "--- Evaluation FAILED for $model_id (Check Logs) ---"; fi
    else
       echo "--- Skipping Quantization run for $model_id ---"
    fi
    echo "*********** FINISHED SETTING 1 ***********"

elif [ "$EVALUATION_SETTING" == "fixed_generator" ]; then
    # --- Setting 2: Fixed Generator, Varying Verifiers ---
    echo ""
    echo "*********** RUNNING SETTING 2: FIXED GENERATOR ***********"
    echo "Generator Response File: $FIXED_GENERATOR_RESP_FILE"
    echo "Verifier Models: ${VARYING_VERIFIERS[@]}"
    echo "**********************************************************"

    if [ ! -f "$FIXED_GENERATOR_RESP_FILE" ]; then
        echo "Error: Fixed generator response file not found: $FIXED_GENERATOR_RESP_FILE. Run Step 1 for the corresponding model."
        exit 1
    fi

    # Loop through verifiers
    for model_id in "${VARYING_VERIFIERS[@]}"; do
        echo ""
        echo "======================================================"
        echo "Evaluating using Verifier: $model_id"
        echo "======================================================"
        safe_model_name=$(echo "$model_id" | tr '/' '_')
        generator_basename=$(basename "$FIXED_GENERATOR_RESP_FILE" .jsonl)

        # --- Run WITHOUT Quantization ---
        echo "--- Attempting run WITHOUT Quantization ---"
        output_eval_file_nq="${OUTPUT_EVAL_DIR}/eval_FIXED_GEN_${generator_basename}_verifier_${safe_model_name}_NoQuant.json"
        quant_flag_run=""
        echo "Output file: $output_eval_file_nq"

        # --- FIX: Moved comment BEFORE backslash ---
        CMD="PYTHONPATH=\"$PROJECT_ROOT:$EXAMPLE_DIR:$PYTHONPATH\" python \"$SRC_DIR/evaluate_logprob_filter.py\" \
            --response_files \"$FIXED_GENERATOR_RESP_FILE\" \
            --original_data_file \"$ORIGINAL_DATA_FILE\" \
            --output_eval_file \"$output_eval_file_nq\" \
            --verifier_model_name \"$model_id\" \
            --max_context_len \"$MAX_CONTEXT_CHARS\" \
            --model_max_length \"$MODEL_MAX_LEN\" \
            --pass_k $PASS_K \
            $CACHE_FLAG \
            $quant_flag_run"
        # --- End Fix ---
        echo "Executing: $CMD"
        eval "$CMD"
        if [ $? -eq 0 ]; then echo "--- Evaluation finished successfully. ---"; else echo "--- Evaluation FAILED for $model_id (Check Logs) ---"; fi
        echo ""
        sleep 2

        # --- Run WITH Quantization (Optional) ---
        run_quant=true
        case "$model_id" in
             Qwen/Qwen1.5-1.8B-Chat|TinyLlama/TinyLlama-1.1B-Chat-v1.0|microsoft/phi-2|Qwen/Qwen2.5-3B-Instruct) run_quant=false ;;
        esac
        if [ "$run_quant" = true ]; then
           echo "--- Attempting run WITH Quantization ---"
           output_eval_file_q="${OUTPUT_EVAL_DIR}/eval_FIXED_GEN_${generator_basename}_verifier_${safe_model_name}_Quant.json"
           quant_flag_run="--use_quantization"
           echo "Output file: $output_eval_file_q"
           # --- FIX: Moved comment BEFORE backslash ---
           CMD_Q="PYTHONPATH=\"$PROJECT_ROOT:$EXAMPLE_DIR:$PYTHONPATH\" python \"$SRC_DIR/evaluate_logprob_filter.py\" \
               --response_files \"$FIXED_GENERATOR_RESP_FILE\" \
               --original_data_file \"$ORIGINAL_DATA_FILE\" \
               --output_eval_file \"$output_eval_file_q\" \
               --verifier_model_name \"$model_id\" \
               --max_context_len \"$MAX_CONTEXT_CHARS\" \
               --model_max_length \"$MODEL_MAX_LEN\" \
               --pass_k $PASS_K \
               $CACHE_FLAG \
               $quant_flag_run"
           # --- End Fix ---
           echo "Executing: $CMD_Q"
           eval "$CMD_Q"
           if [ $? -eq 0 ]; then echo "--- Evaluation finished successfully. ---"; else echo "--- Evaluation FAILED for $model_id (Check Logs) ---"; fi
        else
           echo "--- Skipping Quantization run for $model_id ---"
        fi
        echo "======================================================"
        echo "Finished Evaluation for Verifier: $model_id"
        echo "======================================================"
        echo "Attempting memory cleanup..."
        python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
        sleep 5
    done # End verifier loop
    echo "*********** FINISHED SETTING 2 ***********"
    echo "NOTE: Run 'scripts/3a_analyze_logprob_agreement.sh' script next to analyze results."

else
    echo "Error: Invalid EVALUATION_SETTING specified: '$EVALUATION_SETTING'. Choose 'fixed_verifier' or 'fixed_generator'."
    exit 1
fi

echo ""
echo "--- All Selected Evaluations Attempted ---"
echo "Check evaluation summaries in: $OUTPUT_EVAL_DIR"

