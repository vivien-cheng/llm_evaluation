#!/usr/bin/env bash
# Removed 'set -e' to allow continuation after individual model failures
# Errors will still be reported for each failed step.

echo "--- Step 5: Evaluating Logprob Filtering Strategy ---"

# --- Basic Setup ---
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
EXAMPLE_DIR=$(realpath "$SCRIPT_DIR/..")
SRC_DIR="$EXAMPLE_DIR/src"
RESPONSES_DIR="$EXAMPLE_DIR/outputs/model_responses"
ORIGINAL_DATA_FILE="$EXAMPLE_DIR/data/processed/longhealth_cleaned.jsonl"
OUTPUT_EVAL_DIR="$EXAMPLE_DIR/outputs/logprob_evaluation"
# OUTPUT_SCORED_RESPONSES_DIR="$EXAMPLE_DIR/outputs/scored_responses_detail" # Optional

# --- Configuration ---
# !!! EDIT THIS LIST !!!
VERIFIER_MODELS_TO_TEST=(
    "tiiuae/falcon-7b-instruct"
    "Qwen/Qwen1.5-1.8B-Chat"
    # Add other models here
)

MODEL_MAX_LEN=4096
CACHE_FLAG="--use_cache"
MAX_CONTEXT_CHARS=10000
PROJECT_ROOT=$(realpath "$EXAMPLE_DIR/../..")

# --- Prerequisites Reminder ---
echo "######################################################################"
echo "# Ensure prerequisites are met:                                    #"
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

if [ ! -d "$RESPONSES_DIR" ] || [ -z "$(ls -A $RESPONSES_DIR/*.jsonl 2>/dev/null)" ]; then
    echo "Error: Responses directory '$RESPONSES_DIR' not found or empty. Run Step 1 first."
    exit 1
fi
if [ ! -f "$ORIGINAL_DATA_FILE" ]; then
    echo "Error: Original data file not found: $ORIGINAL_DATA_FILE. Run Step 0 first."
    exit 1
fi

# --- Loop through Verifier Models ---
for model_id in "${VERIFIER_MODELS_TO_TEST[@]}"; do
    echo ""
    echo "======================================================"
    echo "Evaluating Logprob Filter using Verifier: $model_id"
    echo "======================================================"
    safe_model_name=$(echo "$model_id" | tr '/' '_')

    # --- Run WITHOUT Quantization ---
    echo "--- Attempting run WITHOUT Quantization ---"
    output_eval_file_nq="${OUTPUT_EVAL_DIR}/eval_logprob_${safe_model_name}_NoQuant.json"
    detail_output_arg_nq=""
    # if [ -n "$OUTPUT_SCORED_RESPONSES_DIR" ]; then # Optional detail dir logic
    #     mkdir -p "$OUTPUT_SCORED_RESPONSES_DIR/${safe_model_name}_NoQuant"
    #     detail_output_arg_nq="--output_scored_responses_dir \"$OUTPUT_SCORED_RESPONSES_DIR/${safe_model_name}_NoQuant\""
    # fi
    quant_flag_run=""
    echo "Output file: $output_eval_file_nq"

    # Construct command string - ensure quoting is correct for eval
    CMD="PYTHONPATH=\"$PROJECT_ROOT:$EXAMPLE_DIR:$PYTHONPATH\" python \"$SRC_DIR/evaluate_logprob_filter.py\" \
        --responses_dir \"$RESPONSES_DIR\" \
        --original_data_file \"$ORIGINAL_DATA_FILE\" \
        --output_eval_file \"$output_eval_file_nq\" \
        $detail_output_arg_nq \
        --verifier_model_name \"$model_id\" \
        --max_context_len \"$MAX_CONTEXT_CHARS\" \
        --model_max_length \"$MODEL_MAX_LEN\" \
        $CACHE_FLAG \
        $quant_flag_run"

    # Execute using eval
    echo "Executing: $CMD" # Debug echo
    eval "$CMD"
    if [ $? -eq 0 ]; then
        echo "--- Evaluation WITHOUT Quantization finished successfully. ---"
    else
        echo "--- Evaluation WITHOUT Quantization FAILED for $model_id (Check Logs Above) ---"
    fi
    echo ""
    sleep 2

    # --- Run WITH Quantization (Optional) ---
    run_quant=true
    case "$model_id" in
        Qwen/Qwen1.5-1.8B-Chat|TinyLlama/TinyLlama-1.1B-Chat-v1.0|microsoft/phi-2|Qwen/Qwen2.5-1.5B-Instruct|Qwen/Qwen2.5-3B-Instruct)
            run_quant=false
            echo "--- Skipping Quantization run for $model_id ---"
            ;;
    esac

    if [ "$run_quant" = true ]; then
       echo "--- Attempting run WITH Quantization ---"
       output_eval_file_q="${OUTPUT_EVAL_DIR}/eval_logprob_${safe_model_name}_Quant.json"
       detail_output_arg_q=""
       # if [ -n "$OUTPUT_SCORED_RESPONSES_DIR" ]; then # Optional detail dir logic
       #     mkdir -p "$OUTPUT_SCORED_RESPONSES_DIR/${safe_model_name}_Quant"
       #     detail_output_arg_q="--output_scored_responses_dir \"$OUTPUT_SCORED_RESPONSES_DIR/${safe_model_name}_Quant\""
       # fi
       quant_flag_run="--use_quantization"
       echo "Output file: $output_eval_file_q"

       # Construct command string
       CMD_Q="PYTHONPATH=\"$PROJECT_ROOT:$EXAMPLE_DIR:$PYTHONPATH\" python \"$SRC_DIR/evaluate_logprob_filter.py\" \
           --responses_dir \"$RESPONSES_DIR\" \
           --original_data_file \"$ORIGINAL_DATA_FILE\" \
           --output_eval_file \"$output_eval_file_q\" \
           $detail_output_arg_q \
           --verifier_model_name \"$model_id\" \
           --max_context_len \"$MAX_CONTEXT_CHARS\" \
           --model_max_length \"$MODEL_MAX_LEN\" \
           $CACHE_FLAG \
           $quant_flag_run"

       # Execute using eval
       echo "Executing: $CMD_Q" # Debug echo
       eval "$CMD_Q"
       if [ $? -eq 0 ]; then
           echo "--- Evaluation WITH Quantization finished successfully. ---"
       else
           echo "--- Evaluation WITH Quantization FAILED for $model_id (Check Logs Above) ---"
       fi
    fi

    echo "======================================================"
    echo "Finished Evaluation for Verifier: $model_id"
    echo "======================================================"
    echo "Attempting memory cleanup..."
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    sleep 5
done

echo "--- All Verifier Evaluations Attempted ---"
echo "Check evaluation summaries in: $OUTPUT_EVAL_DIR"

