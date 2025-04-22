#!/usr/bin/env bash
# Script to rank predefined MCQ options using verifier log probabilities.
# Note: This was a preliminary step to help select verifiers for Step 2c/5.
# It does NOT score generated text.

echo "--- Preliminary Step 4: Ranking Predefined Options via Verifier Log Probs ---"

# --- Basic Setup ---
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
EXAMPLE_DIR=$(realpath "$SCRIPT_DIR/..")
SRC_DIR="$EXAMPLE_DIR/src"
INPUT_DATA_FILE="$EXAMPLE_DIR/data/processed/longhealth_cleaned.jsonl" # Needs output from Step 0
OUTPUT_DIR="$EXAMPLE_DIR/outputs/predefined_option_ranking" # Separate output dir
PROJECT_ROOT=$(realpath "$EXAMPLE_DIR/../..")

# --- Shared Parameters ---
MAX_CONTEXT_CHARS=10000
MODEL_MAX_LEN=4096
CACHE_FLAG="--use_cache"

# --- Models To Test (Example List - Ensure Access!) ---
LOCAL_MODELS_TO_TEST=(
    "tiiuae/falcon-7b-instruct"
    "Qwen/Qwen1.5-1.8B-Chat"
    "mistralai/Mistral-7B-Instruct-v0.3" # Gated
    # Add other models tested in the preliminary phase
)
LOCAL_MODELS_SKIP_QUANT=(
    "Qwen/Qwen1.5-1.8B-Chat"
    # Add others if needed
)

# --- Prerequisites ---
echo "Ensure Step 0 is complete and required models are accessible."
sleep 3

# --- Ensure output directory exists ---
mkdir -p "$OUTPUT_DIR"

if [ ! -f "$INPUT_DATA_FILE" ]; then
    echo "Error: Input data file not found: $INPUT_DATA_FILE. Run Step 0 first."
    exit 1
fi

# --- Loop through LOCAL Models ---
echo "*********** STARTING PRELIMINARY VERIFIER TESTS (RANKING OPTIONS) ***********"
for model_id in "${LOCAL_MODELS_TO_TEST[@]}"; do
    echo ""
    echo "======================================================"
    echo "Starting Preliminary Test for Model: $model_id"
    echo "======================================================"
    safe_model_name=$(echo "$model_id" | tr '/' '_')

    # --- Run WITHOUT Quantization ---
    echo "--- Attempting run WITHOUT Quantization ---"
    output_filename_noquant="${OUTPUT_DIR}/ranked_options_${safe_model_name}_NoQuant.jsonl"
    quant_flag_run=""
    echo "Output file: $output_filename_noquant"

    CMD="PYTHONPATH=\"$PROJECT_ROOT:$EXAMPLE_DIR:$PYTHONPATH\" python \"$SRC_DIR/rank_options_logprobs.py\" \
        --input_data_file \"$INPUT_DATA_FILE\" \
        --output_file \"$output_filename_noquant\" \
        --verifier_model_name \"$model_id\" \
        --max_context_len \"$MAX_CONTEXT_CHARS\" \
        --model_max_length \"$MODEL_MAX_LEN\" \
        $CACHE_FLAG \
        $quant_flag_run"
    echo "Executing: $CMD"
    eval "$CMD"
    if [ $? -eq 0 ]; then echo "--- Run finished successfully. ---"; else echo "--- Run FAILED for $model_id (Check Logs) ---"; fi
    echo ""
    sleep 2

    # --- Run WITH Quantization ---
    skip_quant=false
    for skip_model in "${LOCAL_MODELS_SKIP_QUANT[@]}"; do if [[ "$model_id" == "$skip_model" ]]; then skip_quant=true; break; fi; done

    if [ "$skip_quant" = false ]; then
        echo "--- Attempting run WITH Quantization ---"
        output_filename_quant="${OUTPUT_DIR}/ranked_options_${safe_model_name}_Quant.jsonl"
        quant_flag_run="--use_quantization"
        echo "Output file: $output_filename_quant"

        CMD_Q="PYTHONPATH=\"$PROJECT_ROOT:$EXAMPLE_DIR:$PYTHONPATH\" python \"$SRC_DIR/rank_options_logprobs.py\" \
            --input_data_file \"$INPUT_DATA_FILE\" \
            --output_file \"$output_filename_quant\" \
            --verifier_model_name \"$model_id\" \
            --max_context_len \"$MAX_CONTEXT_CHARS\" \
            --model_max_length \"$MODEL_MAX_LEN\" \
            $CACHE_FLAG \
            $quant_flag_run"
        echo "Executing: $CMD_Q"
        eval "$CMD_Q"
        if [ $? -eq 0 ]; then echo "--- Run finished successfully. ---"; else echo "--- Run FAILED for $model_id (Check Logs) ---"; fi
    else
        echo "--- Skipping Quantization run for $model_id ---"
    fi

    echo "======================================================"
    echo "Finished Preliminary Test for Model: $model_id"
    echo "======================================================"
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    sleep 5
done
echo "*********** FINISHED PRELIMINARY VERIFIER TESTS ***********"
echo ""
