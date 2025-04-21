#!/usr/bin/env bash
# Removed 'set -e' to allow continuation after individual model failures
# Errors will still be reported for each failed step.

echo "--- Step 4: Ranking Options via Verifier Log Probs (Batch Run) ---"

# --- Basic Setup ---
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
EXAMPLE_DIR=$(realpath "$SCRIPT_DIR/..")
SRC_DIR="$EXAMPLE_DIR/src"
# Ensure using the correctly processed real data subset
INPUT_DATA_FILE="$EXAMPLE_DIR/data/processed/longhealth_cleaned.jsonl"
OUTPUT_DIR="$EXAMPLE_DIR/outputs"
PROJECT_ROOT=$(realpath "$EXAMPLE_DIR/../..")

# --- Shared Parameters ---
MAX_CONTEXT_CHARS=10000
# General max length for most models; some might have shorter limits (e.g., 2048)
# The python script handles truncation based on this, but warnings might occur.
MODEL_MAX_LEN=4096
CACHE_FLAG="--use_cache" # Use KV Caching for local models

# --- Local Models Configuration ---
# Models listed in the user's provided script (uncommented) + corrected Qwen names
LOCAL_MODELS_TO_TEST=(
    "Qwen/Qwen2.5-3B-Instruct"            # Corrected Name, Gated? Check HF
    "Qwen/Qwen2.5-7B-Instruct"            # Corrected Name, Gated? Check HF
    "Qwen/Qwen1.5-1.8B-Chat"              # Tested OK
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Tested OK
    "microsoft/phi-2"                     # Tested OK
    "tiiuae/falcon-7b-instruct"           # Tested OK
    "mistralai/Mistral-7B-Instruct-v0.3"  # Gated, Tested OK
    "google/gemma-7b-it"                  # Gated, Tested OK
    "meta-llama/Llama-2-7b-chat-hf"       # Gated, Tested OK
    "upstage/SOLAR-10.7B-Instruct-v1.0"   # Tested OK
    "meta-llama/Llama-3.1-8B-Instruct"    # Gated, FAILED PREVIOUSLY (Access/Token Issue)
    "meta-llama/Llama-2-13b-chat-hf"      # Gated, FAILED PREVIOUSLY (Access/Token Issue)
)

# Models where quantization is generally skipped (too small or less beneficial)
LOCAL_MODELS_SKIP_QUANT=(
    "Qwen/Qwen2.5-1.5B-Instruct" 
    "Qwen/Qwen2.5-3B-Instruct" 
    "Qwen/Qwen1.5-1.8B-Chat"
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    "microsoft/phi-2" 
)

# --- API Models Configuration ---
API_MODELS_TO_TEST=(
    # Check exact model ID on Fireworks website / API docs
    "accounts/fireworks/models/llama-v3p1-405b-instruct"
)
FW_API_KEY_ENV_VAR="FIREWORKS_API_KEY" # Ensure this matches your .env file key name

# --- Prerequisites Reminder ---
echo "######################################################################"
echo "# Running tests for multiple models (Local and API).               #"
echo "# IMPORTANT: Ensure prerequisites are met:                         #"
echo "# 1. Gated model access requested/approved (Llama, Mistral, Gemma).#"
echo "#    -> Especially check specific Llama model access!              #"
echo "# 2. Logged in via 'huggingface-cli login' with READ token.        #"
echo "# 3. 'bitsandbytes' installed for local quantization (if using).   #"
echo "# 4. '$FW_API_KEY_ENV_VAR' set in .env file for Fireworks API runs.    #"
echo "# 5. Input data '$INPUT_DATA_FILE' exists and is correct (run Step 0).#"
echo "######################################################################"
echo "Starting runs in 3 seconds..."
sleep 3

# --- Ensure output directory exists ---
mkdir -p "$OUTPUT_DIR"

# --- Check if input data exists ---
if [ ! -f "$INPUT_DATA_FILE" ]; then
    echo "Error: Input data file not found: $INPUT_DATA_FILE. Run Step 0 first (without --use-dummy)."
    exit 1
fi

# --- Loop through LOCAL Models ---
echo ""
echo "*********** STARTING LOCAL MODEL RUNS ***********"
for model_id in "${LOCAL_MODELS_TO_TEST[@]}"; do
    echo ""
    echo "======================================================"
    echo "Starting LOCAL Test for Model: $model_id"
    echo "======================================================"
    safe_model_name=$(echo "$model_id" | tr '/' '_') # Sanitize name

    # --- Run WITHOUT Quantization ---
    echo "--- Attempting run WITHOUT Quantization ---"
    output_filename_noquant="${OUTPUT_DIR}/ranked_options_logprobs_${safe_model_name}_Local_NoQuant.jsonl"
    quant_flag_run=""
    echo "Output file: $output_filename_noquant"

    # Run command in subshell and use || true to prevent exit on error
    ( PYTHONPATH="$PROJECT_ROOT:$EXAMPLE_DIR:$PYTHONPATH" python "$SRC_DIR/rank_options_logprobs.py" \
        --input_data_file "$INPUT_DATA_FILE" \
        --output_file "$output_filename_noquant" \
        --verifier_model_name "$model_id" \
        --max_context_len "$MAX_CONTEXT_CHARS" \
        --model_max_length "$MODEL_MAX_LEN" \
        $CACHE_FLAG \
        $quant_flag_run \
    && echo "--- Run WITHOUT Quantization finished successfully. ---" ) || echo "--- Run WITHOUT Quantization FAILED for $model_id (Check Logs Above) ---"
    echo ""
    sleep 2

    # --- Run WITH Quantization (if not skipped) ---
    skip_quant=false
    for skip_model in "${LOCAL_MODELS_SKIP_QUANT[@]}"; do
        if [[ "$model_id" == "$skip_model" ]]; then
            skip_quant=true
            break
        fi
    done

    if [ "$skip_quant" = false ]; then
        echo "--- Attempting run WITH Quantization ---"
        output_filename_quant="${OUTPUT_DIR}/ranked_options_logprobs_${safe_model_name}_Local_Quant.jsonl"
        quant_flag_run="--use_quantization"
        echo "Output file: $output_filename_quant"

        # Run command in subshell and use || true
        ( PYTHONPATH="$PROJECT_ROOT:$EXAMPLE_DIR:$PYTHONPATH" python "$SRC_DIR/rank_options_logprobs.py" \
            --input_data_file "$INPUT_DATA_FILE" \
            --output_file "$output_filename_quant" \
            --verifier_model_name "$model_id" \
            --max_context_len "$MAX_CONTEXT_CHARS" \
            --model_max_length "$MODEL_MAX_LEN" \
            $CACHE_FLAG \
            $quant_flag_run \
        && echo "--- Run WITH Quantization finished successfully. ---" ) || echo "--- Run WITH Quantization FAILED for $model_id (Check Logs Above) ---"
    else
        echo "--- Skipping Quantization run for $model_id ---"
    fi

    echo "======================================================"
    echo "Finished LOCAL Tests for Model: $model_id"
    echo "======================================================"
    echo "Attempting memory cleanup..."
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true # Try cleanup, ignore errors
    sleep 5
done
echo "*********** FINISHED LOCAL MODEL RUNS ***********"
echo ""


# --- Loop through API Models ---
echo ""
echo "*********** STARTING API MODEL RUNS ***********"
# Check if API key env var is actually set before attempting API calls
printenv "$FW_API_KEY_ENV_VAR" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "WARNING: Environment variable '$FW_API_KEY_ENV_VAR' not set or not exported. Skipping API model runs."
else
    echo "Found $FW_API_KEY_ENV_VAR. Proceeding with API runs..."
    for api_model_id in "${API_MODELS_TO_TEST[@]}"; do
        echo ""
        echo "======================================================"
        echo "Starting API Test for Model: $api_model_id"
        echo "======================================================"
        safe_model_name=$(echo "$api_model_id" | tr '/' '_') # Sanitize name
        output_filename_api="${OUTPUT_DIR}/ranked_options_logprobs_${safe_model_name}_API.jsonl"
        echo "Output file: $output_filename_api"

        # Call python script with API arguments
        # Use || true to allow the script to continue even if one API model fails
        ( PYTHONPATH="$PROJECT_ROOT:$EXAMPLE_DIR:$PYTHONPATH" python "$SRC_DIR/rank_options_logprobs.py" \
            --input_data_file "$INPUT_DATA_FILE" \
            --output_file "$output_filename_api" \
            --api_model_id "$api_model_id" \ # <-- Corrected argument passing
            --fw_api_key_env "$FW_API_KEY_ENV_VAR" \
            --max_context_len "$MAX_CONTEXT_CHARS" \
        && echo "--- API Run finished successfully. ---" ) || echo "--- API Run FAILED for $api_model_id (Check Logs Above) ---"
        # Removed args not relevant for API

        echo "======================================================"
        echo "Finished API Test for Model: $api_model_id"
        echo "======================================================"
        echo ""
        sleep 3 # Pause between API model runs
    done # <-- Correctly placed 'done' for the API loop
fi
echo "*********** FINISHED API MODEL RUNS ***********"
echo ""

echo "--- All Model Tests Attempted ---"