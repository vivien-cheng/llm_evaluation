#!/usr/bin/env bash
set -e
echo "--- Step 1: Generating Responses (Actual Models) ---"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
EXAMPLE_DIR=$(realpath "$SCRIPT_DIR/..")
SRC_DIR="$EXAMPLE_DIR/src"
INPUT_FILE="$EXAMPLE_DIR/data/processed/longhealth_cleaned.jsonl"
OUTPUT_RESPONSES_DIR="$EXAMPLE_DIR/outputs/model_responses"

# Ensure output directory exists
mkdir -p "$OUTPUT_RESPONSES_DIR"

# Check if processed data exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Processed data file not found at $INPUT_FILE. Run Step 0 first."
    exit 1
fi

# --- Configure Models and Parameters ---
MODELS=(
    "Qwen/Qwen2.5-1.5B-Instruct"
    "Qwen/Qwen2.5-3B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
)
# --- Use parameters suitable for debugging/MCQs ---
NUM_RESPONSES=1       # Generate only 1 response per query
TEMPERATURE=0.7       # <<< CHANGED TEMPERATURE TO 0.7
MAX_NEW_TOKENS=50     # Small output length needed for MC answer
TOP_P=0.9
MAX_INPUT_TOKENS=32000 # Max length for input truncation
# --- End Configuration ---

# Generate responses for each model sequentially
for model_id in "${MODELS[@]}"
do
  echo "-----------------------------------------------------"
  echo "Generating responses for model: $model_id"
  echo "-----------------------------------------------------"
  safe_model_name=$(echo "$model_id" | tr '/' '_')
  OUTPUT_FILE="$OUTPUT_RESPONSES_DIR/${safe_model_name}_responses.jsonl"

  PROJECT_ROOT=$(realpath "$EXAMPLE_DIR/../..")

  # Run the python script passing all arguments
  PYTHONPATH="$PROJECT_ROOT:$EXAMPLE_DIR:$PYTHONPATH" python "$SRC_DIR/generate_responses.py" \
    --model_name_or_path "$model_id" \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --num_responses "$NUM_RESPONSES" \
    --temperature "$TEMPERATURE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --top_p "$TOP_P" \
    --max_input_tokens "$MAX_INPUT_TOKENS" # Pass max input length

  echo "Responses for $model_id saved to $OUTPUT_FILE"
  echo "--- Model $model_id Complete ---"

done

echo "-----------------------------------------------------"
echo "Response generation complete for all models."
echo "-----------------------------------------------------"

