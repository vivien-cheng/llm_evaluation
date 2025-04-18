#!/usr/bin/env bash
set -e

echo "--- Step 3: Evaluating Filtered Responses (using LLM Judge) ---"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
EXAMPLE_DIR=$(realpath "$SCRIPT_DIR/..")
SRC_DIR="$EXAMPLE_DIR/src"
FILTERED_RESPONSES_FILE="$EXAMPLE_DIR/outputs/filtered_responses.jsonl"
ORIGINAL_DATA_FILE="$EXAMPLE_DIR/data/processed/longhealth_cleaned.jsonl" # Use processed file for context/query/etc.
OUTPUT_METRICS_FILE="$EXAMPLE_DIR/outputs/evaluation_results.json"
OUTPUT_DIR=$(dirname "$OUTPUT_METRICS_FILE")

# --- Define Path to Harness Config ---
PROJECT_ROOT=$(realpath "$EXAMPLE_DIR/../..")
HARNESS_CONFIG_PATH="$PROJECT_ROOT/config/models.yaml" # Path to your judge config
# --- End Config Path ---

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Check if required files exist
if [ ! -f "$FILTERED_RESPONSES_FILE" ]; then
    echo "Error: Filtered responses file not found at $FILTERED_RESPONSES_FILE. Run Step 2 first."
    exit 1
fi
if [ ! -f "$ORIGINAL_DATA_FILE" ]; then
    echo "Error: Original data file not found at $ORIGINAL_DATA_FILE. Run Step 0 first."
    exit 1
fi
if [ ! -f "$HARNESS_CONFIG_PATH" ]; then
    echo "Error: Harness config file not found at $HARNESS_CONFIG_PATH."
    echo "Please ensure config/models.yaml is configured for the LLM judge."
    exit 1
fi

# Run the evaluation script, passing the config path
echo "Running evaluation using config: $HARNESS_CONFIG_PATH"
PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python "$SRC_DIR/evaluate_pipeline.py" \
  --filtered_responses_file "$FILTERED_RESPONSES_FILE" \
  --original_data_file "$ORIGINAL_DATA_FILE" \
  --output_metrics_file "$OUTPUT_METRICS_FILE" \
  --harness_config_path "$HARNESS_CONFIG_PATH" # <-- Pass the config path

echo "Evaluation complete. Metrics saved to: $OUTPUT_METRICS_FILE"
echo "-----------------------------------------"
