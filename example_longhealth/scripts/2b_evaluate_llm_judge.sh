#!/usr/bin/env bash
set -e

echo "--- Workflow A - Step 2b: Evaluating Filtered Responses (using LLM Judge) ---" # Updated title

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
EXAMPLE_DIR=$(realpath "$SCRIPT_DIR/..")
SRC_DIR="$EXAMPLE_DIR/src"
FILTERED_RESPONSES_FILE="$EXAMPLE_DIR/outputs/filtered_responses.jsonl" # Input from Step 2a
ORIGINAL_DATA_FILE="$EXAMPLE_DIR/data/processed/longhealth_cleaned.jsonl" # Input from Step 0
OUTPUT_METRICS_FILE="$EXAMPLE_DIR/outputs/evaluation_results.json" # Final HTA metrics
OUTPUT_DIR=$(dirname "$OUTPUT_METRICS_FILE")

# --- Define Path to Harness Config ---
PROJECT_ROOT=$(realpath "$EXAMPLE_DIR/../..")
# Ensure this path points to your main config file
HARNESS_CONFIG_PATH="$PROJECT_ROOT/config/models.yaml"
# --- End Config Path ---

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Check if required files exist
if [ ! -f "$FILTERED_RESPONSES_FILE" ]; then
    echo "Error: Filtered responses file not found at $FILTERED_RESPONSES_FILE. Run Step 2a first."
    exit 1
fi
if [ ! -f "$ORIGINAL_DATA_FILE" ]; then
    echo "Error: Original data file not found at $ORIGINAL_DATA_FILE. Run Step 0 first."
    exit 1
fi
if [ ! -f "$HARNESS_CONFIG_PATH" ]; then
    echo "Error: Harness config file not found at $HARNESS_CONFIG_PATH."
    echo "Please ensure config/models.yaml exists and is configured for the LLM judge (e.g., OpenAI API key in .env)."
    exit 1
fi

# Run the evaluation script, passing the config path
echo "Running LLM Judge evaluation using config: $HARNESS_CONFIG_PATH"
# Ensure python can find evaluate_pipeline.py and the harness library
PYTHONPATH="$PROJECT_ROOT:$EXAMPLE_DIR:$PYTHONPATH" python "$SRC_DIR/evaluate_pipeline.py" \
  --filtered_responses_file "$FILTERED_RESPONSES_FILE" \
  --original_data_file "$ORIGINAL_DATA_FILE" \
  --output_metrics_file "$OUTPUT_METRICS_FILE" \
  --harness_config_path "$HARNESS_CONFIG_PATH" # Pass the config path

echo "LLM Judge evaluation complete. Metrics saved to: $OUTPUT_METRICS_FILE"
echo "-------------------------------------------------------------------"
