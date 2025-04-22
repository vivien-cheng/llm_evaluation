#!/usr/bin/env bash
set -e

echo "--- Workflow A - Step 2a: Filtering Responses (Semantic) ---" # Updated title

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
EXAMPLE_DIR=$(realpath "$SCRIPT_DIR/..")
SRC_DIR="$EXAMPLE_DIR/src"
INPUT_RESPONSES_DIR="$EXAMPLE_DIR/outputs/model_responses"
OUTPUT_FILTERED_FILE="$EXAMPLE_DIR/outputs/filtered_responses.jsonl" # Output used by next step (2b)
OUTPUT_DIR=$(dirname "$OUTPUT_FILTERED_FILE")

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Find all generated response files
MODEL_RESPONSE_FILES=("$INPUT_RESPONSES_DIR"/*.jsonl)

# Check if response files exist
if [ ${#MODEL_RESPONSE_FILES[@]} -eq 0 ] || [ ! -f "${MODEL_RESPONSE_FILES[0]}" ]; then
    echo "Error: No model response files found in $INPUT_RESPONSES_DIR. Run Step 1 first."
    exit 1
fi

echo "Found response files to filter:"
printf " - %s\n" "${MODEL_RESPONSE_FILES[@]}"

# --- Configuration for Semantic Filtering ---
EMBEDDING_MODEL="all-MiniLM-L6-v2" # Ensure sentence-transformers is installed
SIMILARITY_THRESHOLD=0.85
MAX_RESPONSES=1 # Keep only the top N distinct responses per query
# --- End Configuration ---

# Run the filtering script
PROJECT_ROOT=$(realpath "$EXAMPLE_DIR/../..")
PYTHONPATH="$PROJECT_ROOT:$EXAMPLE_DIR:$PYTHONPATH" python "$SRC_DIR/filter_responses.py" \
  --model_response_files "${MODEL_RESPONSE_FILES[@]}" \
  --output_file "$OUTPUT_FILTERED_FILE" \
  --embedding_model_name "$EMBEDDING_MODEL" \
  --similarity_threshold "$SIMILARITY_THRESHOLD" \
  --max_responses_per_query "$MAX_RESPONSES"

echo "Semantic filtering complete. Output: $OUTPUT_FILTERED_FILE"
echo "--------------------------------------------------------"
