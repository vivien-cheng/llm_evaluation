#!/usr/bin/env bash
set -e # Exit immediately if a command exits with a non-zero status.

# --- Default values ---
USE_DUMMY=false
MAX_ITEMS=10          # Default num of items to process from real data for testing

# --- Parse Command Line Arguments ---
# Simple parsing: check if --use-dummy is present
for arg in "$@"
do
    case $arg in
        --use-dummy)
        USE_DUMMY=true
        shift # Remove --use-dummy from processing
        ;;
    esac
done

# --- Define Paths ---
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
EXAMPLE_DIR=$(realpath "$SCRIPT_DIR/..") # Navigate up to the example root (e.g., example_longhealth)
SRC_DIR="$EXAMPLE_DIR/src"
RAW_DATA_DIR="$EXAMPLE_DIR/data/raw"
PROCESSED_DATA_DIR="$EXAMPLE_DIR/data/processed"
PROCESSED_DATA_FILE="$PROCESSED_DATA_DIR/longhealth_cleaned.jsonl"
PROJECT_ROOT=$(realpath "$EXAMPLE_DIR/../..") # Navigate up to the main project root (e.g., llm_evaluation)

# Input file paths
DUMMY_DATA_FILE="$RAW_DATA_DIR/longhealth_dummy.jsonl"
REAL_DATA_FILE="$RAW_DATA_DIR/benchmark_v5.json"

# Ensure output directory exists
mkdir -p "$PROCESSED_DATA_DIR"

# --- Conditional Logic ---
if [ "$USE_DUMMY" = true ]; then
    # --- Use Dummy Data ---
    echo "--- Step 0: Preparing Data (USING DUMMY DATA) ---"
    if [ ! -f "$DUMMY_DATA_FILE" ]; then
        echo "Error: Dummy data file not found at $DUMMY_DATA_FILE"
        exit 1
    fi
    echo "Copying dummy data from $DUMMY_DATA_FILE to $PROCESSED_DATA_FILE..."
    # Overwrite existing processed file with dummy data
    cp "$DUMMY_DATA_FILE" "$PROCESSED_DATA_FILE"
    echo "Dummy data preparation complete. Output: $PROCESSED_DATA_FILE"

else
    # --- Use Real Data (Subset) ---
    echo "--- Step 0: Preparing Data (USING REAL BENCHMARK DATA - Subset: ${MAX_ITEMS} items) ---"
    if [ ! -f "$REAL_DATA_FILE" ]; then
        echo "Error: Real benchmark data file not found at $REAL_DATA_FILE"
        echo "Please download benchmark_v5.json from the LongHealth repository or source and place it there."
        exit 1
    fi
    echo "Processing real benchmark data using data_utils.py (max ${MAX_ITEMS} items)..."

    # Run the data processing script, ensuring PYTHONPATH includes project root
    # Pass the --max_items argument to the python script
    PYTHONPATH="$PROJECT_ROOT:$EXAMPLE_DIR:$PYTHONPATH" python "$SRC_DIR/data_utils.py" \
       --input_file "$REAL_DATA_FILE" \
       --output_file "$PROCESSED_DATA_FILE" \
       --max_items "$MAX_ITEMS" # <<< Ensures only MAX_ITEMS are processed

    echo "Data preparation complete (subset: ${MAX_ITEMS} items). Output: $PROCESSED_DATA_FILE"
fi

echo "------------------------------------------------------"
