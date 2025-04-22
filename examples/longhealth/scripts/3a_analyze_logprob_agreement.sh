#!/usr/bin/env bash
set -e

echo "--- Workflow B - Step 3a: Analyzing Verifier Agreement ---"

# --- Basic Setup ---
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
EXAMPLE_DIR=$(realpath "$SCRIPT_DIR/..")
SRC_DIR="$EXAMPLE_DIR/src"
# Directory containing evaluation summary files from Step 2c (Setting 2)
INPUT_EVAL_DIR="$EXAMPLE_DIR/outputs/logprob_evaluation"
# Output file for the agreement analysis
OUTPUT_ANALYSIS_FILE="$EXAMPLE_DIR/outputs/verifier_agreement_summary.json"
PROJECT_ROOT=$(realpath "$EXAMPLE_DIR/../..")

# --- Check if input directory exists ---
if [ ! -d "$INPUT_EVAL_DIR" ]; then
    echo "Error: Evaluation directory not found: $INPUT_EVAL_DIR"
    echo "Run Step 2c (Setting 2: Fixed Generator) first."
    exit 1
fi
if [ -z "$(ls -A $INPUT_EVAL_DIR/eval_FIXED_GEN_*.json 2>/dev/null)" ]; then
     echo "Error: No evaluation files matching 'eval_FIXED_GEN_*.json' found in $INPUT_EVAL_DIR."
     echo "Run Step 2c (Setting 2: Fixed Generator) first."
     exit 1
fi


# --- Run the Python analysis script ---
echo "Analyzing agreement from files in: $INPUT_EVAL_DIR"
echo "Saving analysis summary to: $OUTPUT_ANALYSIS_FILE"

PYTHONPATH="$PROJECT_ROOT:$EXAMPLE_DIR:$PYTHONPATH" python "$SRC_DIR/analyze_verifier_agreement.py" \
    --eval_files_dir "$INPUT_EVAL_DIR" \
    --output_analysis_file "$OUTPUT_ANALYSIS_FILE"

echo "--- Agreement Analysis Complete ---"
