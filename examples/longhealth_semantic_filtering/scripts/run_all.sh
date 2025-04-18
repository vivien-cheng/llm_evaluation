#!/usr/bin/env bash
set -e # Exit on first error

echo "===== Running Full LongHealth Example Pipeline ====="

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# --- CHOOSE DATA SOURCE ---
# Select ONE of the options below by uncommenting the desired line
# and commenting out the other one.

# Option 1: Run with Dummy Data (Fastest for testing pipeline structure)
# bash "$SCRIPT_DIR/0_prepare_data.sh" --use-dummy

# Option 2: Run with Real Data Subset (Recommended for testing real data processing & generation)
# By default, 0_prepare_data.sh processes the first 10 items from benchmark_v5.json.
# You can change the subset size by editing MAX_ITEMS inside 0_prepare_data.sh.
bash "$SCRIPT_DIR/0_prepare_data.sh" # No flag = use real data subset (as configured in 0_prepare_data.sh)

# Option 3: Run with Full Real Data (VERY SLOW - Modify 0_prepare_data.sh to remove/increase MAX_ITEMS)
# bash "$SCRIPT_DIR/0_prepare_data.sh"
# --- END CHOOSE DATA SOURCE ---

# Run subsequent steps (Generation, Filtering, Evaluation)
bash "$SCRIPT_DIR/1_generate_responses.sh"
bash "$SCRIPT_DIR/2_filter_responses.sh"
bash "$SCRIPT_DIR/3_evaluate.sh"

echo "===== Pipeline Finished Successfully ====="

