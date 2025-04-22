#!/usr/bin/env bash
set -e # Exit on first error

echo "===== Running Full LongHealth Example Pipeline ====="

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# --- CHOOSE DATA SOURCE ---
# ... (Keep existing options) ...
bash "$SCRIPT_DIR/0_prepare_data.sh" # Default to real data subset

# --- Run Standard Steps ---
bash "$SCRIPT_DIR/1_generate_responses.sh"
bash "$SCRIPT_DIR/2_filter_responses.sh"
bash "$SCRIPT_DIR/3_evaluate.sh"

# --- NEW OPTIONAL STEP: Rank Options with Log Probs ---
echo ""
read -p "Do you want to run the Log Probability Ranking step (Step 4)? (y/N): " run_logprobs
if [[ "$run_logprobs" =~ ^[Yy]$ ]]; then
    bash "$SCRIPT_DIR/4_rank_options_logprobs.sh"
else
    echo "Skipping Step 4 (Log Probability Ranking)."
fi
# --- End New Step ---


echo "===== Pipeline Finished Successfully ====="