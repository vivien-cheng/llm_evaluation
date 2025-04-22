#!/usr/bin/env bash
# Runs the Log Probability Filter evaluation pipeline (Steps 0 -> 1 -> 2c -> optional 3a)
set -e

echo "===== Running Workflow B: Log Probability Filter ====="

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Run steps sequentially
bash "$SCRIPT_DIR/0_prepare_data.sh" "$@" # Pass any args like --use-dummy
bash "$SCRIPT_DIR/1_generate_responses.sh"
# IMPORTANT: Configure desired EVALUATION_SETTING and models inside 2c script first!
bash "$SCRIPT_DIR/2c_evaluate_logprob_filter.sh"

# Optional: Run agreement analysis if Setting 2 was used in step 2c
echo ""
read -p "Run Verifier Agreement Analysis (Step 3a)? Requires Step 2c ran with Setting 2. (y/N): " run_analysis
if [[ "$run_analysis" =~ ^[Yy]$ ]]; then
    bash "$SCRIPT_DIR/3a_analyze_logprob_agreement.sh"
else
    echo "Skipping Step 3a (Agreement Analysis)."
fi


echo "===== Workflow B Complete ====="
