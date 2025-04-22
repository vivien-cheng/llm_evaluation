#!/usr/bin/env bash
# Runs the Semantic Filter + LLM Judge evaluation pipeline (Steps 0 -> 1 -> 2a -> 2b)
set -e

echo "===== Running Workflow A: Semantic Filter + LLM Judge ====="

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Run steps sequentially
bash "$SCRIPT_DIR/0_prepare_data.sh" "$@" # Pass any args like --use-dummy
bash "$SCRIPT_DIR/1_generate_responses.sh"
bash "$SCRIPT_DIR/2a_filter_semantic.sh"
bash "$SCRIPT_DIR/2b_evaluate_llm_judge.sh"

echo "===== Workflow A Complete ====="
