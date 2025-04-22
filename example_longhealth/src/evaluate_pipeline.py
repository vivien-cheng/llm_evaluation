import argparse
import json
from collections import defaultdict
import numpy as np
import os
import sys
from typing import Dict, List, Any, Optional
from tqdm import tqdm # <--- ADD THIS IMPORT

# --- Setup Path ---
# Add the project root to sys.path to allow importing the core harness package
# This assumes the script is run from the examples/longhealth_semantic_filtering directory via `bash scripts/3_evaluate.sh`
# The script `3_evaluate.sh` sets PYTHONPATH correctly. If running directly, adjust this.
try:
    # This import should work if PYTHONPATH is set correctly by the calling script
    from hta_evaluation_harness.evaluator import HTAEvaluator
    from hta_evaluation_harness.utils import load_json, save_json
except ImportError:
    # Fallback if run directly without PYTHONPATH setup (requires setup.py install)
    print("Warning: Could not import hta_evaluation_harness directly. Ensure it's installed or PYTHONPATH is set.")
    # As a last resort, try modifying sys.path assuming a standard structure
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    try:
        from hta_evaluation_harness.evaluator import HTAEvaluator
        from hta_evaluation_harness.utils import load_json, save_json
    except ImportError as e:
         print(f"FATAL ERROR: Cannot import HTA Evaluation Harness. Install it ('pip install -e .') or check PYTHONPATH.")
         print(f"Sys.path: {sys.path}")
         print(f"Error: {e}")
         sys.exit(1)
# --- End Path Setup ---


def map_response_to_subtask(response_data: Dict[str, Any], query: str, index: int) -> Dict[str, Any]: # Use Dict[str, Any] for more specific typing
    """
    Maps a filtered response dictionary to the format expected by HTAEvaluator.evaluate_subtask.
    This mapping defines how the 5 HTA metrics are interpreted for a single response.
    """
    response_text = response_data.get('response_text', '')
    model_name = response_data.get('model', 'unknown_model')
    subtask_id = f"{response_data.get('query_id', 'q_unknown')}_resp_{index}_model_{model_name}"

    # How to interpret the response text for the metrics?
    # - Comprehensiveness: Length/detail of response_text.
    # - Task Coverage: How well response_text addresses the query.
    # - Usability: Can the steps in response_text be performed? (Assume info retrieval tool)
    # - Efficiency: Is response_text concise and direct?
    # - Accuracy: Is response_text factually correct (needs context/GT/judge)?

    return {
        'id': subtask_id,
        'goal': query, # The 'subtask goal' is the original query the response addresses
        'description': [response_text], # Treat the response as a single-step description
        'plan': response_text, # Can also use text as the 'plan'
        'required_tools': [], # Assume no specific tools explicitly required unless parsed from text
        # --- Add other fields if needed by specific metric implementations ---
        'response_text': response_text, # Keep for potential direct use in metrics
        'model_origin': model_name, # Track origin if needed by metrics
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate filtered LongHealth responses using the HTA Evaluation Harness.")
    parser.add_argument("--filtered_responses_file", type=str, required=True, help="Path to the JSONL file with filtered responses (output of filter_responses.py).")
    parser.add_argument("--original_data_file", type=str, required=True, help="Path to the original/processed data file (e.g., containing query, context, id).")
    parser.add_argument("--output_metrics_file", type=str, required=True, help="Path to save the evaluation metrics JSON file.")
    parser.add_argument("--harness_config_path", type=str, default=None, help="Optional path to harness config YAML (e.g., for LLM judge).")
    args = parser.parse_args()

    print("Starting Evaluation Pipeline...")
    print(f"Filtered Responses: {args.filtered_responses_file}")
    print(f"Original Data: {args.original_data_file}")
    print(f"Output Metrics: {args.output_metrics_file}")
    print(f"Harness Config: {args.harness_config_path or 'None'}")

    # --- Load Data ---
    try:
        filtered_responses = load_json(args.filtered_responses_file, load_by_line=True)
        if filtered_responses is None:
             raise FileNotFoundError(f"Filtered responses file not found or empty: {args.filtered_responses_file}")

        original_data_list = load_json(args.original_data_file, load_by_line=True)
        if original_data_list is None:
             raise FileNotFoundError(f"Original data file not found or empty: {args.original_data_file}")

        # Create a lookup for original context by query_id
        original_data_map = {item.get('id'): item for item in original_data_list if item.get('id')}
        if not original_data_map:
             print("Warning: Could not create lookup map from original data file (missing 'id' field?). Context might be missing for evaluation.")

    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        sys.exit(1)


    # --- Initialize the HTA Evaluator ---
    print("Initializing HTA Evaluator...")
    try:
        # Load config if path is provided
        harness_config = None
        if args.harness_config_path:
            print(f"Attempting to load harness config from: {args.harness_config_path}")
            # Use core utils load_json which handles basic JSON/YAML loading attempt
            # Need to refine utils.load_json or use PyYAML directly if config is always YAML
            try:
                 import yaml
                 with open(args.harness_config_path, 'r') as f:
                     harness_config = yaml.safe_load(f)
                 print("Harness config loaded successfully.")
            except ImportError:
                 print("Warning: PyYAML not installed, cannot load harness config file.")
            except Exception as e:
                 print(f"Warning: Failed to load harness config: {e}")

        evaluator = HTAEvaluator(config=harness_config) # Pass loaded config dict directly
    except Exception as e:
        print(f"FATAL ERROR: Failed to initialize HTAEvaluator: {e}")
        sys.exit(1)

    # --- Prepare Evaluation Data ---
    # Group filtered responses by the original query_id
    responses_by_query_id = defaultdict(list)
    for resp_data in filtered_responses:
        query_id = resp_data.get("query_id")
        if query_id:
            responses_by_query_id[query_id].append(resp_data)

    print(f"Prepared {len(responses_by_query_id)} unique queries for evaluation.")

    # --- Run Evaluation per Query ---
    all_results = {}
    # Define assumed available tools for the context of LongHealth queries
    # In a real scenario, this might depend on the agent setup.
    available_tools: List[str] = ["web_search", "medical_database_lookup", "information_retrieval"] # Added type hint for clarity

    # Use tqdm for the progress bar
    for query_id, responses in tqdm(responses_by_query_id.items(), desc="Evaluating queries"):
        # Find the original query text and context
        original_item = original_data_map.get(query_id)
        if not original_item:
            # Use print inside tqdm loop for warnings
            # print(f"Warning: Skipping query_id '{query_id}' - matching original data not found.")
            continue # Skip this iteration

        original_query = original_item.get('query', '')
        original_context = original_item.get('context', '')
        # gold_answer = original_item.get('gold_answer') # Get ground truth if available

        if not original_query:
            # Use print inside tqdm loop for warnings
            # print(f"Warning: Skipping query_id '{query_id}' - original query text is empty.")
            continue # Skip this iteration

        # Use print statement instead of tqdm description for better logging within loop
        # print(f"\nEvaluating Query ID: {query_id} ('{original_query[:50]}...')")

        # Map the list of responses to the HTA structure (treating each response as a 'subtask')
        generated_subtasks: List[Dict[str, Any]] = [] # Added type hint
        for i, resp_data in enumerate(responses):
            subtask = map_response_to_subtask(resp_data, original_query, i)
            generated_subtasks.append(subtask)

        # Evaluate this "HTA" (list of response-subtasks) for the original query
        # Note: Ground truth HTA is likely not applicable here unless you have GT *responses*.
        # We pass original_context for potential use in accuracy metric.
        evaluation_output = evaluator.evaluate_hta(
            generated_subtasks=generated_subtasks,
            original_goal=original_query,
            available_tools=available_tools,
            ground_truth_hta=None, # No GT HTA for responses
            context=original_context # Pass context for accuracy check
        )

        # Store results, adding query/context for reference
        all_results[query_id] = {
            "query": original_query,
            "context": original_context,
            **evaluation_output # Includes 'subtask_scores' and 'average_scores'
        }

    # --- Save Results ---
    print(f"\nSaving evaluation results to {args.output_metrics_file}...")
    try:
        save_json(all_results, args.output_metrics_file, indent=2)
        print("Evaluation pipeline finished successfully.")
    except Exception as e: # Catch broader exceptions during save
        print(f"Error saving evaluation results: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()