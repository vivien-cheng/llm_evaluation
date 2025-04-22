import argparse
import json
from collections import defaultdict
import numpy as np
import os
import sys
from typing import Dict, List, Any, Optional
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

# --- Setup Path ---
try:
    from hta_evaluation_harness.evaluator import HTAEvaluator
    from hta_evaluation_harness.utils import load_json, save_json
except ImportError:
    print("Warning: Could not import hta_evaluation_harness directly. Ensuring PYTHONPATH is set correctly...")
    # Adjust path resolution for new directory structure
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    example_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    if example_root not in sys.path:
        sys.path.insert(0, example_root)
        
    try:
        from hta_evaluation_harness.evaluator import HTAEvaluator
        from hta_evaluation_harness.utils import load_json, save_json
    except ImportError as e:
         print(f"FATAL ERROR: Cannot import HTA Evaluation Harness. Install it ('pip install -e .') or check PYTHONPATH.")
         print(f"Project Root: {project_root}")
         print(f"Example Root: {example_root}")
         print(f"Sys.path: {sys.path}")
         print(f"Error: {e}")
         sys.exit(1)
# --- End Path Setup ---

# map_response_to_subtask function remains the same as before
def map_response_to_subtask(response_data: Dict[str, Any], query: str, index: int) -> Dict[str, Any]:
    """
    Maps a filtered response dictionary to the format expected by HTAEvaluator.evaluate_subtask.
    """
    response_text = response_data.get('response_text', '')
    model_name = response_data.get('model', 'unknown_model')
    # Use the query_id that evaluate_pipeline uses for grouping
    query_id_for_subtask = response_data.get("query_id", response_data.get("id", "q_unknown"))
    subtask_id = f"{query_id_for_subtask}_resp_{index}_model_{model_name}"

    return {
        'id': subtask_id,
        'goal': query,
        'description': [response_text],
        'plan': response_text,
        'required_tools': [],
        'response_text': response_text,
        'model_origin': model_name,
        # Pass through other relevant fields needed by metrics
        'correct_answer_letter': response_data.get("correct_answer_letter"),
        'correct_answer_text': response_data.get("correct_answer_text"),
        'options': response_data.get("options"),
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate filtered LongHealth responses using the HTA Evaluation Harness.")
    parser.add_argument("--filtered_responses_file", type=str, required=True, help="Path to the JSONL file with filtered responses.")
    parser.add_argument("--original_data_file", type=str, required=True, help="Path to the original/processed data file.")
    parser.add_argument("--output_metrics_file", type=str, required=True, help="Path to save the evaluation metrics.")
    parser.add_argument("--harness_config_path", type=str, default=None, help="Optional path to harness config YAML.")
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

        # Create a lookup map for original data by query ID ('id' field in original data)
        original_data_map = {item.get('id'): item for item in original_data_list if item.get('id')}
        if not original_data_map:
             print("Warning: Could not create lookup map from original data file (missing 'id' field?). Context/Query might be missing.")

    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        sys.exit(1)

    # --- Initialize the HTA Evaluator ---
    # (Initialization logic remains the same)
    print("Initializing HTA Evaluator...")
    try:
        harness_config = None
        if args.harness_config_path:
            print(f"Attempting to load harness config from: {args.harness_config_path}")
            try:
                 import yaml
                 with open(args.harness_config_path, 'r') as f:
                     harness_config = yaml.safe_load(f)
                 print("Harness config loaded successfully.")
            except ImportError:
                 print("Warning: PyYAML not installed, cannot load harness config file.")
            except Exception as e:
                 print(f"Warning: Failed to load harness config: {e}")

        evaluator = HTAEvaluator(config=harness_config)
    except Exception as e:
        print(f"FATAL ERROR: Failed to initialize HTAEvaluator: {e}")
        sys.exit(1)

    # --- Prepare Evaluation Data ---
    # Group filtered responses by the original query_id
    responses_by_query_id = defaultdict(list)
    print("Grouping filtered responses...")
    for resp_data in filtered_responses:
        # --- MODIFIED ID CHECK ---
        # Try 'query_id' first (what filter_responses *should* save)
        query_id = resp_data.get("query_id")
        # If 'query_id' is missing or None/empty, try falling back to 'id'
        if not query_id:
            query_id = resp_data.get("id")
        # --- END MODIFIED ID CHECK ---

        # Proceed only if we found a valid identifier
        if query_id:
            responses_by_query_id[query_id].append(resp_data)
        else:
            # Add a warning if neither key yields a valid ID
            print(f"Warning: Skipping response due to missing or invalid 'query_id' or 'id': {resp_data}")

    print(f"Prepared {len(responses_by_query_id)} unique queries for evaluation.") # This should now be > 0

    # --- Run Evaluation per Query ---
    all_results = {}
    available_tools: List[str] = ["web_search", "medical_database_lookup", "information_retrieval"]

    if not responses_by_query_id:
         print("Error: No queries prepared for evaluation. Check filtered responses file format and ID keys ('query_id' or 'id').")
    else:
        for query_id, responses in tqdm(responses_by_query_id.items(), desc="Evaluating queries"):
            # Find the original query text and context using the ID
            original_item = original_data_map.get(query_id) # Use query_id (which might be 'q1', 'q2', etc.) to lookup in map
            if not original_item:
                print(f"Warning: Skipping query_id '{query_id}' - matching original data not found in map.")
                continue

            original_query = original_item.get('query', '')
            original_context = original_item.get('context', '') # Context from original dummy/processed file

            if not original_query:
                print(f"Warning: Skipping query_id '{query_id}' - original query text is empty.")
                continue

            # Map the list of responses to the HTA structure
            generated_subtasks: List[Dict[str, Any]] = []
            for i, resp_data in enumerate(responses):
                # Pass original item data to map_response_to_subtask if needed
                subtask = map_response_to_subtask(resp_data, original_query, i)
                generated_subtasks.append(subtask)

            # Evaluate this "HTA" (list of response-subtasks)
            evaluation_output = evaluator.evaluate_hta(
                generated_subtasks=generated_subtasks,
                original_goal=original_query,
                available_tools=available_tools,
                ground_truth_hta=None,
                context=original_context # Pass context for accuracy check by LLM Judge
            )

            # Store results
            all_results[query_id] = {
                "query": original_query,
                # "context": original_context, # Maybe exclude huge context from final results
                **evaluation_output
            }

    # --- Save Results ---
    print(f"\nSaving evaluation results to {args.output_metrics_file}...")
    try:
        save_json(all_results, args.output_metrics_file, indent=2)
        print("Evaluation pipeline finished successfully.")
    except Exception as e:
        print(f"Error saving evaluation results: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
