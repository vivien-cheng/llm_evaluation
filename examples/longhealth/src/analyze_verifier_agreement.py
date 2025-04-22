import argparse
import json
import os
import glob
import logging
from collections import defaultdict, Counter

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Basic local file handling functions ---
def load_json_file(file_path):
    """Loads a single JSON file."""
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {e}")
        return None

# --- FIX: Add save_json function definition ---
def save_json(data, file_path, save_by_line=False, indent=2):
    """Saves data to a JSON or JSONL file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            if save_by_line:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            else:
                json.dump(data, f, ensure_ascii=False, indent=indent)
        logger.info(f"Data successfully saved to: {file_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON to {file_path}: {e}")
# --- End Fix ---


def analyze_agreement(eval_files_dir: str, output_analysis_file: str):
    """
    Analyzes verifier agreement based on multiple evaluation summary files.

    Args:
        eval_files_dir: Directory containing the eval_*.json files from Step 2c runs
                        (specifically from Setting 2: Fixed Generator, Varying Verifiers).
        output_analysis_file: Path to save the agreement analysis summary.
    """
    logger.info(f"--- Starting Verifier Agreement Analysis ---")
    logger.info(f"Reading evaluation files from: {eval_files_dir}")

    # Find files matching the pattern for Setting 2 outputs
    file_pattern = os.path.join(eval_files_dir, "eval_FIXED_GEN_*.json")
    eval_files = glob.glob(file_pattern)

    if not eval_files:
        logger.error(f"No evaluation files matching '{file_pattern}' found in {eval_files_dir}. Run Step 2c (Setting 2) first.")
        return

    logger.info(f"Found {len(eval_files)} evaluation files.")

    # Structure to hold chosen answer per query per verifier
    # { query_id: { verifier_key: chosen_generated_text_or_None, ... }, ... }
    query_choices = defaultdict(dict)
    verifier_accuracies = {} # Store individual accuracies {verifier_key: accuracy}

    for file_path in eval_files:
        data = load_json_file(file_path)
        if not data:
            logger.warning(f"Skipping invalid file: {file_path}")
            continue

        verifier_model = data.get("verifier_model")
        quantized = data.get("quantized", False)
        # Use top1 accuracy for consistency
        accuracy = data.get("accuracy_percentage_top1")
        details = data.get("evaluation_details")

        if not verifier_model or details is None or accuracy is None:
            logger.warning(f"Skipping file {os.path.basename(file_path)}: Missing verifier_model, evaluation_details, or accuracy.")
            continue

        # Create a unique key for the verifier run
        verifier_key = f"{verifier_model}{'_Q' if quantized else '_NQ'}"
        verifier_accuracies[verifier_key] = accuracy
        logger.info(f"Processing results for verifier: {verifier_key} (Accuracy: {accuracy:.2f}%)")

        for item in details:
            query_id = item.get("id")
            if not query_id: continue # Skip if no ID

            best_response = item.get("best_response_by_logprob")
            # Store the text of the chosen response, or None if none chosen/scored
            chosen_text = best_response.get("generated_text") if best_response else None
            query_choices[query_id][verifier_key] = chosen_text


    # --- Calculate Agreement Metrics ---
    total_queries = len(query_choices)
    if total_queries == 0:
        logger.error("No query data loaded from evaluation files.")
        return

    num_verifiers = len(verifier_accuracies)
    if num_verifiers < 2:
        logger.warning(f"Only {num_verifiers} verifier result(s) found. Agreement analysis requires at least 2.")
        # Still save basic info
        analysis_output = { "error": "Insufficient verifier results for agreement analysis." }
        save_json(analysis_output, output_analysis_file, save_by_line=False, indent=2)
        return

    full_agreement_count = 0
    majority_agreement_count = 0 # Majority means > num_verifiers / 2

    agreement_details = {}

    for query_id, choices in query_choices.items():
        # Use Counter to find the frequency of each chosen response text
        # Handle None values (where a verifier failed to choose)
        valid_choices = [text for text in choices.values() if text is not None]
        choice_counts = Counter(valid_choices)
        most_common = choice_counts.most_common(1) # Gets [(text, count)] or []

        agreement_details[query_id] = {
            "choices_by_verifier": choices,
            "choice_counts": dict(choice_counts),
            "num_valid_choices": len(valid_choices)
        }

        if not most_common: # No valid choices made by any verifier
             agreement_details[query_id]["agreement_level"] = "None"
             continue

        top_choice_text, top_choice_count = most_common[0]

        # Check agreement among ALL verifiers that provided a choice
        num_participating_verifiers = len(choices) # How many verifiers processed this query ID
        if top_choice_count == num_participating_verifiers:
            full_agreement_count += 1
            agreement_details[query_id]["agreement_level"] = "Full"
            agreement_details[query_id]["agreed_upon_choice"] = top_choice_text
        elif top_choice_count > num_participating_verifiers / 2:
            majority_agreement_count += 1
            agreement_details[query_id]["agreement_level"] = "Majority"
            agreement_details[query_id]["agreed_upon_choice"] = top_choice_text
        else:
            agreement_details[query_id]["agreement_level"] = "Plurality/Disagreement"
            agreement_details[query_id]["agreed_upon_choice"] = top_choice_text # Still note the most common


    # --- Summarize ---
    logger.info("\n--- Verifier Agreement Summary ---")
    logger.info(f"Total Queries Analyzed: {total_queries}")
    logger.info(f"Number of Verifiers Compared: {num_verifiers}")
    logger.info(f"Individual Verifier Accuracies (%): {verifier_accuracies}")
    full_agreement_rate = full_agreement_count / total_queries if total_queries else 0
    majority_agreement_rate = majority_agreement_count / total_queries if total_queries else 0
    logger.info(f"Full Agreement Count (all participating verifiers chose same response): {full_agreement_count} ({full_agreement_rate:.1%})")
    logger.info(f"Majority Agreement Count (>50% participating verifiers chose same response): {majority_agreement_count} ({majority_agreement_rate:.1%})")

    # --- Save Analysis ---
    analysis_output = {
        "settings": {
            "eval_files_dir": eval_files_dir,
            "num_verifiers": num_verifiers,
            "verifiers": list(verifier_accuracies.keys()),
            "total_queries": total_queries
        },
        "summary_metrics": {
            "individual_accuracies_top1": verifier_accuracies,
            "full_agreement_count": full_agreement_count,
            "full_agreement_rate": full_agreement_rate,
            "majority_agreement_count": majority_agreement_count,
            "majority_agreement_rate": majority_agreement_rate,
        },
        "agreement_details_by_query": agreement_details
    }

    # This call should now work
    save_json(analysis_output, output_analysis_file, save_by_line=False, indent=2)

    logger.info(f"--- Verifier Agreement Analysis Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze verifier agreement from logprob evaluation files.")
    parser.add_argument("--eval_files_dir", type=str, required=True, help="Directory containing the eval_FIXED_GEN_*.json files.")
    parser.add_argument("--output_analysis_file", type=str, required=True, help="Path to save the agreement analysis summary JSON file.")
    args = parser.parse_args()
    analyze_agreement(args.eval_files_dir, args.output_analysis_file)
