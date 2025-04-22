import argparse
import json
import os
import glob
import logging
from collections import defaultdict

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_accuracy_from_file(file_path: str) -> tuple[int, int, float | None]:
    """
    Reads a JSONL file containing ranked options and calculates Top-1 accuracy.

    Args:
        file_path: Path to the .jsonl file.

    Returns:
        A tuple containing: (correct_count, total_processed, accuracy_percentage).
        Returns (0, 0, None) if the file cannot be processed or is empty.
    """
    correct_count = 0
    total_processed = 0
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return 0, 0, None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                try:
                    item = json.loads(line)
                    total_processed += 1

                    # Validate necessary keys
                    if "ranked_options" not in item or "correct_answer_letter" not in item:
                        logger.warning(f"Skipping line {i+1} in {os.path.basename(file_path)}: Missing 'ranked_options' or 'correct_answer_letter'.")
                        continue
                    if not item["ranked_options"]:
                        logger.warning(f"Skipping line {i+1} in {os.path.basename(file_path)}: 'ranked_options' list is empty.")
                        continue
                    if item["correct_answer_letter"] is None:
                         logger.warning(f"Skipping line {i+1} in {os.path.basename(file_path)}: 'correct_answer_letter' is null.")
                         continue


                    # Get the top-ranked letter and the correct letter
                    top_ranked_letter = item["ranked_options"][0].get("letter")
                    correct_letter = item["correct_answer_letter"]

                    if top_ranked_letter is None:
                         logger.warning(f"Skipping line {i+1} in {os.path.basename(file_path)}: Top ranked option has no 'letter'.")
                         continue

                    # Compare (case-insensitive just in case)
                    if top_ranked_letter.strip().upper() == correct_letter.strip().upper():
                        correct_count += 1

                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line {i+1} in {os.path.basename(file_path)}: {line[:100]}...")
                except Exception as e:
                    logger.warning(f"Error processing line {i+1} in {os.path.basename(file_path)}: {e}")

        if total_processed == 0:
            logger.warning(f"No valid items processed in file: {file_path}")
            return 0, 0, None

        accuracy = (correct_count / total_processed) * 100
        return correct_count, total_processed, accuracy

    except Exception as e:
        logger.error(f"Failed to read or process file {file_path}: {e}")
        return 0, 0, None

def parse_filename(filename: str) -> dict:
    """Parses model details from the filename."""
    parts = os.path.basename(filename).replace("ranked_options_logprobs_", "").replace(".jsonl", "").split('_')
    details = {"Filename": os.path.basename(filename)} # Keep original filename
    if len(parts) >= 3:
        details["Quantized"] = "Quant" in parts[-1]
        details["Method"] = parts[-2] # Should be Local or API
        details["Model Name"] = "_".join(parts[:-2]) # Join remaining parts for model name
    else:
        # Fallback if filename format is unexpected
        details["Model Name"] = os.path.basename(filename)
        details["Method"] = "Unknown"
        details["Quantized"] = "Unknown"
    return details


def main():
    parser = argparse.ArgumentParser(description="Calculate Top-1 accuracy from logprob ranking JSONL files.")
    parser.add_argument("--results_dir", type=str, required=True, help="Path to the directory containing the ranked_options_logprobs_*.jsonl files.")
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        logger.error(f"Results directory not found: {args.results_dir}")
        return

    # Find all relevant files
    file_pattern = os.path.join(args.results_dir, "ranked_options_logprobs_*.jsonl")
    result_files = glob.glob(file_pattern)

    if not result_files:
        logger.warning(f"No result files found matching pattern '{file_pattern}'.")
        return

    logger.info(f"Found {len(result_files)} result files to analyze.")

    results_summary = []
    for file_path in sorted(result_files):
        logger.info(f"Analyzing file: {os.path.basename(file_path)}...")
        correct, total, accuracy = calculate_accuracy_from_file(file_path)
        if accuracy is not None:
            file_details = parse_filename(os.path.basename(file_path))
            results_summary.append({
                "Model": file_details["Model Name"].replace("_", "/"), # Try to restore original format
                "Quant": "Yes" if file_details["Quantized"] else "No",
                "Correct": correct,
                "Total": total,
                "Accuracy (%)": f"{accuracy:.1f}",
                "File": file_details["Filename"] # Include filename for reference
            })
        else:
             logger.warning(f"Could not calculate accuracy for {os.path.basename(file_path)}")

    # --- Print Summary Table ---
    if not results_summary:
        logger.info("No results to display.")
        return

    # Determine column widths
    headers = ["Model", "Quant", "Correct", "Total", "Accuracy (%)", "File"]
    col_widths = {h: len(h) for h in headers}
    for row in results_summary:
        for h in headers:
            col_widths[h] = max(col_widths[h], len(str(row.get(h, ''))))

    # Print header
    header_line = " | ".join(f"{h:<{col_widths[h]}}" for h in headers)
    separator = "-+-".join("-" * col_widths[h] for h in headers)
    print("\n--- Accuracy Summary ---")
    print(header_line)
    print(separator)

    # Print rows
    for row in sorted(results_summary, key=lambda x: (x["Model"], x["Quant"])): # Sort for consistent output
        print(" | ".join(f"{str(row.get(h, '')):<{col_widths[h]}}" for h in headers))

    print("------------------------\n")


if __name__ == "__main__":
    main()
