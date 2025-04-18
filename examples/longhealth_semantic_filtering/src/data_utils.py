import json
import argparse
import os
import sys # Import sys
from tqdm import tqdm
from typing import Dict, List, Any

def process_patient_data(patient_id: str, patient_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Processes questions for a single patient, concatenating all their texts."""
    processed_items = []
    texts = patient_info.get("texts", {})
    questions = patient_info.get("questions", [])

    # Concatenate all available text documents for this patient
    # Ensure consistent ordering by sorting text keys (e.g., text_0, text_1)
    sorted_text_keys = sorted(texts.keys())
    full_context = "\n\n--- Document Separator ---\n\n".join(
        texts[key] for key in sorted_text_keys if texts[key]
    )

    if not full_context:
        print(f"Warning: No text found for patient {patient_id}. Skipping questions.")
        return []

    for question_data in questions:
        question_no = question_data.get("No")
        query = question_data.get("question")
        options = {
            "A": question_data.get("answer_a"),
            "B": question_data.get("answer_b"),
            "C": question_data.get("answer_c"),
            "D": question_data.get("answer_d"),
            "E": question_data.get("answer_e"),
        }
        # Filter out options that are None or empty strings
        options = {k: v for k, v in options.items() if v}

        correct_answer_text = question_data.get("correct")
        correct_answer_letter = None
        for letter, text in options.items():
            # Find the letter corresponding to the correct text
            if text and correct_answer_text and text.strip() == correct_answer_text.strip():
                correct_answer_letter = letter
                break

        if not query or not correct_answer_letter:
            print(f"Warning: Skipping question {question_no} for patient {patient_id} due to missing query or unmatchable correct answer.")
            continue

        # Create the output dictionary for this question item
        item_id = f"{patient_id}_q{question_no}"
        processed_item = {
            "id": item_id,
            "patient_id": patient_id,
            "question_no": question_no,
            "context": full_context,
            "query": query,
            "options": options,
            "correct_answer_letter": correct_answer_letter,
            "correct_answer_text": correct_answer_text # Keep original correct text for reference
        }
        processed_items.append(processed_item)

    return processed_items


def main():
    parser = argparse.ArgumentParser(description="Process LongHealth benchmark JSON data into JSONL format.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input benchmark JSON file (e.g., benchmark_v5.json).")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output JSONL file.")
    # --- ADDED ARGUMENT ---
    parser.add_argument("--max_items", type=int, default=None, help="Maximum number of question items to process (for testing).")
    args = parser.parse_args()

    print("Starting data preparation...")
    print(f"Input Benchmark File: {args.input_file}")
    print(f"Output JSONL File: {args.output_file}")
    if args.max_items:
        print(f"Processing a maximum of {args.max_items} question items.")

    try:
        with open(args.input_file, 'r', encoding='utf-8') as f_in:
            data = json.load(f_in)
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input_file}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {args.input_file}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred loading the input file: {e}")
        sys.exit(1)

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_file)
    if output_dir: # Create dir only if path includes one
        os.makedirs(output_dir, exist_ok=True)

    total_items_processed = 0
    all_processed_data = []

    print(f"Loaded data for {len(data)} patients.")

    # Iterate through patients and process their questions; use tqdm for progress bar over patients
    for patient_id, patient_info in tqdm(data.items(), desc="Processing patients"):
        # print(f"  Processing patient: {patient_id}...") # Reduce verbosity
        patient_items = process_patient_data(patient_id, patient_info)
        for item in patient_items:
             all_processed_data.append(item)
             total_items_processed += 1
             # --- ADDED CHECK ---
             if args.max_items is not None and total_items_processed >= args.max_items:
                 print(f"\nReached max_items limit ({args.max_items}). Stopping processing.")
                 break # Break inner loop
        if args.max_items is not None and total_items_processed >= args.max_items:
             break # Break outer loop

    # Write processed data to JSONL file
    try:
        with open(args.output_file, 'w', encoding='utf-8') as f_out:
            for item in all_processed_data:
                f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
    except IOError as e:
        print(f"Error writing output file {args.output_file}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred writing the output file: {e}")
        sys.exit(1)


    print(f"\nSuccessfully processed {total_items_processed} question items.")
    print(f"Processed data saved to: {args.output_file}")
    print("Data preparation finished.")

if __name__ == "__main__":
    main()