import argparse
import json
import os
import sys
import torch
import logging
import glob
import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from dotenv import load_dotenv

# --- Load Environment Variables ---
load_dotenv()

# --- Path Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
try:
    from hta_evaluation_harness.utils import load_json, save_json
except ImportError:
    print("Warning: Could not import utils from hta_evaluation_harness. Using basic local file handling.")
    # Basic local file handling functions (same as provided previously)
    def load_json(file_path, load_by_line=False):
         if not os.path.exists(file_path): return None
         data = []
         try:
             with open(file_path, 'r', encoding='utf-8') as f:
                 if load_by_line:
                     for line in f:
                         if line.strip():
                             try: data.append(json.loads(line))
                             except json.JSONDecodeError: print(f"Warning: Skipping invalid JSON line: {line[:100]}")
                     return data
                 else: # Try loading as single object or fallback to JSONL
                     content = f.read()
                     if not content.strip(): return []
                     try: return [json.loads(content)]
                     except json.JSONDecodeError:
                         f.seek(0)
                         for line in f:
                              if line.strip():
                                  try: data.append(json.loads(line))
                                  except json.JSONDecodeError: print(f"Warning: Skipping invalid JSON line: {line[:100]}")
                         return data
         except Exception as e: print(f"Error loading JSON from {file_path}: {e}"); return None

    def save_json(data, file_path, save_by_line=False, indent=2):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            if save_by_line:
                for item in data: f.write(json.dumps(item, ensure_ascii=False) + '\n')
            else: json.dump(data, f, ensure_ascii=False, indent=indent)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Model Loading (Re-used) ---
VERIFIER_MODEL_CACHE = {}
# (load_verifier_model function remains the same - include it here)
def load_verifier_model(model_name_or_path, device_type_str, use_quantization=False):
    """Loads the verifier model and tokenizer."""
    cache_key = f"{model_name_or_path}_{device_type_str}_{use_quantization}"
    if cache_key in VERIFIER_MODEL_CACHE:
        logger.info(f"Using cached verifier model/tokenizer for {model_name_or_path} on {device_type_str}")
        return VERIFIER_MODEL_CACHE[cache_key]

    logger.info(f"Loading verifier model: {model_name_or_path}...")
    try:
        quantization_config = None
        if use_quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            )
            logger.info("Using 4-bit quantization.")

        dtype = torch.float32
        if device_type_str == 'cuda' and torch.cuda.is_bf16_supported() and not use_quantization:
            dtype = torch.bfloat16

        logger.info(f"Attempting to load with dtype: {dtype}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=dtype, device_map="auto",
            trust_remote_code=True, quantization_config=quantization_config
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                logger.warning("pad_token not set, using eos_token as pad_token.")
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                logger.warning("Adding new pad token '<|pad|>'.")
                tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
                model.resize_token_embeddings(len(tokenizer))

        model.eval()
        VERIFIER_MODEL_CACHE[cache_key] = (model, tokenizer)
        try: device_map_str = str(model.hf_device_map)
        except Exception: device_map_str = "N/A"
        logger.info(f"Verifier model {model_name_or_path} loaded successfully. Device map: {device_map_str}")
        return model, tokenizer

    except Exception as e:
        logger.exception(f"FATAL ERROR loading verifier model {model_name_or_path}: {e}", exc_info=True)
        logger.error("Check model name, internet connection, access permissions, dependencies, VRAM.")
        sys.exit(1)

# --- Prompt Formatting (Re-used) ---
def format_scoring_prefix(context: str, query: str, max_context_len: int = 10000) -> str:
    """Formats the prefix part (context + query + instruction for completion)."""
    if len(context) > max_context_len:
        truncated_context = context[:max_context_len] + "\n... [Context Truncated] ..."
    else:
        truncated_context = context
    prompt = f"""Context:\n{truncated_context}\n\nQuery: {query}\n\nAnswer:"""
    return prompt

# --- Core Logic: Calculate AvgNLL for a Generated Response (Re-used) ---
@torch.no_grad()
def calculate_response_neg_log_likelihood(
    model, tokenizer, prefix_text: str, generated_response_text: str,
    use_cache: bool = True, max_length: int = 4096
    ) -> Optional[float]:
    """ Calculates AvgNLL for generated responses, handling device placement. """
    try:
        model_device = model.device
        prefix_inputs = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False)
        response_inputs = tokenizer(" " + generated_response_text.strip(), return_tensors="pt", add_special_tokens=False)

        prefix_ids = prefix_inputs.input_ids.to(model_device)
        response_ids = response_inputs.input_ids.to(model_device)

        if getattr(tokenizer, 'add_bos_token', False) and tokenizer.bos_token_id is not None:
             if prefix_ids.shape[1] == 0 or prefix_ids[0, 0] != tokenizer.bos_token_id:
                  bos_tensor = torch.tensor([[tokenizer.bos_token_id]], device=model_device)
                  prefix_ids = torch.cat([bos_tensor, prefix_ids], dim=1)

        combined_ids = torch.cat([prefix_ids, response_ids], dim=1)

        if combined_ids.shape[1] > max_length:
            logger.warning(f"Combined len {combined_ids.shape[1]} > max {max_length}. Skipping scoring for '{generated_response_text[:30]}...'.")
            return None

        prefix_len = prefix_ids.shape[1]
        labels = combined_ids.clone()
        labels[:, :prefix_len] = -100
        labels = labels.to(model_device)

        loss = None
        if use_cache:
            outputs_prefix = model(input_ids=prefix_ids, use_cache=True)
            past_key_values = outputs_prefix.past_key_values
            outputs_response = model(input_ids=response_ids, past_key_values=past_key_values, labels=labels[:, prefix_len:])
            loss = outputs_response.loss
        else:
            outputs_full = model(input_ids=combined_ids, labels=labels)
            loss = outputs_full.loss

        if loss is None or torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"Invalid loss ({loss}) for response '{generated_response_text[:30]}...'. Skipping.")
            return None
        return loss.item()
    except Exception as e:
        logger.error(f"Error calculating NLL for response '{generated_response_text[:50]}...': {e}", exc_info=True)
        return None

# --- Helper to Extract Answer Letter (Re-used) ---
def extract_answer_letter(text: str) -> Optional[str]:
    """Tries to extract A, B, C, D, or E from the beginning of the text."""
    if not text: return None
    match = re.match(r"^\s*([A-E])[\s:.)]*", text.strip(), re.IGNORECASE)
    if match: return match.group(1).upper()
    return None

# --- Main Evaluation Logic ---
def main():
    parser = argparse.ArgumentParser(description="Evaluate logprob filtering by scoring generated responses.")
    # Allow specifying response files directly or via directory
    parser.add_argument("--response_files", type=str, nargs='+', help="Paths to specific generated response JSONL files (e.g., for fixed generator).")
    parser.add_argument("--responses_dir", type=str, help="Directory containing generated response JSONL files (used if --response_files is not specified).")
    parser.add_argument("--original_data_file", type=str, required=True, help="Path to Step 0 processed data JSONL file.")
    parser.add_argument("--output_eval_file", type=str, required=True, help="Path to save the evaluation summary JSON file.")
    parser.add_argument("--output_scored_responses_dir", type=str, default=None, help="(Optional) Directory to save detailed scored responses.")
    parser.add_argument("--verifier_model_name", type=str, required=True, help="Verifier LLM Hugging Face ID.")
    parser.add_argument("--model_max_length", type=int, default=4096, help="Max sequence length for verifier.")
    parser.add_argument("--use_cache", action='store_true', default=True)
    parser.add_argument("--no_cache", action='store_false', dest='use_cache')
    parser.add_argument("--use_quantization", action='store_true', default=False)
    parser.add_argument("--max_context_len", type=int, default=10000)
    parser.add_argument("--pass_k", type=int, default=3, help="Value of K for Pass@K accuracy calculation.") # <-- New Arg

    args = parser.parse_args()

    # Validate response input source
    if not args.response_files and not args.responses_dir:
        parser.error("Either --response_files or --responses_dir must be specified.")
    if args.response_files and args.responses_dir:
        logger.warning("Both --response_files and --responses_dir specified. Using --response_files.")
        args.responses_dir = None # Prioritize specific files

    logger.info(f"--- Starting Logprob Filter Evaluation (Pass@{args.pass_k}) ---")
    logger.info(f"Verifier Model: {args.verifier_model_name}")
    # ... log other args ...

    original_data_map = {}
    original_data = load_json(args.original_data_file, load_by_line=True)
    if not original_data: sys.exit(f"Failed to load original data from {args.original_data_file}")
    for item in original_data:
        if "id" in item and "correct_answer_letter" in item:
            original_data_map[item["id"]] = item
    logger.info(f"Loaded {len(original_data_map)} original items with IDs and correct answers.")

    # --- Load Generated Responses ---
    if args.response_files:
        response_files_to_load = args.response_files
        logger.info(f"Loading specific response files: {response_files_to_load}")
    else:
        response_files_to_load = glob.glob(os.path.join(args.responses_dir, "*.jsonl"))
        logger.info(f"Loading response files from directory: {args.responses_dir}")

    if not response_files_to_load: sys.exit(f"No response files found.")

    all_responses = defaultdict(list)
    loaded_filenames = []
    for f_path in response_files_to_load:
        if not os.path.exists(f_path):
             logger.warning(f"Specified response file not found: {f_path}. Skipping.")
             continue
        fname = os.path.basename(f_path)
        loaded_filenames.append(fname)
        responses = load_json(f_path, load_by_line=True)
        if responses:
            logger.info(f"Loading {len(responses)} responses from {fname}")
            for resp in responses:
                if "id" in resp and "generated_text" in resp:
                    all_responses[resp["id"]].append(resp)
    logger.info(f"Loaded responses for {len(all_responses)} unique query IDs from {len(loaded_filenames)} files.")

    if not all_responses: sys.exit("No valid responses with IDs were loaded.")

    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tokenizer = load_verifier_model(args.verifier_model_name, device_type, args.use_quantization)
    if not model or not tokenizer: sys.exit("Model loading failed.")

    evaluation_details = []
    correctly_chosen_top1_count = 0
    correctly_chosen_topK_count = 0 # <-- New counter for Pass@K
    total_evaluated_queries = 0
    queries_with_scoring_errors = 0

    if args.output_scored_responses_dir: os.makedirs(args.output_scored_responses_dir, exist_ok=True)

    for query_id, responses_for_query in tqdm(all_responses.items(), desc="Evaluating Queries"):
        original_item = original_data_map.get(query_id)
        if not original_item: continue # Skip if no original data

        context = original_item.get("context", "")
        query = original_item.get("query", "")
        correct_letter = original_item.get("correct_answer_letter")
        if not context or not query or not correct_letter: continue # Skip if missing core info

        total_evaluated_queries += 1
        prefix = format_scoring_prefix(context, query, args.max_context_len)
        scored_query_responses = []
        valid_scored_responses = [] # Store only responses with valid scores

        for resp_item in responses_for_query:
            generated_text = resp_item.get("generated_text", "").strip()
            if not generated_text:
                resp_item[f"verifier_avg_nll"] = None
                scored_query_responses.append(resp_item)
                continue

            avg_nll = calculate_response_neg_log_likelihood(
                model, tokenizer, prefix, generated_text,
                args.use_cache, args.model_max_length
            )
            resp_item[f"verifier_avg_nll"] = avg_nll
            scored_query_responses.append(resp_item)

            if avg_nll is not None:
                valid_scored_responses.append(resp_item) # Add to list for ranking

        # --- Rank valid responses and evaluate ---
        chosen_letter_top1 = None
        is_correct_top1 = False
        is_correct_topK = False
        min_avg_nll = None
        top_k_letters = []

        if valid_scored_responses:
            # Sort by AvgNLL (lower is better)
            valid_scored_responses.sort(key=lambda x: x.get("verifier_avg_nll", float('inf')))

            # --- Top-1 Evaluation ---
            best_response_item = valid_scored_responses[0]
            min_avg_nll = best_response_item.get("verifier_avg_nll")
            chosen_text_top1 = best_response_item.get("generated_text", "")
            chosen_letter_top1 = extract_answer_letter(chosen_text_top1)
            if chosen_letter_top1 is not None and chosen_letter_top1 == correct_letter:
                is_correct_top1 = True
                correctly_chosen_top1_count += 1

            # --- Pass@K Evaluation ---
            top_k_items = valid_scored_responses[:args.pass_k]
            top_k_letters = [extract_answer_letter(item.get("generated_text", "")) for item in top_k_items]
            top_k_letters = [letter for letter in top_k_letters if letter is not None] # Filter out None values
            if correct_letter in top_k_letters:
                is_correct_topK = True
                correctly_chosen_topK_count += 1

        else:
            queries_with_scoring_errors += 1
            best_response_item = None # No valid scores

        eval_item = {
            "id": query_id,
            "correct_answer_letter": correct_letter,
            "best_response_by_logprob": best_response_item,
            "chosen_letter_by_logprob": chosen_letter_top1,
            "is_correct_top1": is_correct_top1,
            "is_correct_topK": is_correct_topK, # <-- Store Pass@K result
            "top_k_letters": top_k_letters,     # <-- Store Top K letters
            "min_avg_nll": min_avg_nll
        }
        evaluation_details.append(eval_item)

        if args.output_scored_responses_dir:
             detail_file = os.path.join(args.output_scored_responses_dir, f"{query_id}_scored.jsonl")
             save_json(scored_query_responses, detail_file, save_by_line=True)

    # --- Calculate Final Accuracies ---
    final_accuracy_top1 = (correctly_chosen_top1_count / total_evaluated_queries) * 100 if total_evaluated_queries > 0 else 0
    final_accuracy_topK = (correctly_chosen_topK_count / total_evaluated_queries) * 100 if total_evaluated_queries > 0 else 0

    logger.info(f"Evaluation Complete for Verifier: {args.verifier_model_name}")
    logger.info(f"Total Queries Evaluated: {total_evaluated_queries}")
    logger.info(f"Queries where Logprob chose Correct Answer (Top-1): {correctly_chosen_top1_count}")
    logger.info(f"Top-1 Accuracy: {final_accuracy_top1:.2f}%")
    logger.info(f"Queries where Correct Answer was in Top-{args.pass_k}: {correctly_chosen_topK_count}") # <-- Log Pass@K
    logger.info(f"Pass@{args.pass_k} Accuracy: {final_accuracy_topK:.2f}%") # <-- Log Pass@K
    if queries_with_scoring_errors > 0:
         logger.warning(f"Could not determine best response for {queries_with_scoring_errors} queries due to scoring errors.")

    # --- Save Summary ---
    summary_output = {
        "verifier_model": args.verifier_model_name,
        "quantized": args.use_quantization,
        "pass_k_value": args.pass_k,
        "total_queries_processed": total_evaluated_queries,
        "correctly_chosen_by_logprob_top1": correctly_chosen_top1_count,
        "accuracy_percentage_top1": final_accuracy_top1,
        "correctly_chosen_by_logprob_topK": correctly_chosen_topK_count, # <-- Save Pass@K
        "accuracy_percentage_topK": final_accuracy_topK, # <-- Save Pass@K
        "queries_with_scoring_errors": queries_with_scoring_errors,
        "response_files_used": sorted(loaded_filenames),
        "original_data_file": args.original_data_file,
        "evaluation_details": evaluation_details
    }
    save_json(summary_output, args.output_eval_file, save_by_line=False, indent=2)
    logger.info(f"Evaluation summary saved to: {args.output_eval_file}")
    logger.info(f"--- Logprob Filter Evaluation Finished ---")

if __name__ == "__main__":
    main()
