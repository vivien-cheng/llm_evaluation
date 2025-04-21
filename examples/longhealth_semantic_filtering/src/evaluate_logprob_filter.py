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

# --- Model Loading (Re-used from previous script) ---
VERIFIER_MODEL_CACHE = {}
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
        # Add leading space to response text, common practice
        response_inputs = tokenizer(" " + generated_response_text.strip(), return_tensors="pt", add_special_tokens=False)

        prefix_ids = prefix_inputs.input_ids.to(model_device)
        response_ids = response_inputs.input_ids.to(model_device)

        # Handle BOS only if tokenizer expects it AND it's not already there
        if getattr(tokenizer, 'add_bos_token', False) and tokenizer.bos_token_id is not None:
             if prefix_ids.shape[1] == 0 or prefix_ids[0, 0] != tokenizer.bos_token_id:
                  bos_tensor = torch.tensor([[tokenizer.bos_token_id]], device=model_device)
                  prefix_ids = torch.cat([bos_tensor, prefix_ids], dim=1)

        combined_ids = torch.cat([prefix_ids, response_ids], dim=1)

        if combined_ids.shape[1] > max_length:
            logger.warning(f"Combined len {combined_ids.shape[1]} > max {max_length}. Skipping scoring.")
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

# --- Helper to Extract Answer Letter ---
def extract_answer_letter(text: str) -> Optional[str]:
    """Tries to extract A, B, C, D, or E from the beginning of the text."""
    if not text:
        return None
    # Match variations like "A:", "A.", "A)", "A " or just "A" at the start
    match = re.match(r"^\s*([A-E])[\s:.)]*", text.strip(), re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None # Could not extract a letter

# --- Main Evaluation Logic ---
def main():
    parser = argparse.ArgumentParser(description="Evaluate logprob filtering by scoring generated responses.")
    parser.add_argument("--responses_dir", type=str, required=True, help="Directory containing Step 1 response JSONL files.")
    parser.add_argument("--original_data_file", type=str, required=True, help="Path to Step 0 processed data JSONL file.")
    parser.add_argument("--output_eval_file", type=str, required=True, help="Path to save the evaluation summary JSON file.")
    parser.add_argument("--output_scored_responses_dir", type=str, default=None, help="(Optional) Directory to save detailed scored responses.")
    parser.add_argument("--verifier_model_name", type=str, required=True, help="Verifier LLM Hugging Face ID.")
    parser.add_argument("--model_max_length", type=int, default=4096, help="Max sequence length for verifier.")
    parser.add_argument("--use_cache", action='store_true', default=True)
    parser.add_argument("--no_cache", action='store_false', dest='use_cache')
    parser.add_argument("--use_quantization", action='store_true', default=False)
    parser.add_argument("--max_context_len", type=int, default=10000)
    args = parser.parse_args()

    logger.info(f"--- Starting Logprob Filter Evaluation ---")
    logger.info(f"Verifier Model: {args.verifier_model_name}")
    # ... log other args ...

    # --- Load Original Data ---
    original_data_map = {}
    original_data = load_json(args.original_data_file, load_by_line=True)
    if not original_data: sys.exit(f"Failed to load original data from {args.original_data_file}")
    for item in original_data:
        if "id" in item and "correct_answer_letter" in item:
            original_data_map[item["id"]] = item
    logger.info(f"Loaded {len(original_data_map)} original items with IDs and correct answers.")

    # --- Load Generated Responses ---
    response_files = glob.glob(os.path.join(args.responses_dir, "*.jsonl"))
    if not response_files: sys.exit(f"No response files (*.jsonl) found in {args.responses_dir}")

    all_responses = defaultdict(list)
    for f_path in response_files:
        responses = load_json(f_path, load_by_line=True)
        if responses:
            logger.info(f"Loading {len(responses)} responses from {os.path.basename(f_path)}")
            for resp in responses:
                if "id" in resp and "generated_text" in resp:
                    all_responses[resp["id"]].append(resp)
    logger.info(f"Loaded responses for {len(all_responses)} unique query IDs.")

    # --- Load Verifier Model ---
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tokenizer = load_verifier_model(args.verifier_model_name, device_type, args.use_quantization)
    if not model or not tokenizer: sys.exit("Model loading failed.")

    # --- Score Responses and Evaluate ---
    evaluation_details = []
    correctly_chosen_count = 0
    total_evaluated_queries = 0
    queries_with_scoring_errors = 0

    if args.output_scored_responses_dir:
        os.makedirs(args.output_scored_responses_dir, exist_ok=True)

    for query_id, responses in tqdm(all_responses.items(), desc="Evaluating Queries"):
        original_item = original_data_map.get(query_id)
        if not original_item:
            logger.warning(f"Skipping query ID '{query_id}': Not found in original data.")
            continue

        context = original_item.get("context", "")
        query = original_item.get("query", "")
        correct_letter = original_item.get("correct_answer_letter")

        if not context or not query or not correct_letter:
            logger.warning(f"Skipping query ID '{query_id}': Missing context, query, or correct answer in original data.")
            continue

        total_evaluated_queries += 1
        prefix = format_scoring_prefix(context, query, args.max_context_len)
        scored_query_responses = []
        min_nll = float('inf')
        best_response_item = None
        all_scores_failed = True

        for resp_item in responses:
            generated_text = resp_item.get("generated_text", "").strip()
            if not generated_text:
                logger.warning(f"Skipping empty generated text for query '{query_id}' from model {resp_item.get('model_name')}")
                continue

            avg_nll = calculate_response_neg_log_likelihood(
                model, tokenizer, prefix, generated_text,
                args.use_cache, args.model_max_length
            )
            resp_item[f"verifier_avg_nll"] = avg_nll # Add score
            scored_query_responses.append(resp_item) # Keep track even if score is None

            if avg_nll is not None:
                all_scores_failed = False
                if avg_nll < min_nll:
                    min_nll = avg_nll
                    best_response_item = resp_item

        # --- Evaluate the chosen response ---
        chosen_letter = None
        if best_response_item:
            chosen_text = best_response_item.get("generated_text", "")
            chosen_letter = extract_answer_letter(chosen_text)
            is_correct = (chosen_letter is not None and chosen_letter == correct_letter)
            if is_correct:
                correctly_chosen_count += 1
        else:
            is_correct = False # Cannot be correct if no response was chosen
            if all_scores_failed:
                 queries_with_scoring_errors += 1


        # Store details for output
        eval_item = {
            "id": query_id,
            "correct_answer_letter": correct_letter,
            "best_response_by_logprob": best_response_item, # Includes text, original model, and score
            "chosen_letter_by_logprob": chosen_letter,
            "is_correct": is_correct,
            "min_avg_nll": min_nll if min_nll != float('inf') else None
        }
        evaluation_details.append(eval_item)

        # Optionally save detailed scored responses for this query
        if args.output_scored_responses_dir:
             detail_file = os.path.join(args.output_scored_responses_dir, f"{query_id}_scored.jsonl")
             save_json(scored_query_responses, detail_file, save_by_line=True)


    # --- Calculate Final Accuracy ---
    final_accuracy = (correctly_chosen_count / total_evaluated_queries) * 100 if total_evaluated_queries > 0 else 0

    logger.info(f"Evaluation Complete for Verifier: {args.verifier_model_name}")
    logger.info(f"Total Queries Evaluated: {total_evaluated_queries}")
    logger.info(f"Queries where Logprob chose Correct Answer: {correctly_chosen_count}")
    logger.info(f"Accuracy: {final_accuracy:.2f}%")
    if queries_with_scoring_errors > 0:
         logger.warning(f"Could not determine best response for {queries_with_scoring_errors} queries due to scoring errors (NaN/Inf/None).")

    # --- Save Summary ---
    summary_output = {
        "verifier_model": args.verifier_model_name,
        "quantized": args.use_quantization,
        "total_queries_processed": total_evaluated_queries,
        "correctly_chosen_by_logprob": correctly_chosen_count,
        "accuracy_percentage": final_accuracy,
        "queries_with_scoring_errors": queries_with_scoring_errors,
        "response_files_used": [os.path.basename(f) for f in response_files],
        "original_data_file": args.original_data_file,
        "evaluation_details": evaluation_details # Include detailed results
    }
    save_json(summary_output, args.output_eval_file, save_by_line=False, indent=2)
    logger.info(f"Evaluation summary saved to: {args.output_eval_file}")

    logger.info(f"--- Logprob Filter Evaluation Finished ---")

if __name__ == "__main__":
    main()
