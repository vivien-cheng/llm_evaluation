import argparse
import json
import os
import sys
import torch
import logging
import math
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Dict, List, Any, Optional, Tuple

# --- Path Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
try:
    from hta_evaluation_harness.utils import load_json, save_json
except ImportError:
    print("Warning: Could not import utils from hta_evaluation_harness. Using basic local file handling.")
    # Basic local file handling functions
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

# --- Prompt Formatting ---
def format_prompt_prefix(context: str, query: str, max_context_len: int = 10000) -> str:
    """Formats the prefix part (context + query)."""
    if len(context) > max_context_len:
        truncated_context = context[:max_context_len] + "\n... [Context Truncated] ..."
    else:
        truncated_context = context
    # Simple prompt structure
    prompt = f"Context:\n{truncated_context}\n\nQuery: {query}\nAnswer:"
    return prompt

# --- Core Logic: Calculate AvgNLL for a specific Option ---
@torch.no_grad()
def calculate_option_neg_log_likelihood(
    model, tokenizer, prefix_text: str, option_text: str,
    use_cache: bool = True, max_length: int = 4096
    ) -> Optional[float]:
    """ Calculates AvgNLL for a predefined option text, given the prefix. """
    try:
        model_device = model.device
        # Add leading space to option text, common practice
        option_text_proc = " " + option_text.strip()

        prefix_inputs = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False)
        option_inputs = tokenizer(option_text_proc, return_tensors="pt", add_special_tokens=False)

        prefix_ids = prefix_inputs.input_ids.to(model_device)
        option_ids = option_inputs.input_ids.to(model_device)

        if getattr(tokenizer, 'add_bos_token', False) and tokenizer.bos_token_id is not None:
             if prefix_ids.shape[1] == 0 or prefix_ids[0, 0] != tokenizer.bos_token_id:
                  bos_tensor = torch.tensor([[tokenizer.bos_token_id]], device=model_device)
                  prefix_ids = torch.cat([bos_tensor, prefix_ids], dim=1)

        combined_ids = torch.cat([prefix_ids, option_ids], dim=1)

        if combined_ids.shape[1] > max_length:
            logger.warning(f"Combined len {combined_ids.shape[1]} > max {max_length}. Skipping scoring for option '{option_text[:30]}...'.")
            return None

        prefix_len = prefix_ids.shape[1]
        labels = combined_ids.clone()
        labels[:, :prefix_len] = -100 # Ignore prefix tokens for loss calculation
        labels = labels.to(model_device)

        loss = None
        if use_cache:
            outputs_prefix = model(input_ids=prefix_ids, use_cache=True)
            past_key_values = outputs_prefix.past_key_values
            # Score only the option tokens using the cache
            outputs_option = model(input_ids=option_ids, past_key_values=past_key_values, labels=labels[:, prefix_len:])
            loss = outputs_option.loss
        else:
            outputs_full = model(input_ids=combined_ids, labels=labels)
            loss = outputs_full.loss

        if loss is None or torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"Invalid loss ({loss}) for option '{option_text[:30]}...'. Skipping.")
            return float('inf') # Assign infinite loss for ranking purposes

        return loss.item()
    except Exception as e:
        logger.error(f"Error calculating NLL for option '{option_text[:50]}...': {e}", exc_info=True)
        return float('inf') # Assign infinite loss on error

# --- Main Processing Logic ---
def main():
    parser = argparse.ArgumentParser(description="Rank MCQ options using verifier LLM log probabilities.")
    parser.add_argument("--input_data_file", type=str, required=True, help="Path to the processed input JSONL file (e.g., longhealth_cleaned.jsonl).")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the ranked options JSONL file.")
    parser.add_argument("--verifier_model_name", type=str, required=True, help="Verifier LLM Hugging Face ID.")
    parser.add_argument("--model_max_length", type=int, default=4096, help="Max sequence length for verifier.")
    parser.add_argument("--use_cache", action='store_true', default=True)
    parser.add_argument("--no_cache", action='store_false', dest='use_cache')
    parser.add_argument("--use_quantization", action='store_true', default=False)
    parser.add_argument("--max_context_len", type=int, default=10000)
    args = parser.parse_args()

    logger.info(f"--- Starting MCQ Option Ranking using Log Probs ---")
    logger.info(f"Input Data: {args.input_data_file}")
    logger.info(f"Output File: {args.output_file}")
    logger.info(f"Verifier Model (Local): {args.verifier_model_name}")
    # ... log other args ...

    # --- Load Data ---
    data = load_json(args.input_data_file, load_by_line=True)
    if not data: sys.exit(f"Failed to load or empty data from {args.input_data_file}")

    # --- Load Verifier Model ---
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tokenizer = load_verifier_model(args.verifier_model_name, device_type, args.use_quantization)
    if not model or not tokenizer: sys.exit("Model loading failed.")

    # --- Process Each Item ---
    results = []
    for item in tqdm(data, desc="Ranking Options"):
        query_id = item.get("id")
        context = item.get("context", "")
        query = item.get("query", "")
        options = item.get("options", {}) # Should be {"A": "Text A", "B": "Text B", ...}
        correct_letter = item.get("correct_answer_letter")

        if not query_id or not context or not query or not options or not correct_letter:
            logger.warning(f"Skipping item {query_id or 'UNKNOWN'} due to missing fields.")
            continue

        logger.info(f"Processing query {query_id} using {args.verifier_model_name}...")
        prefix = format_prompt_prefix(context, query, args.max_context_len)
        option_scores = []

        for letter, text in options.items():
            if not text:
                logger.warning(f"Skipping empty option '{letter}' for query {query_id}.")
                continue

            # Format the option text to include the letter, similar to how a model might generate it
            formatted_option_text = f"{letter}: {text}"

            avg_nll = calculate_option_neg_log_likelihood(
                model, tokenizer, prefix, formatted_option_text,
                args.use_cache, args.model_max_length
            )
            # Use infinity for failed calculations to ensure they rank last
            score = avg_nll if avg_nll is not None else float('inf')
            option_scores.append({"letter": letter, "text": text, "avg_nll": score})

        # Sort options by score (lowest NLL first)
        option_scores.sort(key=lambda x: x["avg_nll"])

        # Log the best option found
        if option_scores and option_scores[0]["avg_nll"] != float('inf'):
            best_option = option_scores[0]
            logger.info(f"Finished query {query_id}. Best option: {best_option['letter']} (AvgNLL: {best_option['avg_nll']:.4f})")
        else:
             logger.warning(f"Finished query {query_id}. Could not score any options reliably.")

        # Store result for this query
        result_item = {
            "id": query_id,
            "correct_answer_letter": correct_letter,
            "ranked_options": option_scores # List of {"letter": L, "text": T, "avg_nll": S} sorted by score
        }
        results.append(result_item)

    # --- Save Results ---
    logger.info(f"Saving {len(results)} ranking results to {args.output_file}")
    save_json(results, args.output_file, save_by_line=True)
    logger.info(f"--- Option Ranking Finished ---")

if __name__ == "__main__":
    main()
