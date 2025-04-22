import argparse
import json
import os
import sys
import torch
import logging
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from typing import Dict, List, Any, Optional

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

# --- Model Loading Cache ---
MODEL_CACHE = {}

def load_model(model_name_or_path, device_type_str):
    """Loads the generation model and tokenizer."""
    cache_key = f"{model_name_or_path}_{device_type_str}"
    if cache_key in MODEL_CACHE:
        logger.info(f"Using cached model/tokenizer for {model_name_or_path}")
        return MODEL_CACHE[cache_key]

    logger.info(f"Loading model: {model_name_or_path}...")
    try:
        # Determine dtype
        dtype = torch.float32
        if device_type_str == 'cuda' and torch.cuda.is_bf16_supported():
             dtype = torch.bfloat16
        logger.info(f"Using dtype: {dtype}")

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True # Needed for some models like Qwen/Phi
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )

        # Set padding token if missing
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                 logger.warning("pad_token not set, using eos_token as pad_token.")
                 tokenizer.pad_token = tokenizer.eos_token
                 tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                 logger.warning("Adding new pad token '<|pad|>'.")
                 tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
                 model.resize_token_embeddings(len(tokenizer)) # Important!

        model.eval()
        MODEL_CACHE[cache_key] = (model, tokenizer)
        try: device_map_str = str(model.hf_device_map)
        except Exception: device_map_str = "N/A"
        logger.info(f"Model {model_name_or_path} loaded successfully using {dtype} onto devices: {device_map_str}")
        return model, tokenizer

    except Exception as e:
        logger.exception(f"FATAL ERROR loading model {model_name_or_path}: {e}", exc_info=True)
        logger.error("Check model name, internet connection, access permissions, dependencies, VRAM.")
        sys.exit(1)

def format_generation_prompt(context: str, query: str, options: Dict[str, str], max_input_tokens: int, tokenizer) -> Optional[str]:
    """Formats the prompt for the generation model, truncating context if needed."""
    option_str = "\n".join([f"{key}: {value}" for key, value in options.items()])
    base_prompt = f"""Context:
{context}

Query: {query}

Options:
{option_str}

Based *only* on the context provided, select the best option letter and provide the corresponding text. Your answer should start with the letter followed by a colon and the text (e.g., 'A: Text for option A').

Answer: """

    try:
        full_tokens = tokenizer.encode(base_prompt, add_special_tokens=False)

        if len(full_tokens) > max_input_tokens:
            logger.warning(f"Input prompt token length ({len(full_tokens)}) exceeds max_input_tokens ({max_input_tokens}). Truncating context.")
            non_context_prompt = f"""Context:\n\nQuery: {query}\n\nOptions:\n{option_str}\n\nAnswer: """
            non_context_tokens = tokenizer.encode(non_context_prompt, add_special_tokens=False)
            allowed_context_tokens = max_input_tokens - len(non_context_tokens) - 10 # Subtract buffer

            if allowed_context_tokens <= 0:
                 logger.error("Query and options alone exceed max_input_tokens. Cannot generate prompt.")
                 return None

            context_tokens = tokenizer.encode(context, add_special_tokens=False)
            truncated_context_tokens = context_tokens[:allowed_context_tokens]
            truncated_context = tokenizer.decode(truncated_context_tokens, skip_special_tokens=True)
            logger.info(f"Context truncated to approx {allowed_context_tokens} tokens.")

            final_prompt = f"""Context:
{truncated_context}

Query: {query}

Options:
{option_str}

Based *only* on the context provided, select the best option letter and provide the corresponding text. Your answer should start with the letter followed by a colon and the text (e.g., 'A: Text for option A').

Answer: """
        else:
            final_prompt = base_prompt

        return final_prompt

    except Exception as e:
        logger.error(f"Error during prompt formatting/tokenization: {e}", exc_info=True)
        return None


@torch.no_grad()
def generate_responses(model, tokenizer, prompt: str, num_responses: int, generation_config: GenerationConfig) -> List[str]:
    """Generates responses using the loaded model."""
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(model.device)
    outputs = model.generate(
        **inputs,
        generation_config=generation_config,
        num_return_sequences=num_responses,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    input_length = inputs.input_ids.shape[1]
    generated_texts = [
        tokenizer.decode(output[input_length:], skip_special_tokens=True).strip()
        for output in outputs
    ]
    return generated_texts

def main():
    parser = argparse.ArgumentParser(description="Generate responses from LLMs for LongHealth benchmark.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Hugging Face model identifier.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the processed input JSONL file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the generated responses JSONL file.")
    parser.add_argument("--num_responses", type=int, default=1, help="Number of responses to generate per query.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Max new tokens to generate.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus sampling.")
    # --- FIX: Added the missing argument definition ---
    parser.add_argument("--max_input_tokens", type=int, default=32000, help="Maximum tokens allowed for the input prompt (context+query+options).")
    # --- End Fix ---
    args = parser.parse_args()

    logger.info("--- Starting Actual Response Generation ---")
    logger.info(f"Model: {args.model_name_or_path}")
    logger.info(f"Input: {args.input_file}")
    logger.info(f"Output: {args.output_file}")
    logger.info(f"Responses per query: {args.num_responses}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Max New Tokens: {args.max_new_tokens}")
    logger.info(f"Top P: {args.top_p}")
    logger.info(f"Max Input Tokens: {args.max_input_tokens}") # Log the value being used


    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device_type == 'cuda': logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}"); torch.cuda.empty_cache()
    else: logger.info("CUDA not available. Using CPU.")

    model, tokenizer = load_model(args.model_name_or_path, device_type)
    data = load_json(args.input_file, load_by_line=True)
    if data is None: sys.exit(f"Failed to load data from {args.input_file}")

    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature if args.temperature > 0 else None,
        top_p=args.top_p if args.temperature > 0 else None,
        do_sample=True if args.temperature > 0 else False,
        num_return_sequences=args.num_responses,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen_params_dict = {
        "temperature": args.temperature, "max_new_tokens": args.max_new_tokens, "top_p": args.top_p,
    }

    results = []
    valid_responses_count = 0
    total_queries = len(data)

    for item in tqdm(data, desc=f"Generating with {os.path.basename(args.model_name_or_path)}"):
        query_id = item.get("id") # Ensure ID is retrieved
        context = item.get("context", "")
        query = item.get("query", "")
        options = item.get("options", {})

        if not query_id or not context or not query or not options:
            logger.warning(f"Skipping item due to missing fields: {item.get('id', 'N/A')}")
            continue

        prompt = format_generation_prompt(context, query, options, args.max_input_tokens, tokenizer)
        if prompt is None:
            logger.warning(f"Skipping item {query_id} due to prompt formatting error.")
            continue

        try:
            generated_texts = generate_responses(model, tokenizer, prompt, args.num_responses, generation_config)
            for i, text in enumerate(generated_texts):
                result_item = {
                    "id": query_id, # Included ID here
                    "generated_text": text,
                    "model_name": args.model_name_or_path,
                    "generation_params": {**gen_params_dict, "index": i}
                }
                results.append(result_item)
                valid_responses_count += 1
        except Exception as e:
            logger.error(f"Error generating response for query ID {query_id}: {e}", exc_info=True)

    logger.info("\n--- Generation Summary ---")
    logger.info(f"Generated and saved {valid_responses_count} valid responses for {total_queries} queries.")
    save_json(results, args.output_file, save_by_line=True)
    logger.info(f"Responses saved to {args.output_file}")
    logger.info("Response generation finished.")

    # Cleanup
    model_key = f"{args.model_name_or_path}_{device_type}"
    if model_key in MODEL_CACHE: del MODEL_CACHE[model_key]
    model = None; tokenizer = None
    if device_type == 'cuda': torch.cuda.empty_cache()
    logger.info("Cleaned up model resources.")

if __name__ == "__main__":
    main()
