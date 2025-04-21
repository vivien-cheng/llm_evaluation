import argparse
import json
import os
import sys
import torch
import logging
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Dict, List, Any, Optional, Tuple
from dotenv import load_dotenv # <-- Import dotenv
from openai import OpenAI, APIError, RateLimitError # <-- Import OpenAI client and errors

# --- Load Environment Variables ---
load_dotenv() # <-- Load .env file variables

# --- Path Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
try:
    from hta_evaluation_harness.utils import load_json, save_json
except ImportError:
     print("Warning: Could not import utils from hta_evaluation_harness. Using basic local file handling.")
     def load_json(file_path, load_by_line=False):
         if not os.path.exists(file_path): return None
         with open(file_path, 'r', encoding='utf-8') as f:
             if load_by_line:
                 data = []
                 for line in f:
                      if line.strip():
                           try:
                                data.append(json.loads(line))
                           except json.JSONDecodeError:
                                print(f"Warning: Skipping invalid JSON line: {line[:100]}")
                 return data
             else:
                 return json.load(f)

     def save_json(data, file_path, save_by_line=False, indent=2):
         os.makedirs(os.path.dirname(file_path), exist_ok=True)
         with open(file_path, 'w', encoding='utf-8') as f:
             if save_by_line:
                 for item in data:
                     f.write(json.dumps(item, ensure_ascii=False) + '\n')
             else:
                 json.dump(data, f, ensure_ascii=False, indent=indent)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Initialize Fireworks Client ---
fireworks_client = None
fireworks_api_key = os.getenv("FIREWORKS_API_KEY")
if fireworks_api_key:
    try:
        fireworks_client = OpenAI(
            base_url="https://api.fireworks.ai/inference/v1", # Standard Fireworks OpenAI-compatible endpoint
            api_key=fireworks_api_key,
        )
        logger.info("Fireworks AI client initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Fireworks AI client: {e}")
else:
    logger.info("FIREWORKS_API_KEY not found in environment. Fireworks API calls will not be possible.")


# --- Local Model Loading ---
VERIFIER_MODEL_CACHE = {}
# (load_verifier_model function remains the same as the fixed version from previous steps)
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
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            logger.info("Using 4-bit quantization.")

        if device_type_str == 'cpu':
             dtype = torch.float32
        elif torch.cuda.is_bf16_supported() and not use_quantization:
             dtype = torch.bfloat16
        else:
             dtype = torch.float32

        logger.info(f"Attempting to load with dtype: {dtype}")

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=quantization_config
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )

        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                 logger.warning("pad_token not set, using eos_token as pad_token.")
                 tokenizer.pad_token = tokenizer.eos_token
                 tokenizer.pad_token_id = tokenizer.eos_token_id # Ensure ID is also set
            else:
                 logger.warning("pad_token and eos_token not set. Adding new pad token '<|pad|>'.")
                 tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
                 model.resize_token_embeddings(len(tokenizer))

        model.eval()
        VERIFIER_MODEL_CACHE[cache_key] = (model, tokenizer)
        try:
            device_map_str = str(model.hf_device_map)
        except Exception:
            device_map_str = "N/A (Could not get device map)"
        logger.info(f"Verifier model {model_name_or_path} loaded successfully. Device map: {device_map_str}")

        return model, tokenizer

    except Exception as e:
        logger.exception(f"FATAL ERROR loading verifier model {model_name_or_path}: {e}", exc_info=True)
        logger.error("Check model name, internet connection, access permissions (login/gating), dependencies, VRAM.")
        sys.exit(1)


# --- Prompt Formatting ---
# (format_prefix_prompt function remains the same)
def format_prefix_prompt(context: str, query: str, max_context_len: int = 10000) -> str:
    """Formats the prefix part (context + query + instruction)."""
    if len(context) > max_context_len:
        truncated_context = context[:max_context_len] + "\n... [Context Truncated] ..."
        logger.debug(f"Context truncated to {max_context_len} characters for prompt.")
    else:
        truncated_context = context

    prompt = f"""Context:
{truncated_context}

Query: {query}

Based *only* on the context provided, the most likely answer is:"""
    return prompt


# --- LOCAL Calculation Logic ---
# (calculate_option_neg_log_likelihood function remains the same as the fixed version)
@torch.no_grad()
def calculate_option_neg_log_likelihood(
    model,
    tokenizer,
    prefix_text: str,
    option_text: str,
    use_cache: bool = True,
    max_length: int = 4096
    ) -> Optional[float]:
    """ Calculates AvgNLL for local models, handling device placement. """
    try:
        model_device = model.device
        logger.debug(f"  Target device for local tensors: {model_device}")

        # Tokenize with space before option
        prefix_inputs = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False)
        option_inputs = tokenizer(" " + option_text, return_tensors="pt", add_special_tokens=False) # Add leading space

        prefix_ids = prefix_inputs.input_ids.to(model_device)
        option_ids = option_inputs.input_ids.to(model_device)

        if getattr(tokenizer, 'add_bos_token', False) and tokenizer.bos_token_id is not None:
             if prefix_ids.shape[1] == 0 or prefix_ids[0, 0] != tokenizer.bos_token_id:
                  logger.debug("Adding BOS token to prefix_ids.")
                  bos_tensor = torch.tensor([[tokenizer.bos_token_id]], device=model_device)
                  prefix_ids = torch.cat([bos_tensor, prefix_ids], dim=1)

        combined_ids = torch.cat([prefix_ids, option_ids], dim=1)

        if combined_ids.shape[1] > max_length:
            logger.warning(f"Combined sequence length ({combined_ids.shape[1]}) exceeds max_length ({max_length}). Skipping option '{option_text[:30]}...'.")
            return None

        prefix_len = prefix_ids.shape[1]
        labels = combined_ids.clone()
        labels[:, :prefix_len] = -100
        labels = labels.to(model_device)

        loss = None
        if use_cache:
            logger.debug(f"  Processing prefix (length {prefix_ids.shape[1]}) with use_cache=True")
            outputs_prefix = model(input_ids=prefix_ids, use_cache=True)
            past_key_values = outputs_prefix.past_key_values
            logger.debug("  Got past_key_values from prefix.")

            logger.debug(f"  Processing option (length {option_ids.shape[1]}) using cache and labels")
            outputs_option = model(
                input_ids=option_ids,
                past_key_values=past_key_values,
                labels=labels[:, prefix_len:]
            )
            loss = outputs_option.loss
            logger.debug(f"  Loss calculated using cache: {loss.item() if loss is not None else 'N/A'}")

        else:
            logger.debug(f"  Processing combined sequence (length {combined_ids.shape[1]}) with use_cache=False")
            outputs_full = model(input_ids=combined_ids, labels=labels)
            loss = outputs_full.loss
            logger.debug(f"  Loss calculated without cache: {loss.item() if loss is not None else 'N/A'}")

        if loss is None:
             logger.warning(f"  Model returned None for loss for option '{option_text[:30]}...'")
             return None

        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"  Loss is NaN or Inf ({loss.item()}) for option '{option_text[:30]}...'. Skipping.")
            return None

        return loss.item()

    except Exception as e:
        logger.error(f"Error in calculate_option_neg_log_likelihood for option '{option_text[:50]}...': {e}", exc_info=True)
        return None

# --- NEW: API Calculation Logic ---
def calculate_option_neg_log_likelihood_api(
    api_client: OpenAI, # Pass the initialized client
    api_model_id: str,
    prefix_text: str,
    option_text: str,
    # Need tokenizer to count prefix/option tokens for parsing response
    local_tokenizer_for_parsing: AutoTokenizer
    ) -> Optional[float]:
    """
    Calculates the average negative log-likelihood (AvgNLL) using an
    OpenAI-compatible API (like Fireworks AI) that supports logprobs.

    NOTE: This relies on the API returning logprobs in a specific format
          similar to OpenAI's 'echo=True' and 'logprobs' parameter.
          VERIFY against Fireworks AI documentation.
    """
    if not api_client:
        logger.error("API client not initialized.")
        return None

    try:
        # We need the full prompt echoed back with logprobs to isolate the option part.
        # Add a space before the option text for consistency.
        full_prompt = prefix_text + " " + option_text

        logger.debug(f"  Sending API request to {api_model_id} for option '{option_text[:30]}...'")

        # Make the API call requesting logprobs and echoing the prompt
        # Setting max_tokens=1 prevents generation beyond the prompt
        response = api_client.completions.create(
            model=api_model_id,
            prompt=full_prompt,
            max_tokens=1,
            logprobs=1, # Request top 1 logprob for each token position
            echo=True   # Echo the prompt back in the response
        )

        # --- PARSE THE RESPONSE ---
        # This parsing assumes OpenAI's response structure. Verify for Fireworks.
        if not response.choices or not response.choices[0].logprobs:
            logger.error(f"API response for {api_model_id} missing choices or logprobs.")
            return None

        logprobs_data = response.choices[0].logprobs

        # We need tokens and their corresponding logprobs
        if not logprobs_data.tokens or not logprobs_data.token_logprobs:
             logger.error(f"API response for {api_model_id} missing tokens or token_logprobs in logprobs data.")
             return None

        response_tokens = logprobs_data.tokens
        response_token_logprobs = logprobs_data.token_logprobs

        # Find where the option starts in the echoed tokens
        # Use the local tokenizer passed in for this comparison
        # Tokenize prefix separately (without space) to find its length
        prefix_tokens_local = local_tokenizer_for_parsing(prefix_text, add_special_tokens=False).input_ids
        # Assume API might add BOS, but prefix length should align if echo works well
        # Adjust start_index based on how API echo handles BOS vs local tokenizer
        # Heuristic: Check if response starts with BOS and local doesn't.
        api_adds_bos = False # Default assumption
        if local_tokenizer_for_parsing.bos_token_id is not None and len(response_tokens)>0 :
            if response_tokens[0] == local_tokenizer_for_parsing.decode(local_tokenizer_for_parsing.bos_token_id): # Comparing decoded strings
                 api_adds_bos = True

        # Calculate effective prefix length in API response tokens
        prefix_len_in_api_tokens = len(prefix_tokens_local)
        if api_adds_bos and (len(prefix_tokens_local) == 0 or prefix_tokens_local[0] != local_tokenizer_for_parsing.bos_token_id):
            prefix_len_in_api_tokens += 1 # Account for BOS added by API but not in our local count
        # This alignment might need refinement based on actual API response!


        # Tokenize option text (with leading space) locally to get expected length
        option_tokens_local = local_tokenizer_for_parsing(" " + option_text, add_special_tokens=False).input_ids
        option_len = len(option_tokens_local)

        # Define start and end indices in the API response's logprobs list
        start_index = prefix_len_in_api_tokens
        end_index = start_index + option_len

        logger.debug(f"  Attempting to extract logprobs from index {start_index} to {end_index} (inclusive end for slicing python list).")
        logger.debug(f"  Total response tokens: {len(response_tokens)}, logprobs: {len(response_token_logprobs)}")

        # Validate indices and list lengths
        if start_index < 0 or end_index > len(response_token_logprobs):
            logger.error(f"Calculated indices [{start_index}:{end_index}] are out of bounds for API logprobs list (length {len(response_token_logprobs)}). Token alignment likely failed.")
            # Log details for debugging alignment:
            logger.debug(f"  Local Prefix Tokens ({len(prefix_tokens_local)}): {prefix_tokens_local[:10]}...")
            logger.debug(f"  Local Option Tokens ({len(option_tokens_local)}): {option_tokens_local[:10]}...")
            logger.debug(f"  API Response Tokens ({len(response_tokens)}): {response_tokens[:20]}...")
            return None

        # Extract the logprobs corresponding to the option tokens
        option_logprobs = response_token_logprobs[start_index:end_index]

        if len(option_logprobs) != option_len:
            logger.error(f"Extracted {len(option_logprobs)} logprobs, but expected {option_len} for the option. Token alignment likely failed.")
            # Log details for debugging alignment:
            logger.debug(f"  Local Prefix Tokens ({len(prefix_tokens_local)}): {prefix_tokens_local[:10]}...")
            logger.debug(f"  Local Option Tokens ({len(option_tokens_local)}): {option_tokens_local[:10]}...")
            logger.debug(f"  API Response Tokens ({len(response_tokens)}): {response_tokens[:20]}...")
            return None

        # Calculate average NLL (average of negative logprobs)
        avg_nll = -sum(option_logprobs) / len(option_logprobs)
        logger.debug(f"  API AvgNLL calculated: {avg_nll:.4f}")
        return avg_nll

    except (APIError, RateLimitError) as e:
        logger.error(f"API error calling {api_model_id} for option '{option_text[:30]}...': {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error processing API response for option '{option_text[:30]}...': {e}", exc_info=True)
        return None


# --- Main Ranking Logic (Updated) ---
def rank_mcq_options(
    # Local model args (can be None if using API)
    model,
    tokenizer,
    # API model args (can be None if using Local)
    api_client: Optional[OpenAI],
    api_model_id: Optional[str],
    # Common args
    data_item: Dict[str, Any],
    use_cache: bool, # Only for local
    max_context_len: int,
    model_max_length: int # Only for local
    ) -> Dict[str, Any]:
    """
    Ranks the MCQ options for a single data item based on AvgNLL,
    dispatching to either local or API calculation.
    """
    query_id = data_item.get("id", "unknown_id")
    context = data_item.get("context", "")
    query = data_item.get("query", "")
    options = data_item.get("options", {})

    results = {
        "id": query_id,
        "query": query,
        "options_scores": {},
        "ranked_options": [],
        "correct_answer_letter": data_item.get("correct_answer_letter"),
        "error": None,
        "method": "local" if model else "api" if api_model_id else "unknown"
    }

    if not context or not query or not options:
        results["error"] = "Missing context, query, or options."
        logger.warning(f"Skipping {query_id}: Missing context, query, or options.")
        return results

    prefix_prompt = format_prefix_prompt(context, query, max_context_len)
    option_scores = {}

    # --- Determine Calculation Method ---
    use_api = bool(api_model_id and api_client)
    use_local = bool(model and tokenizer)

    if not use_api and not use_local:
         results["error"] = "No valid model (local or API) available for scoring."
         logger.error(f"Skipping {query_id}: No scoring method available.")
         return results

    # Need a tokenizer for API parsing, even if not running locally
    # Load a fallback tokenizer if running API only
    local_tokenizer_for_parsing = tokenizer
    if use_api and not local_tokenizer_for_parsing:
         try:
             # Use a common tokenizer like gpt2 as fallback for parsing lengths
             # Or could load the tokenizer corresponding to the API model if known & accessible
             logger.info("Loading fallback tokenizer (gpt2) for API response parsing.")
             local_tokenizer_for_parsing = AutoTokenizer.from_pretrained("gpt2")
         except Exception as e:
             logger.error(f"Failed to load fallback tokenizer for API parsing: {e}")
             results["error"] = "Failed to load tokenizer needed for API parsing."
             return results


    logger.info(f"Processing query {query_id} using {'API' if use_api else 'Local'} method...")

    for letter, option_text in options.items():
        if not option_text:
            logger.warning(f"Skipping empty option {letter} for query {query_id}")
            continue

        logger.debug(f"  Scoring option {letter}: {option_text[:50]}...")
        avg_nll = None

        if use_api:
            avg_nll = calculate_option_neg_log_likelihood_api(
                 api_client=api_client,
                 api_model_id=api_model_id,
                 prefix_text=prefix_prompt,
                 option_text=option_text,
                 local_tokenizer_for_parsing=local_tokenizer_for_parsing # Pass tokenizer
            )
        elif use_local:
            avg_nll = calculate_option_neg_log_likelihood(
                model=model,
                tokenizer=tokenizer,
                prefix_text=prefix_prompt,
                option_text=option_text,
                use_cache=use_cache,
                max_length=model_max_length
            )

        # Record score or infinity on failure
        option_scores[letter] = avg_nll if avg_nll is not None else float('inf')
        if avg_nll is None:
             logger.warning(f"  Failed to score option {letter} for query {query_id} using {'API' if use_api else 'Local'} method.")


    results["options_scores"] = option_scores

    # Rank options
    if option_scores:
         ranked = sorted(option_scores.items(), key=lambda item: (item[1], item[0]))
         results["ranked_options"] = [
             {"letter": letter, "text": options.get(letter), "avg_nll": score if score != float('inf') else None}
             for letter, score in ranked
         ]
         if ranked and ranked[0][1] != float('inf'):
             logger.info(f"Finished query {query_id}. Best option: {ranked[0][0]} (AvgNLL: {ranked[0][1]:.4f})")
         else:
              logger.warning(f"Finished query {query_id}. All options failed scoring.")
    else:
        results["error"] = "No valid options could be scored."
        logger.warning(f"No valid options scored for query {query_id}")

    return results


# --- Main Execution (Updated) ---
def main():
    parser = argparse.ArgumentParser(description="Rank MCQ options using verifier LLM log probabilities (Local or API).")
    parser.add_argument("--input_data_file", type=str, required=True, help="Path to the processed input JSONL file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the ranked options JSONL file.")
    # --- Arguments for Local vs API ---
    parser.add_argument("--verifier_model_name", type=str, default=None, help="Hugging Face model ID for LOCAL verifier LLM.")
    parser.add_argument("--api_model_id", type=str, default=None, help="Model ID for API-based verifier (e.g., Fireworks 'accounts/fireworks/models/llama-v3p1-405b-instruct').")
    parser.add_argument("--fw_api_key_env", type=str, default="FIREWORKS_API_KEY", help="Environment variable name for Fireworks API Key.")
    # --- Arguments only for Local ---
    parser.add_argument("--model_max_length", type=int, default=4096, help="Maximum sequence length for LOCAL model.")
    parser.add_argument("--use_cache", action='store_true', default=True, help="Enable KV caching (local models only).")
    parser.add_argument("--no_cache", action='store_false', dest='use_cache', help="Disable KV caching (local models only).")
    parser.add_argument("--use_quantization", action='store_true', help="Use 4-bit quantization (local models only).")
    # --- Common Arguments ---
    parser.add_argument("--max_context_len", type=int, default=10000, help="Max characters of context to use in prefix prompt.")

    args = parser.parse_args()

    # --- Validate arguments ---
    if not args.verifier_model_name and not args.api_model_id:
        parser.error("Either --verifier_model_name (for local) or --api_model_id must be specified.")
    if args.verifier_model_name and args.api_model_id:
        parser.error("Cannot specify both --verifier_model_name and --api_model_id.")

    # --- Select Mode ---
    use_api = bool(args.api_model_id)
    use_local = bool(args.verifier_model_name)

    logger.info(f"--- Starting MCQ Option Ranking using Log Probs ---")
    logger.info(f"Input Data: {args.input_data_file}")
    logger.info(f"Output File: {args.output_file}")
    logger.info(f"Max Context Chars: {args.max_context_len}")

    # --- Initialize based on mode ---
    model, tokenizer = None, None
    client_for_api = None

    if use_local:
        logger.info(f"Mode: Local Model")
        logger.info(f"Verifier Model (Local): {args.verifier_model_name}")
        logger.info(f"Use KV Cache: {args.use_cache}")
        logger.info(f"Use Quantization: {args.use_quantization}")
        logger.info(f"Model Max Length: {args.model_max_length}")

        device_type_str = 'cpu'
        if torch.cuda.is_available():
            device_type_str = 'cuda'
            logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
            torch.cuda.empty_cache()
        else:
            logger.info("CUDA not available. Using CPU.")
            if args.use_quantization:
                 logger.warning("Quantization requested but CUDA not found. Will likely fail.")

        model, tokenizer = load_verifier_model(
             args.verifier_model_name,
             device_type_str,
             args.use_quantization
        )
        if not model or not tokenizer:
             logger.error("Failed to load local model/tokenizer. Exiting.")
             sys.exit(1)

    elif use_api:
        logger.info(f"Mode: API")
        logger.info(f"Verifier Model (API): {args.api_model_id}")
        api_key = os.getenv(args.fw_api_key_env)
        if not api_key:
             logger.error(f"API key environment variable '{args.fw_api_key_env}' not found. Cannot use API mode. Exiting.")
             sys.exit(1)
        if not fireworks_client: # Check if global client initialized
             logger.error(f"Fireworks client failed to initialize (API Key={args.fw_api_key_env} found but client init failed?). Exiting.")
             sys.exit(1)
        client_for_api = fireworks_client # Use the globally initialized client

    # --- Load Input Data ---
    input_data = load_json(args.input_data_file, load_by_line=True)
    if input_data is None:
        logger.error(f"Failed to load input data from {args.input_data_file}")
        sys.exit(1)
    logger.info(f"Loaded {len(input_data)} items from {args.input_data_file}")

    # --- Process Data and Rank Options ---
    results_list = []
    for item in tqdm(input_data, desc="Ranking Options"):
        rank_result = rank_mcq_options(
            model=model,         # Pass local model (or None)
            tokenizer=tokenizer,   # Pass local tokenizer (or None)
            api_client=client_for_api, # Pass API client (or None)
            api_model_id=args.api_model_id, # Pass API model ID (or None)
            data_item=item,
            use_cache=args.use_cache,
            max_context_len=args.max_context_len,
            model_max_length=args.model_max_length
        )
        results_list.append(rank_result)

    # --- Save Results ---
    logger.info(f"Saving {len(results_list)} ranking results to {args.output_file}")
    save_json(results_list, args.output_file, save_by_line=True)

    logger.info(f"--- Option Ranking Finished ---")

if __name__ == "__main__":
    main()