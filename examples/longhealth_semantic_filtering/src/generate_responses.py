import argparse
import json
import os
import sys
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Any, Optional # Ensure typing is imported

# --- Path Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End Path Setup ---

# --- Model Loading and Generation Logic ---
MODEL_CACHE = {} # Simple cache to avoid reloading model/tokenizer

def load_model_and_tokenizer(model_name_or_path, device):
    """Loads model and tokenizer, using a simple cache."""
    if model_name_or_path in MODEL_CACHE:
        print(f"Using cached model/tokenizer for {model_name_or_path} on {device}")
        return MODEL_CACHE[model_name_or_path]
    print(f"Loading model: {model_name_or_path}...")
    try:
        # Determine dtype based on device
        # Use bfloat16 for faster computation and lower memory on compatible GPUs (like A10G/L40S)
        dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32
        print(f"Using dtype: {dtype}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
             # Set pad_token to eos_token if not defined
             print("Warning: pad_token not set, using eos_token as pad_token.")
             tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        MODEL_CACHE[model_name_or_path] = (model, tokenizer)
        print(f"Model {model_name_or_path} loaded successfully using {dtype} onto devices: {model.hf_device_map}")
        return model, tokenizer

    except Exception as e:
        print(f"FATAL ERROR loading model {model_name_or_path}: {e}")
        print("Check model name, internet connection, dependencies (torch, transformers, accelerate), and available VRAM.")
        sys.exit(1)

# --- format_multiple_choice_prompt function ---
# Allow options=None for dummy data compatibility
def format_multiple_choice_prompt(context: str, query: str, options: Optional[Dict[str, str]]) -> str:
    """
    Formats the prompt. Uses MC format if options are provided, otherwise uses a simpler format.
    """
    options_text = ""
    # Check if options is a dictionary and has content
    if isinstance(options, dict) and options:
        options_text = "\n".join(
            [f"{label}: {text}" for label, text in options.items() if text]
        )
    # Basic prompt structure
    base_prompt = f"""--------------BEGIN DOCUMENTS--------------

                    {context}

                    --------------END DOCUMENTS--------------

                    {query}"""

    if options_text:
     # Append MC specific parts only if options exist
        prompt = f"""{base_prompt}
        {options_text}

        Please answer using the following format:
        1. Begin your answer with the phrase "The correct answer is".
        2. State the letter of the correct option (e.g., A, B, C, D, E).
        3. Follow the letter with a colon and the exact text of the option you chose.
        4. Make sure your answer is a single, concise sentence.

        For example, if the correct answer to a question is option C, and the text for C is 'Acute Bronchitis', your answer should be:
        'The correct answer is C: Acute Bronchitis.'
        """
    else:
        # If no options (like in dummy data), just ask for a direct answer
        prompt = f"""{base_prompt}

                    Please provide a concise answer to the query based on the documents."""
    return prompt

# --- generate_actual_responses function (with Debugging) ---
def generate_actual_responses(
    model,
    tokenizer,
    context: str,
    query: str,
    options: Optional[Dict[str, str]], # Allow options=None
    num_responses: int = 1,
    temperature: float = 0.1,
    max_new_tokens: int = 50,
    top_p: float = 0.9,
    max_input_tokens: int = 32000
    ) -> list[str]:
    """
    Generates multiple responses for a single context-query pair using the loaded model.
    Formats prompt based on whether options are provided. Includes debugging prints.
    """
    user_prompt_text = format_multiple_choice_prompt(context, query, options)
    messages = [
        {"role": "system", "content": """You are a highly skilled..."""}, # Keep system prompt concise for logging
        {"role": "user", "content": user_prompt_text}
    ]
    try:
         prompt_to_tokenize = tokenizer.apply_chat_template(
             messages,
             tokenize=False,
             add_generation_prompt=True
         )
    except Exception as e:
         print(f"  DEBUG: Error applying chat template: {e}. Using basic formatting.")
         prompt_to_tokenize = f"System: [System Prompt Text]\nUser: {user_prompt_text}\nAssistant:"

    MAX_INPUT_TOKENS = max_input_tokens # Use value from args
    inputs = tokenizer(
        prompt_to_tokenize,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS
    )
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    if input_ids.shape[-1] >= MAX_INPUT_TOKENS:
         print(f"  Warning: Input prompt was truncated to {MAX_INPUT_TOKENS} tokens.")

    responses = []
    with torch.no_grad():
        for i in range(num_responses):
            try:
                outputs = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True if temperature > 0 and temperature != 1.0 else False, # Sample only if temp is not 0 or 1
                    temperature=max(temperature, 1e-4), # Ensure temp is positive for sampling
                    top_p=top_p,
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id, # Use pad_token_id if available
                    eos_token_id=tokenizer.eos_token_id # Explicitly set EOS token
                )
                # Decode only the newly generated tokens
                generated_ids = outputs[0][input_ids.shape[-1]:]
                response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                # Only append if the decoded text is not empty after stripping
                if response_text.strip():
                    responses.append(response_text.strip())
                else:
                    print(f"  DEBUG: Empty response generated for attempt {i+1}.")

            except Exception as e:
                 print(f"  - Error during generation for response {i+1}: {e}")
                 # Don't append error messages to the list of valid responses
                 # responses.append(f"[Generation Error: {e}]") # Avoid adding errors
                 if 'out of memory' in str(e).lower():
                     print("  - Out of memory error detected. Skipping remaining responses for this query.")
                     # No need to fill with errors if we only count valid ones
                     break # Exit inner loop
    return responses

# --- Main Script Logic ---
def main():
    parser = argparse.ArgumentParser(description="Generate LLM responses using Hugging Face models.")
    # (Argument parsing remains the same)
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Hugging Face model identifier.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to processed input JSONL file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save generated responses JSONL file.")
    parser.add_argument("--num_responses", type=int, default=1, help="Number of responses to generate per query.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature.")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Maximum number of *new* tokens for the answer.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling probability.")
    parser.add_argument("--max_input_tokens", type=int, default=32000, help="Maximum number of tokens for input truncation.")
    args = parser.parse_args()

    print(f"--- Starting Actual Response Generation ---")
    print(f"Model: {args.model_name_or_path}")
    print(f"Input: {args.input_file}")
    print(f"Output: {args.output_file}")
    print(f"Responses per query: {args.num_responses}")
    print(f"Temperature: {args.temperature}")
    print(f"Max New Tokens: {args.max_new_tokens}")
    print(f"Top P: {args.top_p}")
    print(f"Max Input Tokens: {args.max_input_tokens}")

    # --- Determine Device ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
        print("CUDA not available. Using CPU.")

    # --- Load Model ---
    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path, device)

    # --- Load Input Data ---
    try:
        with open(args.input_file, "r", encoding='utf-8') as infile:
            data = [json.loads(line) for line in infile if line.strip()]
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading or parsing input file {args.input_file}: {e}")
        sys.exit(1)

    # --- Generate Responses ---
    all_model_responses = []
    total_generated_count = 0 # Use a separate counter for valid responses added
    for item in tqdm(data, desc=f"Generating with {os.path.basename(args.model_name_or_path)}"):
        context = item.get("context", "")
        query = item.get("query", "")
        options = item.get("options") # Returns None if key doesn't exist
        query_id = item.get("id", "unknown_id")

        if not query or not context :
            print(f"Warning: Skipping item {query_id} due to missing query or context.")
            continue

        generated_texts = generate_actual_responses(
            model=model,
            tokenizer=tokenizer,
            context=context,
            query=query,
            options=options, # Pass None if options missing
            num_responses=args.num_responses,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            top_p=args.top_p,
            max_input_tokens=args.max_input_tokens
        )

        # --- Store results ---
        # This loop only runs if generated_texts is not empty
        for i, r_text in enumerate(generated_texts):
            # Check if the response is valid (not empty and not an error placeholder)
            if r_text and not r_text.startswith("[Generation Error"):
                output_entry = item.copy() # Start with original item data
                output_entry["model"] = args.model_name_or_path
                output_entry["response_text"] = r_text # Add the generated text
                output_entry["generation_params"] = { # Add params used
                    "temperature": args.temperature,
                    "max_new_tokens": args.max_new_tokens,
                    "top_p": args.top_p,
                    "index": i # Index within num_responses attempts
                }
                # Clean up large fields from original item before saving
                if "context" in output_entry: del output_entry["context"]
                # Keep options and correct answer if they exist in item
                # if "options" in output_entry: del output_entry["options"]

                all_model_responses.append(output_entry)
                total_generated_count += 1 # Increment counter only for successful appends

    # --- Save Results ---
    try:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, "w", encoding='utf-8') as outfile:
            for entry in all_model_responses: # This list might be empty
                outfile.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"\n--- Generation Summary ---")
        # Use the counter, not len(all_model_responses) directly
        print(f"Generated and saved {total_generated_count} valid responses for {len(data)} queries.")
        print(f"Responses saved to {args.output_file}")
        print("Response generation finished.")
    except IOError as e:
        print(f"Error writing output file {args.output_file}: {e}")
        sys.exit(1)
    finally:
         # Cleanup logic remains the same
         global MODEL_CACHE
         MODEL_CACHE = {}
         if 'model' in locals() : del model
         if 'tokenizer' in locals() : del tokenizer
         if torch.cuda.is_available(): torch.cuda.empty_cache()
         print("Cleaned up model resources.")

if __name__ == "__main__":
    main()
