import os
import yaml
import re
import time
from typing import Dict, Any, Optional

# --- Ensure necessary client libraries are imported ---
try:
    from openai import OpenAI, RateLimitError as OpenAIRateLimitError, APIError as OpenAIAPIError
    print("OpenAI library imported successfully.")
except ImportError:
    print("Warning: OpenAI library not found. Install with 'pip install openai'")
    OpenAI = None # Handle missing dependency
    OpenAIRateLimitError = None
    OpenAIAPIError = None

# Add imports for other providers if you plan to support them
# try:
#     from anthropic import Anthropic, RateLimitError as AnthropicRateLimitError, APIError as AnthropicAPIError
# except ImportError:
#     Anthropic = None
#     AnthropicRateLimitError = None
#     AnthropicAPIError = None

class LLMJudge:
    """
    Interacts with an LLM (e.g., GPT, Claude) to get scores or judgments based on prompts.
    Handles basic API interaction and retries.
    """
    def __init__(self, config_path: Optional[str] = None, config: Optional[Dict] = None):
        """
        Initializes the LLM Judge.

        Args:
            config_path: Path to a YAML configuration file.
            config: A dictionary containing configuration. Takes precedence over config_path.
        """
        self.config = self._load_config(config_path, config)
        self.client = self._initialize_client() # <<< This call initializes the client
        # Added check here to confirm client status after initialization attempt
        if self.client:
             model_name = self.config.get('llm_judge', {}).get('model_name', 'N/A')
             print(f"LLM Judge client initialized successfully for model: {model_name}")
        else:
             print("Warning: LLM Judge client initialization failed. Evaluation will use placeholders.")


    def _load_config(self, config_path: Optional[str], config_dict: Optional[Dict]) -> Optional[Dict]:
        """Loads configuration from dict or file."""
        if config_dict:
            return config_dict
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                print(f"Warning: Could not load LLM judge config from {config_path}: {e}")
        return None

    def _initialize_client(self):
        """Initializes the API client based on config."""
        if not self.config or 'llm_judge' not in self.config:
            print("Error: LLM Judge config section ('llm_judge') not found in config. Cannot initialize client.")
            return None

        judge_config = self.config['llm_judge']
        provider = judge_config.get('provider')
        api_key_env = judge_config.get('api_key_env')
        api_key = os.getenv(api_key_env) if api_key_env else None

        if not api_key:
            # This error should have been caught by the dotenv loading in evaluate_pipeline,
            # but double-check here.
            print(f"Error: API key environment variable '{api_key_env}' not found in environment. Cannot initialize LLM Judge client.")
            return None

        # --- THIS IS THE CRITICAL PART ---
        # Ensure the correct client initialization is UNCOMMENTED
        if provider == "openai":
            if OpenAI is None:
                print("Error: 'openai' library not installed. Please install it (`pip install openai`).")
                return None
            try:
                print(f"Initializing OpenAI client for model {judge_config.get('model_name')}...")
                # --- UNCOMMENTED THIS LINE ---
                client = OpenAI(api_key=api_key)
                # --- END UNCOMMENTED LINE ---
                return client
            except Exception as e:
                print(f"Error initializing OpenAI client: {e}")
                return None

        # elif provider == "anthropic": # Example for another provider
        #     if Anthropic is None:
        #          print("Error: 'anthropic' library not installed...")
        #          return None
        #     try:
        #          print(f"Initializing Anthropic client for model {judge_config.get('model_name')}...")
        #          # --- UNCOMMENT THIS LINE IF USING ANTHROPIC ---
        #          # client = Anthropic(api_key=api_key)
        #          return client
        #     except Exception as e:
        #          print(f"Error initializing Anthropic client: {e}")
        #          return None

        else:
             print(f"Error: LLM Judge provider '{provider}' is not currently supported in llm_judge.py.")
             return None
        # --- End Critical Part ---

    def get_judgment(self, prompt: str, params: Optional[Dict] = None, retries: int = 3, delay: int = 5) -> Optional[str]:
        """
        Sends a prompt to the configured LLM and gets a textual judgment.
        Includes basic retry logic for rate limits / transient errors.
        """
        # --- Check if client was initialized ---
        if not self.client:
             print("Error: LLM Judge client not initialized. Cannot get judgment.")
             # Fallback to placeholder ONLY if client is None
             print(f"Placeholder: Simulating LLM judgment for prompt: '{prompt[:100]}...'")
             return "Placeholder Judgment: Client not init. Score: 3.5" # Keep placeholder score here
        # --- End Check ---

        judge_config = self.config.get('llm_judge', {})
        provider = judge_config.get('provider')
        model_name = judge_config.get('model_name', 'default-model')
        default_params = judge_config.get('generation_params', {})
        final_params = {**default_params, **(params or {})}

        print(f"  - Sending prompt to LLM Judge ({provider}/{model_name})...")

        last_exception = None
        for attempt in range(retries + 1):
            try:
                if provider == "openai":
                    response = self.client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        **final_params
                    )
                    result = response.choices[0].message.content.strip()
                    return result
                # Add elif blocks for other providers here if implemented

                else:
                    print(f"Error: Unsupported provider '{provider}' in get_judgment.")
                    return None

            # Use specific exception types if available
            except (OpenAIRateLimitError, OpenAIAPIError) if OpenAIRateLimitError else Exception as e:
                last_exception = e
                print(f"  - API Error (Attempt {attempt + 1}/{retries + 1}): {e}")
                if attempt < retries:
                    print(f"  - Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print("  - Max retries reached.")
            except Exception as e:
                last_exception = e
                print(f"  - Unexpected Error during API call (Attempt {attempt + 1}): {e}")
                if attempt < retries:
                    print(f"  - Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print("  - Max retries reached.")

        print(f"Error: Failed to get judgment after {retries + 1} attempts. Last error: {last_exception}")
        return None # Return None after all retries fail

    def get_score(self, prompt: str, params: Optional[Dict] = None, min_score: float = 1.0, max_score: float = 5.0) -> Optional[float]:
        """
        Sends a prompt designed to elicit a numerical score and attempts to parse it.
        """
        judgment_text = self.get_judgment(prompt, params)
        if judgment_text is None:
            print(f"  - Warning: Failed to get judgment text from LLM Judge for score parsing.")
            return None

        # --- Score Parsing Logic ---
        # (Parsing logic remains the same)
        try:
            score_match = re.search(r"Score:\s*(\d+(?:\.\d+)?)", judgment_text, re.IGNORECASE)
            if score_match:
                score = float(score_match.group(1))
                score = max(min_score, min(max_score, score))
                print(f"  - Parsed score: {score:.1f}")
                return score

            num_match = re.search(r"(\d+(?:\.\d+)?)\s*$", judgment_text)
            if num_match:
                 score = float(num_match.group(1))
                 score = max(min_score, min(max_score, score))
                 print(f"  - Parsed score (fallback - end of string): {score:.1f}")
                 return score

            num_match = re.search(r"^\s*(\d+(?:\.\d+)?)", judgment_text)
            if num_match:
                 score = float(num_match.group(1))
                 score = max(min_score, min(max_score, score))
                 print(f"  - Parsed score (fallback - start of string): {score:.1f}")
                 return score

            print(f"  - Warning: Could not parse score from judgment text: '{judgment_text[:100]}...'")
            return None

        except ValueError as e:
            print(f"  - Warning: Error converting extracted text to float during score parsing: {e}. Text: '{judgment_text[:100]}...'")
            return None
        except Exception as e:
             print(f"  - Unexpected error during score parsing: {e}. Text: '{judgment_text[:100]}...'")
             return None