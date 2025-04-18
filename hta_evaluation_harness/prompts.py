from typing import Dict, Any, Optional
import json # Using json for potentially long context formatting

# Prompts based on the HTA Evaluation Rubrics for evaluating a generated response
# to a query, given some context.

LLM_JUDGE_PROMPTS = {
    "comprehensiveness": """
    You are evaluating an AI assistant's response based on provided medical documents.
    The goal was to answer the query based *only* on the documents.
    Evaluate the **Comprehensiveness** of the provided 'Generated Response' on a scale of 1-5, according to these levels:
    Level 1: Response is extremely vague with no specific details; critical components are missing entirely.
    Level 2: Response has minimal detail; many critical components are missing or poorly defined.
    Level 3: Response covers main aspects but lacks some details or edge cases relevant to the query and context.
    Level 4: Response is detailed with most aspects covered thoroughly and specific information identified.
    Level 5: Response is exhaustive, covering all relevant aspects from the context with exact details specified clearly.

    Query: "{query}"

    Provided Context (Medical Documents):
    --- START CONTEXT ---
    {context}
    --- END CONTEXT ---

    Generated Response:
    --- START RESPONSE ---
    {response_text}
    --- END RESPONSE ---

    Provide a brief rationale explaining your score based *only* on the information available in the Provided Context and how well the response reflects it, then provide the numeric score.
    Rationale: [Your reasoning here]
    Score: [1-5]
    """,

        "task_coverage": """
    You are evaluating an AI assistant's response based on provided medical documents.
    The goal was to answer the query based *only* on the documents.
    Evaluate the **Task Coverage** of the provided 'Generated Response' regarding the 'Query' on a scale of 1-5, according to these levels:
    Level 1: Response addresses only core requirements of the query while ignoring many necessary components found in the context.
    Level 2: Response addresses most core requirements but misses several important secondary components from the context.
    Level 3: Response addresses all core requirements with some secondary components from the context missing.
    Level 4: Response addresses all core requirements and most secondary components found in the context.
    Level 5: Response comprehensively addresses all requirements of the query with no gaps in coverage relative to the information available in the context.

    Query: "{query}"

    Provided Context (Medical Documents):
    --- START CONTEXT ---
    {context}
    --- END CONTEXT ---

    Generated Response:
    --- START RESPONSE ---
    {response_text}
    --- END RESPONSE ---

    Provide a brief rationale explaining your score based *only* on the information available in the Provided Context, then provide the numeric score.
    Rationale: [Your reasoning here]
    Score: [1-5]
    """,

        "usability": """
    You are evaluating an AI assistant's response based on provided medical documents.
    The goal was to answer the query based *only* on the documents.
    Evaluate the **Usability** of the information provided in the 'Generated Response' for answering the 'Query' on a scale of 1-5. Consider clarity, actionability, and how directly it addresses the query based on the context.
    Level 1: Response is unusable, unclear, ambiguous, or completely irrelevant to the query.
    Level 2: Response is difficult to use, lacks clarity, or only tangentially addresses the query.
    Level 3: Response is reasonably usable but could be clearer or more direct in answering the query.
    Level 4: Response is clear, directly addresses the query, and is easy to understand based on the context.
    Level 5: Response is optimally usable, exceptionally clear, unambiguous, directly answers the query, and presents information effectively based on the context.

    Query: "{query}"

    Provided Context (Medical Documents):
    --- START CONTEXT ---
    {context}
    --- END CONTEXT ---

    Generated Response:
    --- START RESPONSE ---
    {response_text}
    --- END RESPONSE ---

    Provide a brief rationale explaining your score based *only* on the information available in the Provided Context, then provide the numeric score.
    Rationale: [Your reasoning here]
    Score: [1-5]
    """,

        "efficiency": """
    You are evaluating an AI assistant's response based on provided medical documents.
    The goal was to answer the query based *only* on the documents.
    Evaluate the **Efficiency** of the 'Generated Response' in conveying the necessary information from the context to answer the query on a scale of 1-5. Consider conciseness and avoidance of irrelevant information.
    Level 1: Response is extremely verbose or contains significant irrelevant information compared to what's needed from the context.
    Level 2: Response is noticeably verbose or includes distracting irrelevant details.
    Level 3: Response is somewhat verbose or includes minor irrelevant details.
    Level 4: Response is mostly concise with minimal irrelevant information.
    Level 5: Response is optimally concise, conveying the necessary information from the context directly without unnecessary details.

    Query: "{query}"

    Provided Context (Medical Documents):
    --- START CONTEXT ---
    {context}
    --- END CONTEXT ---

    Generated Response:
    --- START RESPONSE ---
    {response_text}
    --- END RESPONSE ---

    Provide a brief rationale explaining your score based *only* on the information available in the Provided Context, then provide the numeric score.
    Rationale: [Your reasoning here]
    Score: [1-5]
    """,

        "accuracy": """
    You are evaluating an AI assistant's response based on provided medical documents.
    The goal was to answer the query based *only* on the documents.
    Evaluate the **Factual Accuracy** of the 'Generated Response' on a scale of 1-5. Accuracy means the information presented in the response is correct and directly supported by the 'Provided Context (Medical Documents)'. Do not use external knowledge.
    Level 1: Response contains significant factual errors or fabrications unsupported by the context.
    Level 2: Response contains noticeable factual inaccuracies or statements not clearly supported by the context.
    Level 3: Response is mostly factually accurate but contains minor inaccuracies or unsupported claims.
    Level 4: Response is factually accurate with only very minor, inconsequential deviations from the context.
    Level 5: Response is completely factually accurate and directly supported by the provided context.

    Query: "{query}"

    Provided Context (Medical Documents):
    --- START CONTEXT ---
    {context}
    --- END CONTEXT ---

    Generated Response:
    --- START RESPONSE ---
    {response_text}
    --- END RESPONSE ---

    Provide a brief rationale explaining your score based *only* on the information available in the Provided Context, then provide the numeric score.
    Rationale: [Your reasoning here]
    Score: [1-5]
    """
}

def get_llm_judge_prompt(metric_name: str, data: Dict[str, Any]) -> Optional[str]:
    """
    Formats a prompt for a given metric using data for the LLM Judge.

    Args:
        metric_name: The name of the metric (e.g., 'comprehensiveness').
        data: A dictionary containing necessary fields like 'query', 'context', 'response_text'.

    Returns:
        The formatted prompt string, or None if the metric name is invalid.
    """
    prompt_template = LLM_JUDGE_PROMPTS.get(metric_name)
    if not prompt_template:
        print(f"Error: No LLM Judge prompt template found for metric '{metric_name}'")
        return None
    # Prepare data, ensuring keys exist and formatting context potentially
    query = data.get('query', data.get('goal', 'N/A')) # Use query or goal
    # Use response_text first, fallback to plan/description if needed
    response_text = data.get('response_text')
    if not response_text:
         response_text = data.get('plan', data.get('description', 'N/A'))

    context = data.get('context', 'No context provided.')

    # Limit context length in prompt to avoid excessive token usage for the judge
    # Adjust this limit as needed based on judge model's context window and cost
    MAX_CONTEXT_CHARS_FOR_JUDGE = 10000 # Example limit
    if isinstance(context, str) and len(context) > MAX_CONTEXT_CHARS_FOR_JUDGE:
        print(f"  - Warning: Truncating context provided to LLM Judge for metric '{metric_name}'.")
        context = context[:MAX_CONTEXT_CHARS_FOR_JUDGE] + "\n... [Context Truncated] ..."
    elif not isinstance(context, str): # Handle non-string context if necessary
         context = str(context) # Convert to string, might need better handling
    format_data = {
        "query": query,
        "context": context,
        "response_text": response_text
    }
    # Check if all required keys for the specific template are present
    # (This basic check assumes all templates use query, context, response_text)
    if not query or query == 'N/A':
         print(f"Warning: Missing 'query'/'goal' for metric '{metric_name}'.")
    if not response_text or response_text == 'N/A':
         print(f"Warning: Missing 'response_text'/'plan'/'description' for metric '{metric_name}'.")
         return None
    try:
        return prompt_template.format(**format_data)
    except KeyError as e:
        print(f"Error formatting LLM judge prompt for '{metric_name}': Missing key {e}")
        return None
    except Exception as e:
         print(f"An unexpected error occurred during LLM judge prompt formatting: {e}")
         return None

