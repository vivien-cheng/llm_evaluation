from typing import Dict, Any, Optional

try:
    from ..llm_judge import LLMJudge
    from ..prompts import get_llm_judge_prompt
except ImportError:
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from llm_judge import LLMJudge
    from prompts import get_llm_judge_prompt

def evaluate_efficiency(
    subtask: Dict[str, Any],
    context: Optional[str] = None,
    llm_judge: Optional[LLMJudge] = None, 
    **kwargs
) -> Optional[float]:
    """
    Evaluates the efficiency (conciseness) of the response using an LLM Judge
    based on the provided rubric.
    """
    metric_name = "efficiency" # Use lowercase for prompt key lookup
    print(f"Evaluating {metric_name.upper()} for subtask: {subtask.get('id', 'N/A')}")

    if not llm_judge:
        print(f"  - Warning: LLM Judge not provided for {metric_name}. Cannot evaluate.")
        return None

    # Prepare data for the prompt formatter
    prompt_data = {
        "query": subtask.get('goal'),
        "context": context,
        "response_text": subtask.get('response_text', subtask.get('plan', subtask.get('description')))
    }

    if not prompt_data["response_text"]:
         print(f"  - Warning: No response text found in subtask for {metric_name}. Cannot evaluate.")
         return None

    # Get the formatted prompt
    prompt = get_llm_judge_prompt(metric_name, prompt_data)

    if not prompt:
        print(f"  - Error: Failed to generate prompt for {metric_name}.")
        return None

    # Call the LLM judge to get the score
    print(f"  - Calling LLM Judge for {metric_name.upper()}...")
    score = llm_judge.get_score(prompt, min_score=1.0, max_score=5.0)

    if score is not None:
        print(f"  - {metric_name.upper()} (LLM Judge Score): {score:.1f}")
    else:
        print(f"  - Warning: LLM Judge failed to return a valid score for {metric_name.upper()}.")
        return None

    return score
