import json
import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np # For averaging scores

# Import metric functions directly
from .metrics.comprehensiveness import evaluate_comprehensiveness
from .metrics.task_coverage import evaluate_task_coverage
from .metrics.usability import evaluate_usability
from .metrics.efficiency import evaluate_efficiency
from .metrics.accuracy import evaluate_accuracy
from .llm_judge import LLMJudge # Import even if using placeholders
from .utils import load_json # Utility for loading config if needed


class HTAEvaluator:
    """
    Evaluates Hierarchical Task Analysis (HTA) decompositions based on 5 metrics.
    """
    METRICS = {
        'comprehensiveness': evaluate_comprehensiveness,
        'task_coverage': evaluate_task_coverage,
        'usability': evaluate_usability,
        'efficiency': evaluate_efficiency,
        'accuracy': evaluate_accuracy,
    }

    def __init__(self, config_path: Optional[str] = None, config: Optional[Dict] = None):
        """
        Initializes the HTA Evaluator.

        Args:
            config_path: Path to optional YAML configuration file (e.g., for LLM judge).
            config: Optional config dictionary (takes precedence over config_path).
        """
        self.config = self._load_config(config_path, config)
        # Initialize LLM judge if configuration is provided
        self.llm_judge = None
        if self.config and 'llm_judge' in self.config:
             print("Attempting to initialize LLM Judge...")
             try:
                 self.llm_judge = LLMJudge(config=self.config)
             except Exception as e:
                 print(f"Warning: Failed to initialize LLM Judge: {e}")
        else:
             print("LLM Judge configuration not found or not provided. Judge functionality disabled.")

        print("HTA Evaluator Initialized.")

    def _load_config(self, config_path: Optional[str], config_dict: Optional[Dict]) -> Optional[Dict]:
        """Loads configuration, prioritizing the dictionary."""
        if config_dict:
            return config_dict
        if config_path:
            loaded_cfg = load_json(config_path) # Assumes YAML loading handled by load_json or use PyYAML
            if loaded_cfg is None and config_path.endswith(('.yaml', '.yml')):
                 try:
                     import yaml
                     print(f"Loading YAML config from: {config_path}")
                     with open(config_path, 'r') as f:
                         return yaml.safe_load(f)
                 except ImportError:
                     print("Warning: PyYAML is not installed. Cannot load YAML config files.")
                     return None
                 except Exception as e:
                     print(f"Warning: Could not load YAML config from {config_path}: {e}")
                     return None
            return loaded_cfg # Return if loaded successfully by load_json
        return None

    def evaluate_subtask(
        self,
        subtask: Dict[str, Any],
        original_goal: str,
        available_tools: List[str],
        ground_truth_subtask: Optional[Dict[str, Any]] = None,
        context: Optional[Any] = None # Add context if needed for accuracy
    ) -> Dict[str, Optional[float]]:
        """
        Evaluates a single subtask based on the 5 HTA metrics.

        Args:
            subtask: The LLM-generated subtask dictionary. Expected keys might include
                     'id', 'goal', 'description' (list or string), 'plan', 'required_tools'.
            original_goal: The overall task goal given to the agent.
            available_tools: List of tools the agent system has access to.
            ground_truth_subtask: Optional human-created subtask for comparison.
            context: Optional context for accuracy evaluation.

        Returns:
            A dictionary containing scores (float or None) for each metric.
        """
        subtask_id = subtask.get('id', 'Unknown ID')
        print(f"\n--- Evaluating Subtask: {subtask_id} ---")
        scores = {}

        # Prepare arguments for metric functions
        metric_args = {
            'subtask': subtask,
            'original_goal': original_goal,
            'available_tools': available_tools,
            'ground_truth': ground_truth_subtask, # Use the specific GT subtask here
            'llm_judge': self.llm_judge,
            'context': context,
        }

        for metric_name, metric_func in self.METRICS.items():
            try:
                score = metric_func(**metric_args)
                # Ensure score is float or None (handle potential ints from simple heuristics)
                scores[metric_name] = float(score) if score is not None else None
            except Exception as e:
                print(f"ERROR evaluating metric '{metric_name}' for subtask '{subtask_id}': {e}")
                # import traceback
                # traceback.print_exc() # Uncomment for detailed debugging
                scores[metric_name] = None # Record failure

        return scores

    def align_subtasks(self,
        generated_subtasks: List[Dict[str, Any]],
        ground_truth_hta: List[Dict[str, Any]]
    ) -> List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]]:
        """
        Aligns generated subtasks with ground truth subtasks.
        Placeholder implementation: Aligns by index or 'id' field.
        Needs refinement for robust alignment (e.g., semantic matching).

        Args:
            generated_subtasks: List of subtasks from the agent.
            ground_truth_hta: List of subtasks from the human annotation.

        Returns:
            A list of tuples: (generated_subtask, matched_ground_truth_subtask or None).
        """
        aligned_pairs = []
        gt_map_by_id = {gt.get('id'): gt for gt in ground_truth_hta if gt.get('id')}
        used_gt_indices = set()

        for i, gen_subtask in enumerate(generated_subtasks):
            matched_gt = None
            gen_id = gen_subtask.get('id')

            # Try matching by ID first
            if gen_id and gen_id in gt_map_by_id:
                matched_gt = gt_map_by_id[gen_id]
                # Find index to mark as used (less efficient but ok for small lists)
                try:
                    gt_index = next(idx for idx, gt in enumerate(ground_truth_hta) if gt.get('id') == gen_id)
                    used_gt_indices.add(gt_index)
                except StopIteration:
                    pass # Should not happen if id is in map

            # If no ID match, try matching by index (if GT list is long enough and index not used)
            elif i < len(ground_truth_hta) and i not in used_gt_indices:
                 print(f"Warning: No ID match for generated subtask '{gen_id or i}'. Falling back to index {i} for alignment.")
                 matched_gt = ground_truth_hta[i]
                 used_gt_indices.add(i)
            else:
                 print(f"Warning: Could not align generated subtask '{gen_id or i}' to any ground truth subtask.")

            aligned_pairs.append((gen_subtask, matched_gt))

        # Note: This doesn't handle GT subtasks that weren't matched by any generated subtask.
        return aligned_pairs


    def evaluate_hta(
        self,
        generated_subtasks: List[Dict[str, Any]],
        original_goal: str,
        available_tools: List[str],
        ground_truth_hta: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Any] = None # Pass context down
    ) -> Dict[str, Any]:
        """
        Evaluates a full HTA (list of subtasks) generated by an agent.

        Args:
            generated_subtasks: List of subtask dictionaries from the LLM agent.
            original_goal: The overall task goal.
            available_tools: List of available tools.
            ground_truth_hta: Optional list of human-created subtasks for alignment.
            context: Optional context for accuracy evaluation.


        Returns:
            A dictionary containing scores per subtask and overall average scores.
            Example: {'subtask_scores': {'subtask_1': {...}, ...}, 'average_scores': {...}}
        """
        print(f"\n===== Evaluating HTA for Goal: {original_goal[:100]}... =====")
        all_subtask_scores = {}
        num_subtasks = len(generated_subtasks)

        if num_subtasks == 0:
            print("Warning: No generated subtasks provided for evaluation.")
            return {"error": "No generated subtasks provided", "subtask_scores": {}, "average_scores": {}}

        # Align generated subtasks with ground truth if available
        aligned_pairs = []
        if ground_truth_hta:
            print(f"Aligning {num_subtasks} generated subtasks with {len(ground_truth_hta)} ground truth subtasks...")
            aligned_pairs = self.align_subtasks(generated_subtasks, ground_truth_hta)
        else:
            # If no GT, create pairs with None for GT
            aligned_pairs = [(gen_sub, None) for gen_sub in generated_subtasks]

        # Evaluate each aligned pair
        valid_scores_count = {metric: 0 for metric in self.METRICS}
        total_scores = {metric: 0.0 for metric in self.METRICS}

        for gen_subtask, gt_subtask in aligned_pairs:
            subtask_id = gen_subtask.get('id', f'gen_subtask_{len(all_subtask_scores)}')
            subtask_scores = self.evaluate_subtask(
                subtask=gen_subtask,
                original_goal=original_goal,
                available_tools=available_tools,
                ground_truth_subtask=gt_subtask, 
                context=context
            )
            all_subtask_scores[subtask_id] = subtask_scores

            # Aggregate scores for averaging, handling None values
            for metric, score in subtask_scores.items():
                if score is not None:
                    try:
                        total_scores[metric] += float(score)
                        valid_scores_count[metric] += 1
                    except (ValueError, TypeError) as e:
                         print(f"Warning: Could not convert score '{score}' for metric '{metric}' to float. Skipping for average. Error: {e}")
                else:
                    print(f"Warning: Metric '{metric}' returned None for subtask '{subtask_id}'. Excluding from average.")

        # Calculate average scores, avoiding division by zero
        average_scores = {}
        for metric in self.METRICS:
            if valid_scores_count[metric] > 0:
                average_scores[metric] = round(total_scores[metric] / valid_scores_count[metric], 3)
            else:
                average_scores[metric] = None # No valid scores for this metric

        print("\n--- HTA Evaluation Summary ---")
        print("Average Scores:")
        for metric, avg_score in average_scores.items():
            print(f"  - {metric.capitalize()}: {avg_score if avg_score is not None else 'N/A'}")
        print("=================================")

        return {
            "subtask_scores": all_subtask_scores,
            "average_scores": average_scores
        }

    def compare_to_human(self, llm_judge_scores, human_scores):
        """
        Compares scores from LLM judges to human scores (correlation analysis).
        Placeholder for grounding study. Requires aligned datasets.
        """
        print("Placeholder: Comparing LLM judge scores to human scores...")
        if not llm_judge_scores or not human_scores:
            print("  - Need both LLM judge scores and human scores.")
            return None

        # Implementation would involve:
        # 1. Aligning scores based on item ID and metric.
        # 2. Using libraries like numpy or scipy for correlation (e.g., Pearson, Spearman).
        # Example:
        # import numpy as np
        # from scipy.stats import pearsonr
        # metric = 'accuracy'
        # llm_vals = [...] # Extract LLM scores for the metric
        # human_vals = [...] # Extract corresponding human scores
        # if len(llm_vals) == len(human_vals) and len(llm_vals) > 1:
        #    corr, p_value = pearsonr(llm_vals, human_vals)
        #    print(f"  - Correlation for {metric}: {corr:.3f} (p={p_value:.3f})")
        # else:
        #    print(f"  - Insufficient or misaligned data for {metric} correlation.")
        pass