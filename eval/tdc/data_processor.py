"""
Data processor for TDC (Therapeutics Data Commons) binary classification tasks.

All TDC tasks in this setup are binary classification where:
- Y=0 maps to answer (A) (the negative/inactive class)
- Y=1 maps to answer (B) (the positive/active class)

The 'text' field contains the full prompt with pre-computed molecular properties
already prepended, so no additional context extraction is needed.
"""

import os
import re
import json
from typing import List, Dict, Any


def load_data(data_path: str) -> List[Dict[str, Any]]:
    """
    Load data from a JSONL file.
    
    Args:
        data_path: Path to the JSONL file
        
    Returns:
        List of dictionaries containing the data
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    
    print(f"Loaded {len(data)} samples from {data_path}")
    return data


class DataProcessor:
    """
    Processor for TDC binary classification tasks.
    
    Handles data preprocessing, answer correctness checking, and accuracy evaluation.
    All tasks follow the same (A)/(B) binary classification format.
    """
    
    # Mapping from Y values to answer choices
    Y_TO_ANSWER = {0: "(A)", 1: "(B)"}
    
    def __init__(self, task_name: str):
        """
        Initialize the data processor.
        
        Args:
            task_name: Name of the TDC task (e.g., 'AMES', 'BBB_Martins')
        """
        self.task_name = task_name
    
    def process_task_data(self, raw_data: List[Dict]) -> List[Dict]:
        """
        Process raw TDC data into the standard format expected by ACE.
        
        The 'text' field already contains the full prompt (pre-computed molecular
        properties + question), so it maps directly to 'question'.
        Context is empty since everything is already in 'text'.
        Y (0 or 1) maps to target: (A) or (B).
        
        Args:
            raw_data: Raw data from JSONL file with 'text', 'Y', 'drug' fields
            
        Returns:
            Processed data in standard ACE format with 'question', 'context', 'target'
        """
        processed_data = []
        
        for item in raw_data:
            text = item.get('text', '').replace("put ONLY your final choice ((A) or (B)) after \"Answer:\"", "give your final answer following the format below.")  # 适配 ace 的格式
            y_value = item.get('Y', 0)
            drug = item.get('drug', '')
            
            # Map Y to answer choice
            target = self.Y_TO_ANSWER.get(y_value, "(A)")
            
            processed_item = {
                "question": text,       # Full prompt is the question
                "context": "",          # No separate context needed
                "target": target,       # (A) or (B)
                "others": {
                    "drug": drug,
                    "y_value": y_value,
                    "task": self.task_name,
                }
            }
            
            processed_data.append(processed_item)
        
        return processed_data
    
    def _normalize_answer(self, answer: str) -> str:
        """
        Normalize a model's answer to (A) or (B).
        
        Handles various formats: "(A)", "A", "(a)", "a", etc.
        
        Args:
            answer: Raw answer string from model
            
        Returns:
            Normalized answer: "(A)" or "(B)" or the original if unrecognizable
        """
        answer = answer.strip()
        
        # Try to find (A) or (B) pattern
        match = re.search(r'\(([ABab])\)', answer)
        if match:
            return f"({match.group(1).upper()})"
        
        # Try standalone A or B (case-insensitive)
        match = re.search(r'\b([ABab])\b', answer)
        if match:
            return f"({match.group(1).upper()})"
        
        # Return original if we can't parse
        return answer
    
    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        """
        Check if the predicted answer matches the ground truth.
        
        Args:
            predicted: Model's answer (may need normalization)
            ground_truth: Ground truth answer, either "(A)" or "(B)"
            
        Returns:
            True if the answer is correct
        """
        normalized_pred = self._normalize_answer(predicted)
        normalized_gt = self._normalize_answer(ground_truth)
        return normalized_pred == normalized_gt
    
    def _answer_to_label(self, answer: str) -> int:
        """
        Convert a normalized answer to an integer label.
        
        Args:
            answer: Answer string (will be normalized)
            
        Returns:
            0 for (A), 1 for (B), -1 if unrecognizable
        """
        normalized = self._normalize_answer(answer)
        if normalized in ["(A)", 'A', ' A', 'A.']:
            return 0
        elif normalized in ["(B)", 'B', ' B', 'B.']:
            return 1
        return -1

    def evaluate_accuracy(self, predictions: List[str], targets: List[str]) -> float:
        """
        Compute accuracy over a list of predictions and targets.
        
        Args:
            predictions: List of model predictions
            targets: List of ground truth targets
            
        Returns:
            Accuracy as a float between 0 and 1
        """
        if len(predictions) != len(targets):
            raise ValueError("Predictions and targets must have the same length.")
        
        if len(predictions) == 0:
            return 0.0
        
        correct = sum(
            1 for pred, target in zip(predictions, targets)
            if self.answer_is_correct(pred, target)
        )
        
        return correct / len(predictions)

    def evaluate_f1(self, predictions: List[str], targets: List[str]) -> float:
        """
        Compute macro F1 score over a list of predictions and targets.
        
        Converts (A)/(B) string answers to integer labels (0/1) and uses
        sklearn's f1_score with average='macro'. Unrecognizable predictions
        are forced to the opposite of the ground truth label (same penalty
        strategy as the reference implementation).
        
        Args:
            predictions: List of model predictions
            targets: List of ground truth targets
            
        Returns:
            Macro F1 score as a float between 0 and 1
        """
        from sklearn.metrics import f1_score
        
        if len(predictions) != len(targets):
            raise ValueError("Predictions and targets must have the same length.")
        
        if len(predictions) == 0:
            return 0.0
        
        y_true = []
        y_pred = []
        
        for pred, target in zip(predictions, targets):
            label_true = self._answer_to_label(target)
            label_pred = self._answer_to_label(pred)
            
            # If prediction is unrecognizable, force wrong prediction
            if label_pred == -1:
                label_pred = 1 - label_true
            
            y_true.append(label_true)
            y_pred.append(label_pred)
        
        return f1_score(y_true, y_pred, average='macro')
