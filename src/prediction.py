#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
prediction.py

Predict functions for novel objects using compound-based reasoning.
"""

from typing import List, Dict, Any, Tuple, Set
from collections import defaultdict
from src.compound_learning import Compound
from src.representation import match_compound


class FunctionPrediction:
    """Represents a predicted function with supporting evidence."""

    def __init__(
        self,
        function: str,
        score: float,
        supporting_compounds: List[Tuple[str, float]]
    ):
        """
        Args:
            function: Function name
            score: Overall prediction score
            supporting_compounds: List of (compound_id, contribution_score) tuples
        """
        self.function = function
        self.score = score
        self.supporting_compounds = supporting_compounds

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'function': self.function,
            'score': self.score,
            'supporting_compounds': [
                {'compound_id': cid, 'contribution': score}
                for cid, score in self.supporting_compounds
            ]
        }

    def __repr__(self):
        return f"Prediction({self.function}): score={self.score:.3f}"


def predict_functions(
    novel_object: Dict[str, Any],
    all_compounds: List[Compound],
    compound_function_map: Dict[str, Dict[str, float]],
    aggregation: str = 'max',
    top_k: int = 10,
    verbose: bool = False
) -> List[FunctionPrediction]:
    """
    Predict functions for a novel object.

    Algorithm:
    1. Find all compounds that match the object
    2. For each function, collect votes from matching compounds
    3. Aggregate votes (max, mean, or sum)
    4. Return top-k functions by score

    Args:
        novel_object: Object dict with 'features' field
        all_compounds: List of all compounds
        compound_function_map: Mapping from compound_id to {function: score}
        aggregation: How to combine votes ('max', 'mean', 'sum')
        top_k: Number of top predictions to return
        verbose: Print debug information

    Returns:
        List of FunctionPrediction objects, sorted by score
    """
    obj_features = set(novel_object.get('features', []))

    # Step 1: Find matching compounds
    matched_compounds = []
    for compound in all_compounds:
        if match_compound(obj_features, compound):
            matched_compounds.append(compound)

    if verbose:
        print(f"[Prediction] Matched {len(matched_compounds)} compounds")

    if not matched_compounds:
        return []

    # Step 2: Collect votes for each function
    function_votes = defaultdict(list)  # function -> list of (compound_id, score)

    for compound in matched_compounds:
        compound_id = compound.id
        if compound_id not in compound_function_map:
            continue

        for func, score in compound_function_map[compound_id].items():
            function_votes[func].append((compound_id, score))

    # Step 3: Aggregate votes
    predictions = []

    for func, votes in function_votes.items():
        scores = [score for _, score in votes]

        if aggregation == 'max':
            final_score = max(scores)
        elif aggregation == 'mean':
            final_score = sum(scores) / len(scores)
        elif aggregation == 'sum':
            final_score = sum(scores)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        # Sort supporting compounds by their contribution
        supporting_compounds = sorted(votes, key=lambda x: x[1], reverse=True)

        prediction = FunctionPrediction(
            function=func,
            score=final_score,
            supporting_compounds=supporting_compounds
        )
        predictions.append(prediction)

    # Step 4: Sort and return top-k
    predictions.sort(key=lambda x: x.score, reverse=True)
    return predictions[:top_k]


def predict_batch(
    objects: List[Dict[str, Any]],
    all_compounds: List[Compound],
    compound_function_map: Dict[str, Dict[str, float]],
    aggregation: str = 'max',
    top_k: int = 10
) -> Dict[str, List[FunctionPrediction]]:
    """
    Predict functions for multiple objects.

    Args:
        objects: List of objects
        all_compounds: All compounds
        compound_function_map: Compound-function mapping
        aggregation: Aggregation method
        top_k: Top predictions per object

    Returns:
        Dictionary mapping object_id to predictions
    """
    results = {}

    for obj in objects:
        obj_id = obj.get('object_id', obj.get('id', 'unknown'))
        predictions = predict_functions(
            obj,
            all_compounds,
            compound_function_map,
            aggregation=aggregation,
            top_k=top_k
        )
        results[obj_id] = predictions

    return results


def explain_prediction(
    prediction: FunctionPrediction,
    compound_map: Dict[str, Compound],
    max_compounds: int = 5
) -> str:
    """
    Generate human-readable explanation for a prediction.

    Args:
        prediction: FunctionPrediction object
        compound_map: Mapping from compound_id to Compound object
        max_compounds: Maximum number of compounds to show

    Returns:
        Explanation string
    """
    lines = [f"Function: {prediction.function} (score: {prediction.score:.3f})"]
    lines.append("Supporting evidence:")

    for i, (cid, score) in enumerate(prediction.supporting_compounds[:max_compounds]):
        if cid in compound_map:
            compound = compound_map[cid]
            features_str = ', '.join(sorted(list(compound.features))[:5])
            if len(compound.features) > 5:
                features_str += '...'

            if compound.layer == 1:
                lines.append(
                    f"  {i+1}. L1[{cid}] ({compound.space}): "
                    f"{features_str} (contribution: {score:.3f})"
                )
            else:
                components_str = ', '.join(compound.components)
                lines.append(
                    f"  {i+1}. L2[{cid}]: "
                    f"{components_str} (contribution: {score:.3f})"
                )

    return '\n'.join(lines)


def evaluate_predictions(
    objects: List[Dict[str, Any]],
    predictions_dict: Dict[str, List[FunctionPrediction]],
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Evaluate prediction accuracy.

    Args:
        objects: Ground truth objects with 'functions' field
        predictions_dict: Predictions from predict_batch
        top_k: Consider top-k predictions

    Returns:
        Dictionary with evaluation metrics
    """
    total_objects = 0
    correct_at_1 = 0
    correct_at_k = 0
    total_precision = 0.0
    total_recall = 0.0

    for obj in objects:
        obj_id = obj.get('object_id', obj.get('id', 'unknown'))
        if obj_id not in predictions_dict:
            continue

        true_functions = set(obj.get('functions', []))
        if not true_functions:
            continue

        predictions = predictions_dict[obj_id]
        if not predictions:
            total_objects += 1
            continue

        predicted_functions = [p.function for p in predictions[:top_k]]

        # Accuracy at 1
        if predictions[0].function in true_functions:
            correct_at_1 += 1

        # Accuracy at k
        if any(func in true_functions for func in predicted_functions):
            correct_at_k += 1

        # Precision and Recall
        predicted_set = set(predicted_functions)
        true_positive = len(predicted_set & true_functions)

        precision = true_positive / len(predicted_set) if predicted_set else 0.0
        recall = true_positive / len(true_functions) if true_functions else 0.0

        total_precision += precision
        total_recall += recall
        total_objects += 1

    if total_objects == 0:
        return {
            'accuracy_at_1': 0.0,
            'accuracy_at_k': 0.0,
            'avg_precision': 0.0,
            'avg_recall': 0.0,
            'total_objects': 0
        }

    return {
        'accuracy_at_1': correct_at_1 / total_objects,
        'accuracy_at_k': correct_at_k / total_objects,
        'avg_precision': total_precision / total_objects,
        'avg_recall': total_recall / total_objects,
        'total_objects': total_objects,
        'top_k': top_k
    }
