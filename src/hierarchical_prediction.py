#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
hierarchical_prediction.py

Enhanced prediction with hierarchical priority:
- L2 compounds get higher priority (they represent cross-space patterns)
- L1 compounds supplement gaps
- Multiple strategies for comparison
"""

from typing import List, Dict, Any, Tuple
from collections import defaultdict
from src.representation import ObjectRepresentation
from src.prediction import FunctionPrediction


def predict_hierarchical(
    obj_representation: ObjectRepresentation,
    compound_function_map: Dict[str, Dict[str, float]],
    strategy: str = 'L2_priority',
    l2_weight: float = 1.5,
    l1_weight: float = 1.0,
    top_k: int = 10
) -> List[FunctionPrediction]:
    """
    Hierarchical prediction with different strategies.

    Strategies:
    - 'equal': All compounds vote equally (baseline)
    - 'weighted': L2 compounds get higher weight
    - 'L2_priority': L2 votes first, L1 supplements only if needed
    - 'L2_implies_L1': L2 votes imply their component L1 votes

    Args:
        obj_representation: Object's hierarchical representation
        compound_function_map: Compound -> function -> score mapping
        strategy: Prediction strategy
        l2_weight: Weight for L2 compounds
        l1_weight: Weight for L1 compounds
        top_k: Number of top predictions to return

    Returns:
        List of FunctionPrediction objects
    """
    function_votes = defaultdict(float)
    supporting_compounds = defaultdict(list)

    if strategy == 'equal':
        # Baseline: all compounds vote equally
        all_compounds = obj_representation.L1 + obj_representation.L2
        for compound_id in all_compounds:
            if compound_id not in compound_function_map:
                continue
            for func, score in compound_function_map[compound_id].items():
                function_votes[func] += score
                supporting_compounds[func].append((compound_id, score))

    elif strategy == 'weighted':
        # L2 compounds get higher weight
        for compound_id in obj_representation.L2:
            if compound_id not in compound_function_map:
                continue
            for func, score in compound_function_map[compound_id].items():
                weighted_score = score * l2_weight
                function_votes[func] += weighted_score
                supporting_compounds[func].append((compound_id, weighted_score))

        for compound_id in obj_representation.L1:
            if compound_id not in compound_function_map:
                continue
            for func, score in compound_function_map[compound_id].items():
                weighted_score = score * l1_weight
                function_votes[func] += weighted_score
                supporting_compounds[func].append((compound_id, weighted_score))

    elif strategy == 'L2_priority':
        # L2 votes first, L1 supplements gaps
        l2_covered_functions = set()

        # Step 1: L2 compounds vote (high priority)
        for compound_id in obj_representation.L2:
            if compound_id not in compound_function_map:
                continue
            for func, score in compound_function_map[compound_id].items():
                weighted_score = score * l2_weight
                function_votes[func] += weighted_score
                supporting_compounds[func].append((compound_id, weighted_score))
                if score >= 0.5:  # Consider as covered
                    l2_covered_functions.add(func)

        # Step 2: L1 compounds supplement uncovered or weak functions
        for compound_id in obj_representation.L1:
            if compound_id not in compound_function_map:
                continue
            for func, score in compound_function_map[compound_id].items():
                # Only add L1 vote if:
                # 1. Function not covered by L2, OR
                # 2. L2 coverage is weak (< 0.5)
                if func not in l2_covered_functions or function_votes[func] < 0.5 * l2_weight:
                    weighted_score = score * l1_weight
                    function_votes[func] += weighted_score
                    supporting_compounds[func].append((compound_id, weighted_score))

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Create predictions
    predictions = []
    for func, total_score in function_votes.items():
        # Sort supporting compounds by contribution
        supports = sorted(supporting_compounds[func], key=lambda x: x[1], reverse=True)

        prediction = FunctionPrediction(
            function=func,
            score=total_score,
            supporting_compounds=supports
        )
        predictions.append(prediction)

    # Sort by score and return top-k
    predictions.sort(key=lambda x: x.score, reverse=True)
    return predictions[:top_k]


def predict_with_confidence(
    obj_representation: ObjectRepresentation,
    compound_function_map: Dict[str, Dict[str, float]],
    all_compounds_list: List[Any],
    strategy: str = 'weighted'
) -> List[Dict[str, Any]]:
    """
    Predict with confidence scores.

    Confidence is based on:
    1. Number of supporting compounds
    2. Strength of association scores
    3. Agreement between L1 and L2

    Args:
        obj_representation: Object representation
        compound_function_map: Compound-function mapping
        all_compounds_list: List of all compound objects (for layer info)
        strategy: Prediction strategy

    Returns:
        List of dicts with function, score, confidence, and supporting_compounds
    """
    predictions = predict_hierarchical(
        obj_representation,
        compound_function_map,
        strategy=strategy
    )

    # Build compound layer map
    compound_layers = {}
    for c in all_compounds_list:
        compound_layers[c.id] = c.layer

    results = []
    for pred in predictions:
        # Calculate confidence metrics
        num_supports = len(pred.supporting_compounds)
        l2_supports = sum(1 for cid, _ in pred.supporting_compounds
                         if compound_layers.get(cid, 1) == 2)
        l1_supports = num_supports - l2_supports

        # Confidence score:
        # - Base: normalized prediction score
        # - Bonus: multiple supporting compounds
        # - Bonus: L2 support (cross-space validation)
        base_confidence = min(pred.score, 1.0)
        support_bonus = min(num_supports / 5.0, 0.3)  # Max 0.3 bonus
        l2_bonus = min(l2_supports / 3.0, 0.2)  # Max 0.2 bonus

        confidence = min(base_confidence + support_bonus + l2_bonus, 1.0)

        results.append({
            'function': pred.function,
            'score': pred.score,
            'confidence': confidence,
            'num_supports': num_supports,
            'l1_supports': l1_supports,
            'l2_supports': l2_supports,
            'supporting_compounds': pred.supporting_compounds[:5]  # Top 5
        })

    return results


def explain_prediction_contrast(
    obj_representation: ObjectRepresentation,
    func_A: str,
    func_B: str,
    compound_function_map: Dict[str, Dict[str, float]],
    compound_map: Dict[str, Any]
) -> str:
    """
    Explain why function A is predicted over function B.

    Args:
        obj_representation: Object representation
        func_A: Predicted function
        func_B: Alternative function
        compound_function_map: Compound-function mapping
        compound_map: Mapping from compound_id to Compound object

    Returns:
        Human-readable explanation string
    """
    # Collect supporting compounds for each function
    supports_A = []
    supports_B = []

    all_compounds = obj_representation.L1 + obj_representation.L2

    for cid in all_compounds:
        if cid not in compound_function_map:
            continue

        score_A = compound_function_map[cid].get(func_A, 0.0)
        score_B = compound_function_map[cid].get(func_B, 0.0)

        if score_A > 0:
            supports_A.append((cid, score_A))
        if score_B > 0:
            supports_B.append((cid, score_B))

    # Sort by score
    supports_A.sort(key=lambda x: x[1], reverse=True)
    supports_B.sort(key=lambda x: x[1], reverse=True)

    # Build explanation
    lines = [f"Comparison: '{func_A}' vs '{func_B}'"]
    lines.append("=" * 60)

    # Function A
    total_A = sum(score for _, score in supports_A)
    lines.append(f"\n'{func_A}' (total score: {total_A:.3f})")
    lines.append(f"  Supported by {len(supports_A)} compounds:")
    for i, (cid, score) in enumerate(supports_A[:3], 1):
        if cid in compound_map:
            c = compound_map[cid]
            features_str = ', '.join(sorted(list(c.features))[:3])
            lines.append(f"    {i}. {cid} (L{c.layer}, {c.space}): {features_str}... → {score:.3f}")

    # Function B
    total_B = sum(score for _, score in supports_B)
    lines.append(f"\n'{func_B}' (total score: {total_B:.3f})")
    lines.append(f"  Supported by {len(supports_B)} compounds:")
    for i, (cid, score) in enumerate(supports_B[:3], 1):
        if cid in compound_map:
            c = compound_map[cid]
            features_str = ', '.join(sorted(list(c.features))[:3])
            lines.append(f"    {i}. {cid} (L{c.layer}, {c.space}): {features_str}... → {score:.3f}")

    # Key difference
    lines.append(f"\nKey Difference:")
    lines.append(f"  Score advantage: {total_A - total_B:.3f}")
    lines.append(f"  Support count: {len(supports_A)} vs {len(supports_B)}")

    # Find discriminative compounds (strong for A, weak for B)
    discriminative = []
    for cid, score_A in supports_A:
        score_B = compound_function_map.get(cid, {}).get(func_B, 0.0)
        diff = score_A - score_B
        if diff > 0.3:  # Significant difference
            discriminative.append((cid, diff, score_A))

    if discriminative:
        discriminative.sort(key=lambda x: x[1], reverse=True)
        lines.append(f"\nDiscriminative compounds (favor '{func_A}'):")
        for i, (cid, diff, score_A) in enumerate(discriminative[:2], 1):
            if cid in compound_map:
                c = compound_map[cid]
                lines.append(f"    {i}. {cid} (L{c.layer}): Δ={diff:.3f} (A:{score_A:.3f})")

    return '\n'.join(lines)


def identify_uncertain_predictions(
    test_objects: List[Dict[str, Any]],
    compound_function_map: Dict[str, Dict[str, float]],
    L1_compounds: Dict[str, List[Any]],
    L2_compounds: List[Any],
    all_compounds_list: List[Any],
    confidence_threshold: float = 0.6
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Identify objects with uncertain predictions.

    These objects may need:
    - More features
    - Human annotation
    - Additional training examples

    Args:
        test_objects: List of test objects
        compound_function_map: Compound-function mapping
        L1_compounds: L1 compounds by space
        L2_compounds: L2 compounds
        all_compounds_list: All compounds
        confidence_threshold: Minimum confidence threshold

    Returns:
        List of (object_id, prediction_info) tuples for uncertain objects
    """
    from src.representation import represent_object

    uncertain_objects = []

    for obj in test_objects:
        obj_id = obj.get('object_id', obj.get('id', 'unknown'))

        # Create representation
        rep = represent_object(obj, L1_compounds, L2_compounds)

        # Get predictions with confidence
        preds = predict_with_confidence(
            rep,
            compound_function_map,
            all_compounds_list,
            strategy='weighted'
        )

        if not preds:
            # No predictions at all - very uncertain
            uncertain_objects.append((obj_id, {
                'reason': 'no_matching_compounds',
                'confidence': 0.0,
                'num_L1': len(rep.L1),
                'num_L2': len(rep.L2)
            }))
        elif preds[0]['confidence'] < confidence_threshold:
            # Low confidence prediction
            uncertain_objects.append((obj_id, {
                'reason': 'low_confidence',
                'confidence': preds[0]['confidence'],
                'top_prediction': preds[0]['function'],
                'score': preds[0]['score'],
                'num_supports': preds[0]['num_supports']
            }))
        elif len(preds) > 1 and preds[0]['score'] - preds[1]['score'] < 0.2:
            # Ambiguous: top 2 predictions very close
            uncertain_objects.append((obj_id, {
                'reason': 'ambiguous',
                'confidence': preds[0]['confidence'],
                'top_predictions': [preds[0]['function'], preds[1]['function']],
                'scores': [preds[0]['score'], preds[1]['score']]
            }))

    return uncertain_objects
