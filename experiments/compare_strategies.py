#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compare_strategies.py

Compare different prediction strategies:
- equal: All compounds vote equally
- weighted: L2 weighted higher than L1
- L2_priority: L2 first, L1 supplements
"""

import sys
import pickle
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hierarchical_prediction import (
    predict_hierarchical,
    predict_with_confidence,
    explain_prediction_contrast,
    identify_uncertain_predictions
)
from src.representation import represent_object


def evaluate_strategy(
    test_objects,
    L1_compounds,
    L2_compounds,
    compound_function_map,
    strategy,
    top_k=5
):
    """
    Evaluate a prediction strategy.

    Returns:
        Dictionary with metrics
    """
    total = 0
    correct_at_1 = 0
    correct_at_k = 0
    total_precision = 0.0
    total_recall = 0.0

    for obj in test_objects:
        true_functions = set(obj.get('functions', []))
        if not true_functions:
            continue

        # Create representation
        rep = represent_object(obj, L1_compounds, L2_compounds)

        # Predict
        predictions = predict_hierarchical(
            rep,
            compound_function_map,
            strategy=strategy,
            top_k=top_k
        )

        if not predictions:
            total += 1
            continue

        # Get predicted functions
        predicted_functions = [p.function for p in predictions]

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
        total += 1

    if total == 0:
        return {
            'accuracy_at_1': 0.0,
            'accuracy_at_k': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'total': 0
        }

    precision_avg = total_precision / total
    recall_avg = total_recall / total
    f1 = 2 * precision_avg * recall_avg / (precision_avg + recall_avg) if (precision_avg + recall_avg) > 0 else 0.0

    return {
        'accuracy_at_1': correct_at_1 / total,
        'accuracy_at_k': correct_at_k / total,
        'precision': precision_avg,
        'recall': recall_avg,
        'f1': f1,
        'total': total
    }


def main():
    """Compare prediction strategies."""
    print("=" * 70)
    print("PREDICTION STRATEGY COMPARISON")
    print("=" * 70)

    # Load model
    print("\n[INFO] Loading model...")
    with open('./output/model.pkl', 'rb') as f:
        model = pickle.load(f)

    L1_compounds = model['L1_compounds']
    L2_compounds = model['L2_compounds']
    compound_function_map = model['compound_function_map']
    all_compounds = model['all_compounds']

    # Load data
    print("[INFO] Loading data...")
    with open('./data/home_objects.json', 'r') as f:
        objects = json.load(f)

    # Split data
    train_objects, test_objects = train_test_split(
        objects,
        test_size=0.2,
        random_state=42
    )

    print(f"[INFO] Train: {len(train_objects)}, Test: {len(test_objects)}")

    # Strategies to compare
    strategies = {
        'equal': 'All compounds vote equally (baseline)',
        'weighted': 'L2 weighted 1.5x, L1 weighted 1.0x',
        'L2_priority': 'L2 votes first, L1 supplements gaps'
    }

    # Run comparison
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    results = {}
    for strategy, description in strategies.items():
        print(f"\n[Strategy] {strategy}")
        print(f"  Description: {description}")

        metrics = evaluate_strategy(
            test_objects,
            L1_compounds,
            L2_compounds,
            compound_function_map,
            strategy=strategy,
            top_k=5
        )

        results[strategy] = metrics

        print(f"  Accuracy@1: {metrics['accuracy_at_1']:.3f}")
        print(f"  Accuracy@5: {metrics['accuracy_at_k']:.3f}")
        print(f"  Precision:  {metrics['precision']:.3f}")
        print(f"  Recall:     {metrics['recall']:.3f}")
        print(f"  F1 Score:   {metrics['f1']:.3f}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Strategy':<15} {'Acc@1':>8} {'Acc@5':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-" * 70)

    for strategy, metrics in results.items():
        print(f"{strategy:<15} "
              f"{metrics['accuracy_at_1']:>8.3f} "
              f"{metrics['accuracy_at_k']:>8.3f} "
              f"{metrics['precision']:>10.3f} "
              f"{metrics['recall']:>8.3f} "
              f"{metrics['f1']:>8.3f}")

    # Find best strategy
    best_strategy = max(results.items(), key=lambda x: x[1]['f1'])[0]
    print(f"\nBest Strategy: {best_strategy} (F1={results[best_strategy]['f1']:.3f})")

    # Example predictions with confidence
    print("\n" + "=" * 70)
    print("EXAMPLE: Prediction with Confidence")
    print("=" * 70)

    sample_obj = test_objects[0]
    obj_id = sample_obj.get('object_id', 'unknown')
    true_funcs = sample_obj.get('functions', [])

    print(f"\nObject: {obj_id}")
    print(f"True functions: {true_funcs}")

    rep = represent_object(sample_obj, L1_compounds, L2_compounds)
    preds_with_conf = predict_with_confidence(
        rep,
        compound_function_map,
        all_compounds,
        strategy='weighted'
    )

    print(f"\nPredictions with confidence:")
    for i, pred in enumerate(preds_with_conf[:5], 1):
        is_correct = "✓" if pred['function'] in true_funcs else "✗"
        print(f"  {i}. {pred['function']:<12} "
              f"score={pred['score']:.3f}, "
              f"conf={pred['confidence']:.3f}, "
              f"L1={pred['l1_supports']}, "
              f"L2={pred['l2_supports']} {is_correct}")

    # Contrastive explanation
    if len(preds_with_conf) >= 2:
        print("\n" + "=" * 70)
        print("EXAMPLE: Contrastive Explanation")
        print("=" * 70)

        compound_map = {c.id: c for c in all_compounds}
        func_A = preds_with_conf[0]['function']
        func_B = preds_with_conf[1]['function']

        explanation = explain_prediction_contrast(
            rep,
            func_A,
            func_B,
            compound_function_map,
            compound_map
        )
        print(f"\n{explanation}")

    # Identify uncertain predictions
    print("\n" + "=" * 70)
    print("UNCERTAIN PREDICTIONS")
    print("=" * 70)

    uncertain = identify_uncertain_predictions(
        test_objects,
        compound_function_map,
        L1_compounds,
        L2_compounds,
        all_compounds,
        confidence_threshold=0.6
    )

    print(f"\nFound {len(uncertain)} uncertain predictions out of {len(test_objects)} test objects")
    print(f"Uncertainty rate: {len(uncertain) / len(test_objects) * 100:.1f}%")

    if uncertain:
        print("\nTop-5 uncertain cases:")
        for i, (obj_id, info) in enumerate(uncertain[:5], 1):
            print(f"\n  {i}. {obj_id}")
            print(f"     Reason: {info['reason']}")
            print(f"     Confidence: {info['confidence']:.3f}")
            if 'top_prediction' in info:
                print(f"     Top prediction: {info['top_prediction']} (score={info['score']:.3f})")

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
