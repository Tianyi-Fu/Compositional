#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate.py

Comprehensive evaluation of the hierarchical feature learning system.
Compares symbolic, RF, and hybrid approaches.

Usage:
    python experiments/evaluate.py
"""

import sys
import os
import json
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prediction import predict_batch, evaluate_predictions
from src.ml_integration import (
    CompoundImportanceAnalyzer,
    DecisionRuleExtractor
)


def load_model(model_path: str):
    """Load trained model."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def load_objects(data_path: str):
    """Load objects."""
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def evaluate_symbolic_method(
    test_objects,
    all_compounds,
    compound_function_map,
    aggregation='max'
):
    """Evaluate pure symbolic method."""
    print("\n" + "=" * 60)
    print("SYMBOLIC METHOD EVALUATION")
    print("=" * 60)

    predictions = predict_batch(
        test_objects,
        all_compounds,
        compound_function_map,
        aggregation=aggregation,
        top_k=10
    )

    # Evaluate at different k values
    for k in [1, 3, 5]:
        metrics = evaluate_predictions(test_objects, predictions, top_k=k)
        print(f"\n[Symbolic] Top-{k} Metrics:")
        print(f"  Accuracy@1: {metrics['accuracy_at_1']:.3f}")
        print(f"  Accuracy@{k}: {metrics['accuracy_at_k']:.3f}")
        print(f"  Avg Precision: {metrics['avg_precision']:.3f}")
        print(f"  Avg Recall: {metrics['avg_recall']:.3f}")

    return predictions, metrics


def evaluate_rf_method(
    train_objects,
    test_objects,
    all_compounds
):
    """Evaluate Random Forest method."""
    print("\n" + "=" * 60)
    print("RANDOM FOREST EVALUATION")
    print("=" * 60)

    # Train RF
    analyzer = CompoundImportanceAnalyzer(all_compounds)
    importances = analyzer.train_all_functions(train_objects, verbose=True)

    # Show top compounds for each function
    print("\n[RF] Top-5 Compounds per Function:")
    for func, func_importances in sorted(importances.items()):
        top_compounds = analyzer.get_top_compounds_for_function(
            func, func_importances, top_k=5
        )
        if top_compounds:
            compounds_str = ', '.join([f"{cid}({score:.3f})" for cid, score in top_compounds])
            print(f"  {func}: {compounds_str}")

    return analyzer, importances


def evaluate_decision_trees(
    train_objects,
    all_compounds
):
    """Evaluate Decision Tree rules."""
    print("\n" + "=" * 60)
    print("DECISION TREE RULES")
    print("=" * 60)

    extractor = DecisionRuleExtractor(all_compounds, max_depth=3)
    rules = extractor.extract_all_rules(train_objects, verbose=True)

    return extractor, rules


def compare_methods(
    train_objects,
    test_objects,
    all_compounds,
    compound_function_map
):
    """Compare all methods."""
    print("\n" + "=" * 60)
    print("METHOD COMPARISON")
    print("=" * 60)

    # Symbolic
    symbolic_preds, symbolic_metrics = evaluate_symbolic_method(
        test_objects,
        all_compounds,
        compound_function_map,
        aggregation='max'
    )

    # Random Forest
    rf_analyzer, rf_importances = evaluate_rf_method(
        train_objects,
        test_objects,
        all_compounds
    )

    # Decision Trees
    dt_extractor, dt_rules = evaluate_decision_trees(
        train_objects,
        all_compounds
    )

    # Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\n[Comparison] Symbolic vs RF:")
    print(f"  Symbolic Accuracy@1: {symbolic_metrics['accuracy_at_1']:.3f}")
    print(f"  (RF accuracy computed during training via cross-validation)")

    return {
        'symbolic': (symbolic_preds, symbolic_metrics),
        'rf': (rf_analyzer, rf_importances),
        'dt': (dt_extractor, dt_rules)
    }


def analyze_compound_quality(
    all_compounds,
    rf_importances,
    compound_function_map
):
    """Analyze compound quality using RF importance."""
    print("\n" + "=" * 60)
    print("COMPOUND QUALITY ANALYSIS")
    print("=" * 60)

    # Aggregate importance across all functions
    compound_total_importance = {}
    for func, func_importances in rf_importances.items():
        for cid, importance in func_importances.items():
            if cid not in compound_total_importance:
                compound_total_importance[cid] = 0.0
            compound_total_importance[cid] += importance

    # Sort by total importance
    sorted_compounds = sorted(
        compound_total_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Show top compounds
    print("\n[Analysis] Top-20 Most Important Compounds (by RF):")
    compound_map = {c.id: c for c in all_compounds}

    for i, (cid, total_imp) in enumerate(sorted_compounds[:20], 1):
        if cid in compound_map:
            c = compound_map[cid]
            features_str = ', '.join(sorted(list(c.features))[:4])
            if len(c.features) > 4:
                features_str += '...'

            # Count how many functions this compound is associated with
            num_functions = len(compound_function_map.get(cid, {}))

            print(f"  {i:2d}. {cid} (L{c.layer}, {c.space}): "
                  f"importance={total_imp:.3f}, "
                  f"functions={num_functions}, "
                  f"PMI={c.pmi:.2f}")
            print(f"      Features: {features_str}")

    # Show least important (potentially prunable)
    print("\n[Analysis] Bottom-10 Least Important Compounds (candidates for pruning):")
    for i, (cid, total_imp) in enumerate(sorted_compounds[-10:], 1):
        if cid in compound_map:
            c = compound_map[cid]
            print(f"  {i:2d}. {cid} (L{c.layer}, {c.space}): importance={total_imp:.3f}")


def main():
    """Main evaluation pipeline."""
    print("=" * 60)
    print("HIERARCHICAL FEATURE LEARNING - EVALUATION")
    print("=" * 60)

    # Configuration
    model_path = './output_augmented/model.pkl'
    data_path = './data/home_objects_augmented.json'
    test_size = 0.2
    random_state = 42

    # Load model
    print(f"\n[INFO] Loading model from {model_path}")
    model = load_model(model_path)

    all_compounds = model['all_compounds']
    compound_function_map = model['compound_function_map']
    L1_compounds = model['L1_compounds']
    L2_compounds = model['L2_compounds']

    print(f"[INFO] Loaded {len(all_compounds)} compounds")
    print(f"  L1: {sum(len(c) for c in L1_compounds.values())}")
    print(f"  L2: {len(L2_compounds)}")

    # Load data and split
    objects = load_objects(data_path)
    train_objects, test_objects = train_test_split(
        objects,
        test_size=test_size,
        random_state=random_state
    )

    print(f"\n[INFO] Data split: {len(train_objects)} train, {len(test_objects)} test")

    # Compare methods
    results = compare_methods(
        train_objects,
        test_objects,
        all_compounds,
        compound_function_map
    )

    # Analyze compound quality
    analyze_compound_quality(
        all_compounds,
        results['rf'][1],  # rf_importances
        compound_function_map
    )

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
