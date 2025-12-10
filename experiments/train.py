#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train.py

Main training script for the hierarchical feature learning system.

Usage:
    python experiments/train.py
"""

import sys
import os
import json
import pickle
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.compound_learning import discover_all_compounds
from src.representation import (
    represent_all_objects,
    get_all_compounds_flat
)
from src.association_learning import (
    learn_compound_function_associations,
    build_compound_function_map
)
from src.prediction import predict_functions, explain_prediction
from src.utils import validate_objects


# ========== Configuration ==========

CONFIG = {
    # Data paths
    'data_path': './data/home_objects_augmented.json',
    'output_dir': './output_augmented',

    # L1 compound discovery
    'l1_pmi_threshold': 0.8,
    'l1_min_support': 2,
    'l1_max_size': 4,

    # L2 compound discovery
    'l2_pmi_threshold': 0.5,
    'l2_min_support': 2,
    'l2_max_size': 3,

    # Association learning
    'min_coverage_pos': 0.3,

    # Prediction
    'prediction_aggregation': 'max',  # 'max', 'mean', or 'sum'
}


def load_objects(path: str):
    """Load objects from JSON file."""
    print(f"\n[INFO] Loading objects from {path}")
    with open(path, 'r', encoding='utf-8') as f:
        objects = json.load(f)

    validate_objects(objects)
    print(f"[INFO] Loaded {len(objects)} objects")
    return objects


def save_model(output_dir: str, **components):
    """Save model components."""
    os.makedirs(output_dir, exist_ok=True)

    # Save as pickle for Python use
    model_path = os.path.join(output_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(components, f)
    print(f"\n[INFO] Model saved to {model_path}")

    # Save compounds as JSON for inspection
    compounds_data = {
        'L1_compounds': {},
        'L2_compounds': []
    }

    for space, compounds in components['L1_compounds'].items():
        compounds_data['L1_compounds'][space] = [c.to_dict() for c in compounds]

    compounds_data['L2_compounds'] = [c.to_dict() for c in components['L2_compounds']]

    json_path = os.path.join(output_dir, 'compounds.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(compounds_data, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Compounds saved to {json_path}")

    # Save associations as JSON
    associations_data = []
    for compound_id, func_assocs in components['associations'].items():
        for func, assoc in func_assocs.items():
            associations_data.append(assoc.to_dict())

    assoc_path = os.path.join(output_dir, 'associations.json')
    with open(assoc_path, 'w', encoding='utf-8') as f:
        json.dump(associations_data, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Associations saved to {assoc_path}")


def test_on_sample_objects(
    objects,
    all_compounds,
    compound_function_map,
    compound_map,
    num_samples=3
):
    """Test prediction on sample objects."""
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS")
    print("=" * 60)

    import random
    sample_objects = random.sample(objects, min(num_samples, len(objects)))

    for obj in sample_objects:
        obj_id = obj.get('object_id', 'unknown')
        true_functions = obj.get('functions', [])

        print(f"\n[Sample] Object: {obj_id}")
        print(f"[Sample] True functions: {true_functions}")
        print(f"[Sample] Features: {obj.get('features', [])[:8]}...")

        predictions = predict_functions(
            obj,
            all_compounds,
            compound_function_map,
            aggregation=CONFIG['prediction_aggregation'],
            top_k=5
        )

        if predictions:
            print(f"[Sample] Top-3 Predictions:")
            for i, pred in enumerate(predictions[:3], 1):
                print(f"  {i}. {pred.function}: {pred.score:.3f}")

                # Show supporting compounds
                top_compounds = pred.supporting_compounds[:2]
                for cid, score in top_compounds:
                    if cid in compound_map:
                        c = compound_map[cid]
                        feats_str = ', '.join(sorted(list(c.features))[:3])
                        print(f"     <- {cid} ({c.space}, {feats_str}...): {score:.3f}")
        else:
            print(f"[Sample] No predictions")

        print("-" * 60)


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("HIERARCHICAL FEATURE LEARNING - TRAINING")
    print("=" * 60)

    # Step 1: Load data
    objects = load_objects(CONFIG['data_path'])

    # Step 2: Discover compounds
    L1_compounds, L2_compounds = discover_all_compounds(
        objects,
        l1_pmi_threshold=CONFIG['l1_pmi_threshold'],
        l2_pmi_threshold=CONFIG['l2_pmi_threshold'],
        l1_min_support=CONFIG['l1_min_support'],
        l2_min_support=CONFIG['l2_min_support'],
        l1_max_size=CONFIG['l1_max_size'],
        l2_max_size=CONFIG['l2_max_size'],
        verbose=True
    )

    # Step 3: Create object representations
    representations = represent_all_objects(
        objects,
        L1_compounds,
        L2_compounds,
        verbose=True
    )

    # Step 4: Learn compound-function associations
    all_compounds = get_all_compounds_flat(L1_compounds, L2_compounds)
    associations = learn_compound_function_associations(
        objects,
        all_compounds,
        min_coverage_pos=CONFIG['min_coverage_pos'],
        verbose=True
    )

    # Step 5: Build compound-function map for prediction
    compound_function_map = build_compound_function_map(associations)

    # Step 6: Save model
    save_model(
        CONFIG['output_dir'],
        L1_compounds=L1_compounds,
        L2_compounds=L2_compounds,
        associations=associations,
        compound_function_map=compound_function_map,
        all_compounds=all_compounds,
        config=CONFIG
    )

    # Step 7: Test on sample objects
    compound_map = {c.id: c for c in all_compounds}
    test_on_sample_objects(
        objects,
        all_compounds,
        compound_function_map,
        compound_map,
        num_samples=3
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nModel saved to: {CONFIG['output_dir']}")
    print("\nNext steps:")
    print("  1. Run: python experiments/evaluate.py")
    print("  2. Integrate Random Forest: python experiments/ml_integration.py")


if __name__ == '__main__':
    main()
