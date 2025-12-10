#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
drinkable_example.py

Complete walkthrough of how the system predicts "drinkable" function.
Shows L0 → L1 → L2 → Prediction pipeline step by step.
"""

import sys
import pickle
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.representation import represent_object
from src.hierarchical_prediction import (
    predict_hierarchical,
    predict_with_confidence,
    explain_prediction_contrast
)


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def main():
    """Complete drinkable prediction example."""

    print_section("DRINKABLE FUNCTION PREDICTION - COMPLETE EXAMPLE")

    # Load model
    print("\n[Step 0] Loading trained model...")
    with open('./output/model.pkl', 'rb') as f:
        model = pickle.load(f)

    L1_compounds = model['L1_compounds']
    L2_compounds = model['L2_compounds']
    all_compounds = model['all_compounds']
    compound_function_map = model['compound_function_map']
    associations = model['associations']

    print(f"  Loaded {len(all_compounds)} compounds")
    print(f"    L1: {sum(len(c) for c in L1_compounds.values())}")
    print(f"    L2: {len(L2_compounds)}")

    # Create compound map for easy lookup
    compound_map = {c.id: c for c in all_compounds}

    # ========================================================================
    # Example 1: Glass Tumbler (should be drinkable)
    # ========================================================================

    print_section("EXAMPLE 1: Glass Tumbler (Novel Object)")

    novel_object = {
        'object_id': 'novel_glass_tumbler',
        'features': [
            # Structure features
            'geom_cavity',      # Has a cavity to hold liquid
            'size_small',       # Small enough to hold

            # Material features
            'mat_glass',        # Made of glass

            # Physical features
            'phys_rigid',       # Rigid structure

            # Appearance features
            'tex_smooth',       # Smooth texture
            'color_transparent',# Transparent

            # Location features
            'loc_tabletop',     # Found on tabletop
            'room_kitchen',     # In kitchen

            # Affordance features (what you can do with it)
            'aff_drink_from',   # Can drink from it
            'aff_graspable',    # Can grasp it
            'aff_store_in',     # Can store liquid in it
        ]
    }

    print("\nNovel Object Features (L0 - Raw Features):")
    print(f"  Total: {len(novel_object['features'])} features")
    for space in ['geom_', 'mat_', 'phys_', 'tex_', 'color_', 'loc_', 'room_', 'aff_']:
        space_feats = [f for f in novel_object['features'] if f.startswith(space)]
        if space_feats:
            print(f"    {space[:-1]}: {space_feats}")

    # ========================================================================
    # Step 1: Create Hierarchical Representation
    # ========================================================================

    print_section("STEP 1: Hierarchical Representation (L0 → L1 → L2)")

    obj_repr = represent_object(novel_object, L1_compounds, L2_compounds)

    print(f"\nL0 (Raw Features): {len(obj_repr.L0)} features")
    print(f"  {obj_repr.L0[:5]}...")

    print(f"\nL1 (Single-Space Compounds): {len(obj_repr.L1)} matched")
    if obj_repr.L1:
        print("  Matched L1 compounds:")
        for cid in obj_repr.L1[:10]:  # Show first 10
            if cid in compound_map:
                c = compound_map[cid]
                features_str = ', '.join(sorted(list(c.features))[:3])
                if len(c.features) > 3:
                    features_str += '...'
                print(f"    {cid} ({c.space}): {features_str} (PMI={c.pmi:.2f})")

    print(f"\nL2 (Cross-Space Compounds): {len(obj_repr.L2)} matched")
    if obj_repr.L2:
        print("  Matched L2 compounds (first 5):")
        for cid in obj_repr.L2[:5]:
            if cid in compound_map:
                c = compound_map[cid]
                components_str = ', '.join(c.components[:3])
                if len(c.components) > 3:
                    components_str += '...'
                print(f"    {cid}: [{components_str}] (PMI={c.pmi:.2f})")

    # ========================================================================
    # Step 2: Examine "drinkable" related compounds
    # ========================================================================

    print_section("STEP 2: Which Compounds Associate with 'drink'?")

    drink_associations = {}
    for cid in obj_repr.L1 + obj_repr.L2:
        if cid in compound_function_map:
            if 'drink' in compound_function_map[cid]:
                score = compound_function_map[cid]['drink']
                drink_associations[cid] = score

    # Sort by score
    sorted_drink = sorted(drink_associations.items(), key=lambda x: x[1], reverse=True)

    print(f"\nFound {len(sorted_drink)} compounds that associate with 'drink':")
    print("\nTop-10 Compounds Supporting 'drink':")
    print(f"  {'ID':<8} {'Layer':<6} {'Space':<12} {'Score':<8} {'Features/Components'}")
    print("  " + "-" * 78)

    for i, (cid, score) in enumerate(sorted_drink[:10], 1):
        if cid in compound_map:
            c = compound_map[cid]
            if c.layer == 1:
                features_str = ', '.join(sorted(list(c.features))[:3])
                if len(c.features) > 3:
                    features_str += '...'
                info = features_str
            else:
                info = ', '.join(c.components[:3])
                if len(c.components) > 3:
                    info += '...'

            print(f"  {cid:<8} L{c.layer:<5} {c.space:<12} {score:<8.3f} {info}")

    # ========================================================================
    # Step 3: Show detailed association data
    # ========================================================================

    print_section("STEP 3: Detailed Association Data for Top Compounds")

    # Show top 3 in detail
    for i, (cid, score) in enumerate(sorted_drink[:3], 1):
        if cid not in compound_map or cid not in associations:
            continue

        c = compound_map[cid]
        assoc_dict = associations[cid]

        print(f"\n{i}. Compound {cid} (Layer {c.layer}, {c.space})")
        print(f"   PMI Score: {c.pmi:.2f}")
        print(f"   Support: {c.support} objects")

        if c.layer == 1:
            print(f"   Features: {sorted(list(c.features))}")
        else:
            print(f"   Components: {c.components}")
            # Show what L1 compounds it's made of
            print(f"   Composed of:")
            for comp_id in c.components[:3]:
                if comp_id in compound_map:
                    comp = compound_map[comp_id]
                    comp_feats = ', '.join(sorted(list(comp.features))[:2])
                    print(f"     - {comp_id} ({comp.space}): {comp_feats}...")

        # Show function associations
        if 'drink' in assoc_dict:
            drink_assoc = assoc_dict['drink']
            print(f"   Association with 'drink':")
            print(f"     Coverage in positive examples: {drink_assoc.coverage_pos:.1%}")
            print(f"     Coverage in negative examples: {drink_assoc.coverage_neg:.1%}")
            print(f"     Support (positive): {drink_assoc.support_pos} objects")
            print(f"     Association score: {drink_assoc.association_score:.3f}")

    # ========================================================================
    # Step 4: Prediction with different strategies
    # ========================================================================

    print_section("STEP 4: Prediction with Different Strategies")

    strategies = ['equal', 'weighted', 'L2_priority']

    for strategy in strategies:
        predictions = predict_hierarchical(
            obj_repr,
            compound_function_map,
            strategy=strategy,
            top_k=5
        )

        print(f"\nStrategy: {strategy}")
        print(f"  Top-5 Predictions:")
        for i, pred in enumerate(predictions[:5], 1):
            is_drink = "★" if pred.function == 'drink' else " "
            print(f"    {i}. {is_drink} {pred.function:<12} score={pred.score:.3f} "
                  f"({len(pred.supporting_compounds)} compounds)")

    # ========================================================================
    # Step 5: Prediction with Confidence
    # ========================================================================

    print_section("STEP 5: Prediction with Confidence Scoring")

    preds_with_conf = predict_with_confidence(
        obj_repr,
        compound_function_map,
        all_compounds,
        strategy='weighted'
    )

    print("\nPredictions with Confidence:")
    print(f"  {'Rank':<6} {'Function':<12} {'Score':<8} {'Confidence':<12} {'L1':<6} {'L2':<6} {'Total'}")
    print("  " + "-" * 70)

    for i, pred in enumerate(preds_with_conf[:8], 1):
        is_drink = "★" if pred['function'] == 'drink' else " "
        print(f"  {i}{is_drink:<5} {pred['function']:<12} "
              f"{pred['score']:<8.3f} {pred['confidence']:<12.3f} "
              f"{pred['l1_supports']:<6} {pred['l2_supports']:<6} "
              f"{pred['num_supports']}")

    # Show supporting compounds for 'drink'
    drink_pred = next((p for p in preds_with_conf if p['function'] == 'drink'), None)
    if drink_pred:
        print(f"\nSupporting Compounds for 'drink' (score={drink_pred['score']:.3f}):")
        for cid, score in drink_pred['supporting_compounds'][:5]:
            if cid in compound_map:
                c = compound_map[cid]
                print(f"  - {cid} (L{c.layer}, {c.space}): {score:.3f}")

    # ========================================================================
    # Step 6: Contrastive Explanation
    # ========================================================================

    print_section("STEP 6: Why 'drink' and not 'eat'?")

    if len(preds_with_conf) >= 2:
        # Find drink and another function
        drink_func = next((p['function'] for p in preds_with_conf if p['function'] == 'drink'), None)
        other_func = next((p['function'] for p in preds_with_conf if p['function'] != 'drink'), None)

        if drink_func and other_func:
            explanation = explain_prediction_contrast(
                obj_repr,
                drink_func,
                other_func,
                compound_function_map,
                compound_map
            )
            print(f"\n{explanation}")

    # ========================================================================
    # Step 7: Manual Verification
    # ========================================================================

    print_section("STEP 7: Manual Verification - Why is it drinkable?")

    print("\nKey Evidence for 'drinkable':")

    # Check structure compounds
    print("\n1. Structure Evidence:")
    structure_compounds = [cid for cid in obj_repr.L1 if cid in compound_map and compound_map[cid].space == 'structure']
    for cid in structure_compounds[:3]:
        c = compound_map[cid]
        if 'drink' in compound_function_map.get(cid, {}):
            score = compound_function_map[cid]['drink']
            print(f"   {cid}: {sorted(list(c.features))} → drink={score:.3f}")
            print(f"      Reasoning: Has cavity (holds liquid) + graspable size")

    # Check affordance compounds
    print("\n2. Affordance Evidence:")
    affordance_compounds = [cid for cid in obj_repr.L1 if cid in compound_map and compound_map[cid].space == 'affordance']
    for cid in affordance_compounds[:3]:
        c = compound_map[cid]
        if 'drink' in compound_function_map.get(cid, {}):
            score = compound_function_map[cid]['drink']
            features = sorted(list(c.features))
            print(f"   {cid}: {features} → drink={score:.3f}")
            if 'aff_drink_from' in c.features:
                print(f"      Reasoning: Explicitly labeled as drinkable")

    # Check cross-space L2 compounds
    print("\n3. Cross-Space Evidence (L2):")
    l2_drink = [(cid, compound_function_map.get(cid, {}).get('drink', 0))
                for cid in obj_repr.L2 if cid in compound_map]
    l2_drink = sorted(l2_drink, key=lambda x: x[1], reverse=True)

    for cid, score in l2_drink[:3]:
        if score > 0:
            c = compound_map[cid]
            print(f"   {cid}: {c.components} → drink={score:.3f}")
            print(f"      Reasoning: Combines structure + affordance + material patterns")

    print("\n4. Holistic Reasoning:")
    print("   The object is drinkable because:")
    print("   - Structure: Has cavity (holds liquid) + small size (graspable)")
    print("   - Material: Glass (safe for drinks, transparent to see contents)")
    print("   - Physical: Rigid (won't spill), smooth (comfortable on lips)")
    print("   - Affordance: Explicitly marked as drink_from + graspable")
    print("   - Location: Kitchen tabletop (typical context for drinking vessels)")
    print("\n   Multiple independent evidence from different feature spaces")
    print("   converge to the same conclusion: DRINKABLE!")

    # ========================================================================
    # Summary
    # ========================================================================

    print_section("SUMMARY")

    print("\nPrediction Pipeline:")
    print("  1. Raw Features (L0) → 12 features")
    print(f"  2. L1 Compounds → {len(obj_repr.L1)} matched")
    print(f"  3. L2 Compounds → {len(obj_repr.L2)} matched")
    print(f"  4. Voting → {len(drink_associations)} compounds vote for 'drink'")
    print(f"  5. Final Score → {drink_pred['score']:.3f}" if drink_pred else "  5. Final Score → N/A")
    print(f"  6. Confidence → {drink_pred['confidence']:.3f}" if drink_pred else "  6. Confidence → N/A")

    print("\nKey Compounds:")
    if sorted_drink:
        top3 = sorted_drink[:3]
        for i, (cid, score) in enumerate(top3, 1):
            c = compound_map.get(cid)
            if c:
                print(f"  {i}. {cid} (L{c.layer}, {c.space}): {score:.3f}")

    print("\nConclusion:")
    if drink_pred and drink_pred['score'] > 0.5:
        print("  ✓ CORRECTLY PREDICTED AS DRINKABLE")
        print(f"  ✓ High confidence: {drink_pred['confidence']:.1%}")
        print(f"  ✓ Strong evidence: {drink_pred['num_supports']} supporting compounds")
    else:
        print("  ✗ Not predicted as drinkable (or low confidence)")

    print("\n" + "=" * 80)
    print("  END OF EXAMPLE")
    print("=" * 80)


if __name__ == '__main__':
    main()
