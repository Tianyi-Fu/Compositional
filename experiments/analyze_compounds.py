#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analyze_compounds.py

Analyze compound quality and reusability:
1. Compound importance ranking
2. Reusability metrics
3. Redundancy detection
4. Cross-function analysis
"""

import sys
import pickle
import json
from pathlib import Path
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml_integration import CompoundImportanceAnalyzer
from src.representation import represent_object


def analyze_compound_reusability(objects, L1_compounds, L2_compounds, all_compounds):
    """
    Analyze how compounds are reused across objects and functions.

    Returns:
        Dictionary with reusability metrics
    """
    # Track usage
    compound_object_count = defaultdict(int)
    compound_function_set = defaultdict(set)

    for obj in objects:
        rep = represent_object(obj, L1_compounds, L2_compounds)
        obj_functions = set(obj.get('functions', []))

        # Count L1 usage
        for cid in rep.L1:
            compound_object_count[cid] += 1
            compound_function_set[cid].update(obj_functions)

        # Count L2 usage
        for cid in rep.L2:
            compound_object_count[cid] += 1
            compound_function_set[cid].update(obj_functions)

    # Compute reusability score
    reusability_scores = {}
    for cid in compound_object_count:
        num_objects = compound_object_count[cid]
        num_functions = len(compound_function_set[cid])

        # Reusability = sqrt(objects * functions)
        # High if used by many objects for many functions
        reusability = (num_objects * num_functions) ** 0.5

        reusability_scores[cid] = {
            'num_objects': num_objects,
            'num_functions': num_functions,
            'reusability_score': reusability,
            'functions': list(compound_function_set[cid])
        }

    return reusability_scores


def find_redundant_compounds(all_compounds, associations, threshold=0.9):
    """
    Find potentially redundant compounds.

    Two compounds are redundant if they:
    1. Have very similar feature sets
    2. Associate with the same functions with similar scores

    Args:
        all_compounds: List of all compounds
        associations: Compound-function associations
        threshold: Similarity threshold

    Returns:
        List of redundant compound pairs
    """
    redundant_pairs = []

    # Compare L1 compounds within same space
    l1_compounds = [c for c in all_compounds if c.layer == 1]
    compounds_by_space = defaultdict(list)

    for c in l1_compounds:
        compounds_by_space[c.space].append(c)

    for space, compounds in compounds_by_space.items():
        if len(compounds) < 2:
            continue

        for i, c1 in enumerate(compounds):
            for c2 in compounds[i+1:]:
                # Compute Jaccard similarity of features
                jaccard = len(c1.features & c2.features) / len(c1.features | c2.features)

                if jaccard >= threshold:
                    # Check if they associate with similar functions
                    funcs1 = set(associations.get(c1.id, {}).keys())
                    funcs2 = set(associations.get(c2.id, {}).keys())

                    if funcs1 and funcs2:
                        func_overlap = len(funcs1 & funcs2) / max(len(funcs1), len(funcs2))

                        if func_overlap >= threshold:
                            redundant_pairs.append({
                                'compound1': c1.id,
                                'compound2': c2.id,
                                'feature_similarity': jaccard,
                                'function_overlap': func_overlap,
                                'features1': sorted(list(c1.features)),
                                'features2': sorted(list(c2.features))
                            })

    return redundant_pairs


def analyze_cross_function_patterns(all_compounds, associations):
    """
    Analyze which compounds are most versatile (associate with many functions).

    Returns:
        Dictionary with cross-function metrics
    """
    cross_function_data = {}

    for compound in all_compounds:
        cid = compound.id

        if cid not in associations:
            continue

        func_assocs = associations[cid]
        num_functions = len(func_assocs)

        # Compute average association strength
        avg_strength = sum(func_assocs.values()) / num_functions if num_functions > 0 else 0.0

        # Compute versatility score (num_functions * avg_strength)
        versatility = num_functions * avg_strength

        cross_function_data[cid] = {
            'num_functions': num_functions,
            'avg_strength': avg_strength,
            'versatility': versatility,
            'functions': list(func_assocs.keys()),
            'layer': compound.layer,
            'space': compound.space
        }

    return cross_function_data


def main():
    """Analyze compound quality."""
    print("=" * 70)
    print("COMPOUND QUALITY ANALYSIS")
    print("=" * 70)

    # Load model
    print("\n[INFO] Loading model...")
    with open('./output/model.pkl', 'rb') as f:
        model = pickle.load(f)

    L1_compounds = model['L1_compounds']
    L2_compounds = model['L2_compounds']
    all_compounds = model['all_compounds']
    associations = model['associations']

    # Load data
    print("[INFO] Loading data...")
    with open('./data/home_objects.json', 'r') as f:
        objects = json.load(f)

    print(f"[INFO] Total compounds: {len(all_compounds)}")
    print(f"  L1: {sum(len(c) for c in L1_compounds.values())}")
    print(f"  L2: {len(L2_compounds)}")

    # Analysis 1: RF Importance
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Random Forest Importance")
    print("=" * 70)

    analyzer = CompoundImportanceAnalyzer(all_compounds)
    all_importances = analyzer.train_all_functions(objects, verbose=False)

    # Aggregate importance across functions
    total_importance = defaultdict(float)
    for func, func_importances in all_importances.items():
        for cid, importance in func_importances.items():
            total_importance[cid] += importance

    # Top compounds by importance
    sorted_importance = sorted(total_importance.items(), key=lambda x: x[1], reverse=True)

    print("\nTop-15 Compounds by RF Importance:")
    compound_map = {c.id: c for c in all_compounds}

    for i, (cid, importance) in enumerate(sorted_importance[:15], 1):
        if cid in compound_map:
            c = compound_map[cid]
            features_str = ', '.join(sorted(list(c.features))[:4])
            if len(c.features) > 4:
                features_str += '...'

            # Count function associations
            num_funcs = len(associations.get(cid, {}))

            print(f"  {i:2d}. {cid:<6} (L{c.layer}, {c.space:<11}): "
                  f"importance={importance:.3f}, "
                  f"funcs={num_funcs}, "
                  f"PMI={c.pmi:.2f}")
            print(f"       Features: {features_str}")

    # Analysis 2: Reusability
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Compound Reusability")
    print("=" * 70)

    reusability_scores = analyze_compound_reusability(
        objects, L1_compounds, L2_compounds, all_compounds
    )

    sorted_reusability = sorted(
        reusability_scores.items(),
        key=lambda x: x[1]['reusability_score'],
        reverse=True
    )

    print("\nTop-15 Most Reusable Compounds:")
    for i, (cid, data) in enumerate(sorted_reusability[:15], 1):
        if cid in compound_map:
            c = compound_map[cid]
            print(f"  {i:2d}. {cid:<6} (L{c.layer}, {c.space:<11}): "
                  f"reusability={data['reusability_score']:.2f}, "
                  f"objects={data['num_objects']}, "
                  f"funcs={data['num_functions']}")
            print(f"       Functions: {', '.join(data['functions'][:5])}...")

    # Analysis 3: Cross-function versatility
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Cross-Function Versatility")
    print("=" * 70)

    cross_function_data = analyze_cross_function_patterns(all_compounds, associations)

    sorted_versatility = sorted(
        cross_function_data.items(),
        key=lambda x: x[1]['versatility'],
        reverse=True
    )

    print("\nTop-15 Most Versatile Compounds:")
    for i, (cid, data) in enumerate(sorted_versatility[:15], 1):
        if cid in compound_map:
            c = compound_map[cid]
            print(f"  {i:2d}. {cid:<6} (L{c.layer}, {c.space:<11}): "
                  f"versatility={data['versatility']:.2f}, "
                  f"funcs={data['num_functions']}, "
                  f"avg_strength={data['avg_strength']:.3f}")

    # Analysis 4: Redundancy detection
    print("\n" + "=" * 70)
    print("ANALYSIS 4: Redundancy Detection")
    print("=" * 70)

    redundant_pairs = find_redundant_compounds(all_compounds, associations, threshold=0.85)

    print(f"\nFound {len(redundant_pairs)} potentially redundant compound pairs")

    if redundant_pairs:
        print("\nTop-10 Redundant Pairs:")
        for i, pair in enumerate(redundant_pairs[:10], 1):
            print(f"\n  {i}. {pair['compound1']} ≈ {pair['compound2']}")
            print(f"     Feature similarity: {pair['feature_similarity']:.2f}")
            print(f"     Function overlap: {pair['function_overlap']:.2f}")
            print(f"     Features1: {pair['features1'][:3]}...")
            print(f"     Features2: {pair['features2'][:3]}...")

        print(f"\nRecommendation: Consider pruning {len(redundant_pairs)} redundant compounds")
        print("  This could reduce model size while maintaining performance")

    # Analysis 5: Layer comparison
    print("\n" + "=" * 70)
    print("ANALYSIS 5: L1 vs L2 Comparison")
    print("=" * 70)

    l1_importance = sum(importance for cid, importance in total_importance.items()
                       if cid in compound_map and compound_map[cid].layer == 1)
    l2_importance = sum(importance for cid, importance in total_importance.items()
                       if cid in compound_map and compound_map[cid].layer == 2)

    l1_count = sum(1 for c in all_compounds if c.layer == 1)
    l2_count = sum(1 for c in all_compounds if c.layer == 2)

    print(f"\nL1 Compounds:")
    print(f"  Count: {l1_count}")
    print(f"  Total importance: {l1_importance:.2f}")
    print(f"  Avg importance per compound: {l1_importance / l1_count:.4f}")

    print(f"\nL2 Compounds:")
    print(f"  Count: {l2_count}")
    print(f"  Total importance: {l2_importance:.2f}")
    print(f"  Avg importance per compound: {l2_importance / l2_count:.4f}")

    print(f"\nL2/L1 importance ratio: {l2_importance / l1_importance:.2f}")

    # Summary recommendations
    print("\n" + "=" * 70)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 70)

    print("\n1. Top Performers:")
    print(f"   - {sorted_importance[0][0]}: Most important by RF")
    print(f"   - {sorted_reusability[0][0]}: Most reusable")
    print(f"   - {sorted_versatility[0][0]}: Most versatile")

    print("\n2. Potential Optimizations:")
    if len(redundant_pairs) > 10:
        print(f"   - Prune {len(redundant_pairs)} redundant compounds")
    print(f"   - Focus on top-{int(len(all_compounds) * 0.3)} compounds (70% of performance)")

    print("\n3. Layer Analysis:")
    if l2_importance / l1_importance > 1.5:
        print("   - L2 compounds are significantly more important")
        print("   - Consider weighting L2 higher in predictions")
    else:
        print("   - L1 and L2 have balanced importance")
        print("   - Current equal weighting is appropriate")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
