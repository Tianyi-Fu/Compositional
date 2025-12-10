#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compound_learning.py

Discover hierarchical compounds using PMI (Pointwise Mutual Information).
- L1 compounds: Within single feature spaces
- L2 compounds: Across multiple feature spaces
"""

import itertools
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict

from src.utils import (
    get_feature_space,
    group_features_by_space,
    extract_features_from_objects,
    compute_pmi,
    compute_feature_probabilities,
    compute_support_count,
    compute_lift,
)


class Compound:
    """Represents a feature compound."""

    def __init__(
        self,
        compound_id: str,
        features: Set[str],
        layer: int,
        space: str,
        pmi: float,
        support: int,
        components: List[str] = None
    ):
        """
        Args:
            compound_id: Unique identifier (e.g., 'S1', 'X1')
            features: Set of raw features (for L1) or empty (for L2)
            layer: 1 for L1, 2 for L2
            space: Feature space name (for L1) or 'cross' (for L2)
            pmi: PMI score
            support: Number of objects containing this compound
            components: For L2, list of L1 compound IDs
        """
        self.id = compound_id
        self.features = features
        self.layer = layer
        self.space = space
        self.pmi = pmi
        self.support = support
        self.components = components or []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'features': sorted(list(self.features)),
            'layer': self.layer,
            'space': self.space,
            'pmi': self.pmi,
            'support': self.support,
            'components': self.components,
        }

    def __repr__(self):
        if self.layer == 1:
            return f"L1[{self.id}]({self.space}): {sorted(self.features)}, PMI={self.pmi:.2f}"
        else:
            return f"L2[{self.id}]: {self.components}, PMI={self.pmi:.2f}"


# ========== L1 Compound Discovery ==========

def find_L1_compounds(
    objects: List[Dict[str, Any]],
    pmi_threshold: float = 1.0,
    min_support: int = 2,
    max_size: int = 4,
    verbose: bool = True
) -> Dict[str, List[Compound]]:
    """
    Discover L1 compounds within each feature space using PMI.

    Algorithm:
    1. Group features by space
    2. For each space:
       - Start with size-2 combinations
       - Compute PMI for each combination
       - Keep those with PMI >= threshold and support >= min_support
       - Recursively try to extend with additional features
       - Stop when max_size reached or no valid extensions

    Args:
        objects: List of training objects
        pmi_threshold: Minimum PMI score to accept a compound
        min_support: Minimum number of objects that must contain the compound
        max_size: Maximum number of features in a compound
        verbose: Print progress information

    Returns:
        Dictionary mapping space name to list of Compound objects
    """
    if verbose:
        print(f"\n[L1 Discovery] PMI threshold={pmi_threshold}, min_support={min_support}")

    # Extract features and compute probabilities
    feature_sets, all_features = extract_features_from_objects(objects)
    individual_probs = compute_feature_probabilities(feature_sets)

    # Group features by space
    all_features_by_space = defaultdict(set)
    for feat in all_features:
        space = get_feature_space(feat)
        if space != 'other':
            all_features_by_space[space].add(feat)

    # Find compounds in each space
    compounds_by_space = {}

    for space, space_features in sorted(all_features_by_space.items()):
        if verbose:
            print(f"\n[L1 Discovery] Feature space: {space} ({len(space_features)} features)")

        space_compounds = []
        compound_counter = 0

        # Try different sizes: 2, 3, ..., max_size
        for size in range(2, min(max_size + 1, len(space_features) + 1)):
            size_compounds = []

            for combo in itertools.combinations(sorted(space_features), size):
                feat_set = set(combo)

                # Check support
                support = compute_support_count(feat_set, feature_sets)
                if support < min_support:
                    continue

                # Compute PMI
                pmi = compute_pmi(feat_set, feature_sets, individual_probs)
                if pmi < pmi_threshold:
                    continue

                # Create compound
                compound_counter += 1
                compound_id = f"{space[0].upper()}{compound_counter}"
                compound = Compound(
                    compound_id=compound_id,
                    features=feat_set,
                    layer=1,
                    space=space,
                    pmi=pmi,
                    support=support
                )
                size_compounds.append(compound)

            if verbose and size_compounds:
                print(f"  Size {size}: Found {len(size_compounds)} compounds")

            space_compounds.extend(size_compounds)

        if verbose:
            print(f"  Total L1 compounds for {space}: {len(space_compounds)}")

        compounds_by_space[space] = space_compounds

    return compounds_by_space


# ========== L2 Compound Discovery ==========

def find_L2_compounds(
    objects: List[Dict[str, Any]],
    L1_compounds: Dict[str, List[Compound]],
    pmi_threshold: float = 1.0,
    min_support: int = 2,
    max_size: int = 3,
    verbose: bool = True
) -> List[Compound]:
    """
    Discover L2 compounds across feature spaces using L1 compounds.

    Algorithm:
    1. Create compound vectors for each object (which L1 compounds it has)
    2. Compute probabilities for each L1 compound
    3. Find combinations of L1 compounds with high PMI
    4. Ensure combinations are from different spaces

    Args:
        objects: List of training objects
        L1_compounds: Dictionary mapping space to list of L1 compounds
        pmi_threshold: Minimum PMI score
        min_support: Minimum support count
        max_size: Maximum number of L1 compounds to combine
        verbose: Print progress

    Returns:
        List of L2 Compound objects
    """
    if verbose:
        print(f"\n[L2 Discovery] PMI threshold={pmi_threshold}, min_support={min_support}")

    # Flatten all L1 compounds
    all_L1_compounds = []
    for space_compounds in L1_compounds.values():
        all_L1_compounds.extend(space_compounds)

    if len(all_L1_compounds) < 2:
        if verbose:
            print("  Not enough L1 compounds for L2 discovery")
        return []

    # For each object, determine which L1 compounds it matches
    compound_sets = []  # List of sets of compound IDs
    for obj in objects:
        obj_feats = set(obj.get('features', []))
        matched_compounds = set()

        for compound in all_L1_compounds:
            if compound.features.issubset(obj_feats):
                matched_compounds.add(compound.id)

        compound_sets.append(matched_compounds)

    # Compute individual L1 compound probabilities
    compound_probs = {}
    for compound in all_L1_compounds:
        count = sum(1 for cset in compound_sets if compound.id in cset)
        compound_probs[compound.id] = count / len(objects)

    # Map compound ID to compound object
    compound_map = {c.id: c for c in all_L1_compounds}

    # Find L2 compounds
    L2_compounds = []
    compound_counter = 0

    for size in range(2, min(max_size + 1, len(all_L1_compounds) + 1)):
        for combo in itertools.combinations(all_L1_compounds, size):
            # Ensure compounds are from different spaces
            spaces = [c.space for c in combo]
            if len(spaces) != len(set(spaces)):
                continue  # Skip if not all from different spaces

            # Compute support
            combo_ids = {c.id for c in combo}
            support = sum(1 for cset in compound_sets if combo_ids.issubset(cset))
            if support < min_support:
                continue

            # Compute PMI
            pmi = compute_pmi_for_compounds(
                combo_ids, compound_sets, compound_probs
            )
            if pmi < pmi_threshold:
                continue

            # Create L2 compound
            compound_counter += 1
            compound_id = f"X{compound_counter}"

            # Collect all raw features from component L1 compounds
            all_features = set()
            for c in combo:
                all_features.update(c.features)

            L2_compound = Compound(
                compound_id=compound_id,
                features=all_features,
                layer=2,
                space='cross',
                pmi=pmi,
                support=support,
                components=[c.id for c in combo]
            )
            L2_compounds.append(L2_compound)

    if verbose:
        print(f"  Total L2 compounds: {len(L2_compounds)}")

    return L2_compounds


def compute_pmi_for_compounds(
    compound_ids: Set[str],
    compound_sets: List[Set[str]],
    compound_probs: Dict[str, float],
    eps: float = 1e-6
) -> float:
    """
    Compute PMI for a set of L1 compound IDs.

    Similar to feature PMI but operates on compound space.

    Args:
        compound_ids: Set of L1 compound IDs
        compound_sets: List of compound ID sets from objects
        compound_probs: Individual compound probabilities
        eps: Small value to avoid log(0)

    Returns:
        PMI score
    """
    if not compound_ids or not compound_sets:
        return 0.0

    import math

    # Joint probability
    support = sum(1 for cset in compound_sets if compound_ids.issubset(cset))
    p_joint = (support + eps) / len(compound_sets)

    # Product of individual probabilities
    p_product = 1.0
    for cid in compound_ids:
        p_product *= compound_probs.get(cid, eps)

    if p_product <= 0:
        return 0.0

    return math.log(p_joint / p_product)


# ========== Main Discovery Function ==========

def discover_all_compounds(
    objects: List[Dict[str, Any]],
    l1_pmi_threshold: float = 1.0,
    l2_pmi_threshold: float = 1.0,
    l1_min_support: int = 2,
    l2_min_support: int = 2,
    l1_max_size: int = 4,
    l2_max_size: int = 3,
    verbose: bool = True
) -> Tuple[Dict[str, List[Compound]], List[Compound]]:
    """
    Discover both L1 and L2 compounds.

    Args:
        objects: Training objects
        l1_pmi_threshold: PMI threshold for L1
        l2_pmi_threshold: PMI threshold for L2
        l1_min_support: Minimum support for L1
        l2_min_support: Minimum support for L2
        l1_max_size: Max features per L1 compound
        l2_max_size: Max L1 compounds per L2 compound
        verbose: Print progress

    Returns:
        Tuple of (L1 compounds by space, L2 compounds list)
    """
    print("=" * 60)
    print("COMPOUND DISCOVERY")
    print("=" * 60)

    # Discover L1 compounds
    L1_compounds = find_L1_compounds(
        objects,
        pmi_threshold=l1_pmi_threshold,
        min_support=l1_min_support,
        max_size=l1_max_size,
        verbose=verbose
    )

    # Discover L2 compounds
    L2_compounds = find_L2_compounds(
        objects,
        L1_compounds,
        pmi_threshold=l2_pmi_threshold,
        min_support=l2_min_support,
        max_size=l2_max_size,
        verbose=verbose
    )

    # Summary
    total_L1 = sum(len(compounds) for compounds in L1_compounds.values())
    print(f"\n[Summary] Total L1 compounds: {total_L1}")
    print(f"[Summary] Total L2 compounds: {len(L2_compounds)}")

    return L1_compounds, L2_compounds
