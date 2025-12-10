#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
utils.py

Core utility functions for the hierarchical feature learning system.
Includes PMI calculation and feature space classification.
"""

import math
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict


# ========== Feature Space Classification ==========

FEATURE_SPACE_MAPPING = {
    'structure': ['geom_', 'part_', 'size_'],
    'appearance': ['color_', 'tex_', 'appearance_'],
    'location': ['loc_', 'room_'],
    'material': ['mat_'],
    'physical': ['phys_'],
    'affordance': ['aff_'],
}


def get_feature_space(feature: str) -> str:
    """
    Classify a feature into its feature space based on prefix.

    Args:
        feature: Feature string (e.g., 'geom_cavity', 'mat_glass')

    Returns:
        Feature space name (e.g., 'structure', 'material', 'other')
    """
    for space, prefixes in FEATURE_SPACE_MAPPING.items():
        if any(feature.startswith(prefix) for prefix in prefixes):
            return space
    return 'other'


def group_features_by_space(features: List[str]) -> Dict[str, List[str]]:
    """
    Group a list of features by their feature spaces.

    Args:
        features: List of feature strings

    Returns:
        Dictionary mapping space name to list of features
    """
    grouped = defaultdict(list)
    for feat in features:
        space = get_feature_space(feat)
        if space != 'other':  # Filter out 'other' features
            grouped[space].append(feat)
    return dict(grouped)


# ========== Object Feature Extraction ==========

def extract_features_from_objects(
    objects: List[Dict[str, Any]]
) -> Tuple[List[Set[str]], Set[str]]:
    """
    Extract feature sets from objects.

    Args:
        objects: List of object dictionaries with 'features' field

    Returns:
        Tuple of (list of feature sets, set of all unique features)
    """
    feature_sets = []
    all_features = set()

    for obj in objects:
        feats = set(obj.get('features', []))
        feature_sets.append(feats)
        all_features.update(feats)

    return feature_sets, all_features


# ========== PMI Calculation ==========

def compute_pmi(
    features: Set[str],
    feature_sets: List[Set[str]],
    individual_probs: Dict[str, float],
    eps: float = 1e-6
) -> float:
    """
    Compute Pointwise Mutual Information (PMI) for a set of features.

    PMI(X) = log(P(X) / ∏P(xi))

    Where:
    - P(X) is the joint probability of all features appearing together
    - ∏P(xi) is the product of individual feature probabilities

    Args:
        features: Set of features to compute PMI for
        feature_sets: List of feature sets from training objects
        individual_probs: Dictionary mapping each feature to its probability
        eps: Small value to avoid log(0)

    Returns:
        PMI score (higher = stronger association)
    """
    if not features or not feature_sets:
        return 0.0

    # Compute P(X): joint probability
    support_count = sum(1 for fset in feature_sets if features.issubset(fset))
    p_joint = (support_count + eps) / len(feature_sets)

    # Compute ∏P(xi): product of individual probabilities
    p_product = 1.0
    for feat in features:
        p_feat = individual_probs.get(feat, eps)
        p_product *= p_feat

    # PMI = log(P(X) / ∏P(xi))
    if p_product <= 0:
        return 0.0

    pmi = math.log(p_joint / p_product)
    return pmi


def compute_feature_probabilities(
    feature_sets: List[Set[str]]
) -> Dict[str, float]:
    """
    Compute individual feature probabilities.

    Args:
        feature_sets: List of feature sets from objects

    Returns:
        Dictionary mapping feature to its probability P(feature)
    """
    if not feature_sets:
        return {}

    feature_counts = defaultdict(int)
    for fset in feature_sets:
        for feat in fset:
            feature_counts[feat] += 1

    n = len(feature_sets)
    return {feat: count / n for feat, count in feature_counts.items()}


def compute_support_count(
    features: Set[str],
    feature_sets: List[Set[str]]
) -> int:
    """
    Count how many objects contain all features in the set.

    Args:
        features: Set of features
        feature_sets: List of feature sets from objects

    Returns:
        Support count
    """
    return sum(1 for fset in feature_sets if features.issubset(fset))


def compute_lift(
    features: Set[str],
    feature_sets: List[Set[str]],
    individual_probs: Dict[str, float],
    eps: float = 1e-6
) -> float:
    """
    Compute lift for a feature set.

    Lift = P(X) / ∏P(xi)

    This is equivalent to exp(PMI).

    Args:
        features: Set of features
        feature_sets: List of feature sets
        individual_probs: Individual feature probabilities
        eps: Small value to avoid division by zero

    Returns:
        Lift score (>1 means positive association)
    """
    if not features or not feature_sets:
        return 0.0

    support_count = sum(1 for fset in feature_sets if features.issubset(fset))
    p_joint = support_count / len(feature_sets)

    p_product = 1.0
    for feat in features:
        p_feat = individual_probs.get(feat, eps)
        p_product *= p_feat

    if p_product <= 0:
        return 0.0

    return p_joint / p_product


# ========== Contrast Metrics ==========

def compute_contrast(
    features: Set[str],
    pos_feature_sets: List[Set[str]],
    neg_feature_sets: List[Set[str]],
    eps: float = 1e-6
) -> float:
    """
    Compute contrast between positive and negative examples.

    Contrast = P_pos(X) / P_neg(X)

    Higher values mean the feature set appears much more in positives.

    Args:
        features: Set of features
        pos_feature_sets: Feature sets from positive examples
        neg_feature_sets: Feature sets from negative examples
        eps: Small value to avoid division by zero

    Returns:
        Contrast score
    """
    if not pos_feature_sets:
        return 0.0

    pos_count = sum(1 for fset in pos_feature_sets if features.issubset(fset))
    p_pos = pos_count / len(pos_feature_sets)

    if not neg_feature_sets:
        return float('inf') if p_pos > 0 else 0.0

    neg_count = sum(1 for fset in neg_feature_sets if features.issubset(fset))
    p_neg = (neg_count + eps) / len(neg_feature_sets)

    return (p_pos + eps) / p_neg


# ========== Validation Helpers ==========

def validate_objects(objects: List[Dict[str, Any]]) -> bool:
    """
    Validate that objects have required fields.

    Args:
        objects: List of object dictionaries

    Returns:
        True if valid, raises ValueError otherwise
    """
    for i, obj in enumerate(objects):
        if 'features' not in obj:
            raise ValueError(f"Object {i} missing 'features' field")
        if 'functions' not in obj:
            raise ValueError(f"Object {i} missing 'functions' field")
        if not isinstance(obj['features'], list):
            raise ValueError(f"Object {i} 'features' must be a list")
        if not isinstance(obj['functions'], list):
            raise ValueError(f"Object {i} 'functions' must be a list")
    return True
