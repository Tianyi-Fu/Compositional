#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
association_learning.py

Learn associations between compounds and functions.
For each compound, compute its strength of association with each function.
"""

from typing import List, Dict, Any, Set
from collections import defaultdict
from src.compound_learning import Compound
from src.representation import match_compound


class CompoundFunctionAssociation:
    """Represents association between a compound and a function."""

    def __init__(
        self,
        compound_id: str,
        function: str,
        support_pos: int,
        support_neg: int,
        coverage_pos: float,
        coverage_neg: float,
        association_score: float
    ):
        """
        Args:
            compound_id: ID of the compound
            function: Function name
            support_pos: Number of positive examples with this compound
            support_neg: Number of negative examples with this compound
            coverage_pos: P(compound | function=True)
            coverage_neg: P(compound | function=False)
            association_score: Association strength (e.g., coverage_pos)
        """
        self.compound_id = compound_id
        self.function = function
        self.support_pos = support_pos
        self.support_neg = support_neg
        self.coverage_pos = coverage_pos
        self.coverage_neg = coverage_neg
        self.association_score = association_score

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'compound_id': self.compound_id,
            'function': self.function,
            'support_pos': self.support_pos,
            'support_neg': self.support_neg,
            'coverage_pos': self.coverage_pos,
            'coverage_neg': self.coverage_neg,
            'association_score': self.association_score,
        }

    def __repr__(self):
        return (
            f"Association({self.compound_id} -> {self.function}): "
            f"score={self.association_score:.3f}, "
            f"cov_pos={self.coverage_pos:.3f}, cov_neg={self.coverage_neg:.3f}"
        )


def learn_compound_function_associations(
    objects: List[Dict[str, Any]],
    all_compounds: List[Compound],
    min_coverage_pos: float = 0.3,
    verbose: bool = True
) -> Dict[str, Dict[str, CompoundFunctionAssociation]]:
    """
    Learn associations between compounds and functions.

    For each (compound, function) pair:
    - Compute coverage_pos: fraction of objects with the function that also have the compound
    - Compute coverage_neg: fraction of objects without the function that have the compound
    - Association score = coverage_pos (or could use coverage_pos - coverage_neg)

    Args:
        objects: Training objects
        all_compounds: List of all compounds (L1 + L2)
        min_coverage_pos: Minimum coverage_pos to consider association meaningful
        verbose: Print progress

    Returns:
        Nested dict: associations[compound_id][function] = CompoundFunctionAssociation
    """
    if verbose:
        print("\n" + "=" * 60)
        print("ASSOCIATION LEARNING")
        print("=" * 60)

    # Build function index
    function_to_objects = defaultdict(list)
    all_functions = set()

    for obj in objects:
        for func in obj.get('functions', []):
            function_to_objects[func].append(obj)
            all_functions.add(func)

    if verbose:
        print(f"\n[Association] Total functions: {len(all_functions)}")
        print(f"[Association] Total compounds: {len(all_compounds)}")

    # Learn associations
    associations = defaultdict(dict)
    total_associations = 0

    for func in sorted(all_functions):
        pos_objects = function_to_objects[func]
        neg_objects = [obj for obj in objects if func not in obj.get('functions', [])]

        num_pos = len(pos_objects)
        num_neg = len(neg_objects)

        for compound in all_compounds:
            # Count how many pos/neg objects have this compound
            support_pos = sum(
                1 for obj in pos_objects
                if match_compound(set(obj.get('features', [])), compound)
            )
            support_neg = sum(
                1 for obj in neg_objects
                if match_compound(set(obj.get('features', [])), compound)
            )

            # Compute coverage
            coverage_pos = support_pos / num_pos if num_pos > 0 else 0.0
            coverage_neg = support_neg / num_neg if num_neg > 0 else 0.0

            # Association score (you can use different formulas)
            # Here we use coverage_pos as the primary score
            association_score = coverage_pos

            # Filter weak associations
            if coverage_pos < min_coverage_pos:
                continue

            # Create association object
            assoc = CompoundFunctionAssociation(
                compound_id=compound.id,
                function=func,
                support_pos=support_pos,
                support_neg=support_neg,
                coverage_pos=coverage_pos,
                coverage_neg=coverage_neg,
                association_score=association_score
            )

            associations[compound.id][func] = assoc
            total_associations += 1

    if verbose:
        print(f"\n[Association] Total associations learned: {total_associations}")
        print(f"[Association] Average associations per compound: "
              f"{total_associations / len(all_compounds):.2f}")

    return associations


def get_top_compounds_for_function(
    function: str,
    associations: Dict[str, Dict[str, CompoundFunctionAssociation]],
    top_k: int = 10
) -> List[CompoundFunctionAssociation]:
    """
    Get top-k compounds most associated with a function.

    Args:
        function: Function name
        associations: Association dictionary
        top_k: Number of top compounds to return

    Returns:
        List of top-k CompoundFunctionAssociation objects, sorted by score
    """
    function_associations = []

    for compound_id, func_assocs in associations.items():
        if function in func_assocs:
            function_associations.append(func_assocs[function])

    # Sort by association score
    function_associations.sort(key=lambda x: x.association_score, reverse=True)

    return function_associations[:top_k]


def get_functions_for_compound(
    compound_id: str,
    associations: Dict[str, Dict[str, CompoundFunctionAssociation]],
    min_score: float = 0.0
) -> List[CompoundFunctionAssociation]:
    """
    Get all functions associated with a compound.

    Args:
        compound_id: Compound ID
        associations: Association dictionary
        min_score: Minimum association score

    Returns:
        List of CompoundFunctionAssociation objects, sorted by score
    """
    if compound_id not in associations:
        return []

    func_assocs = [
        assoc for assoc in associations[compound_id].values()
        if assoc.association_score >= min_score
    ]

    func_assocs.sort(key=lambda x: x.association_score, reverse=True)
    return func_assocs


def build_compound_function_map(
    associations: Dict[str, Dict[str, CompoundFunctionAssociation]]
) -> Dict[str, Dict[str, float]]:
    """
    Build a simple mapping: compound_id -> {function: score}

    Useful for prediction.

    Args:
        associations: Full association dictionary

    Returns:
        Simplified mapping
    """
    compound_func_map = {}

    for compound_id, func_assocs in associations.items():
        compound_func_map[compound_id] = {
            func: assoc.association_score
            for func, assoc in func_assocs.items()
        }

    return compound_func_map
