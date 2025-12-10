#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
representation.py

Build hierarchical representations of objects using discovered compounds.
L0: Raw features
L1: Single-space compounds
L2: Cross-space compounds
"""

from typing import List, Dict, Any, Set
from src.compound_learning import Compound


class ObjectRepresentation:
    """Hierarchical representation of an object."""

    def __init__(
        self,
        object_id: str,
        L0_features: List[str],
        L1_compounds: List[str],
        L2_compounds: List[str]
    ):
        """
        Args:
            object_id: Unique object identifier
            L0_features: Raw feature list
            L1_compounds: List of matched L1 compound IDs
            L2_compounds: List of matched L2 compound IDs
        """
        self.object_id = object_id
        self.L0 = L0_features
        self.L1 = L1_compounds
        self.L2 = L2_compounds

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'object_id': self.object_id,
            'L0': self.L0,
            'L1': self.L1,
            'L2': self.L2,
        }

    def __repr__(self):
        return (
            f"Object({self.object_id}):\n"
            f"  L0: {len(self.L0)} features\n"
            f"  L1: {self.L1}\n"
            f"  L2: {self.L2}"
        )


def match_compound(
    obj_features: Set[str],
    compound: Compound
) -> bool:
    """
    Check if an object matches a compound.

    An object matches a compound if it contains all the compound's features.

    Args:
        obj_features: Set of features from the object
        compound: Compound to match against

    Returns:
        True if object matches compound
    """
    return compound.features.issubset(obj_features)


def represent_object(
    obj: Dict[str, Any],
    L1_compounds_by_space: Dict[str, List[Compound]],
    L2_compounds: List[Compound]
) -> ObjectRepresentation:
    """
    Create hierarchical representation for a single object.

    Args:
        obj: Object dictionary with 'features' field
        L1_compounds_by_space: L1 compounds organized by space
        L2_compounds: List of L2 compounds

    Returns:
        ObjectRepresentation instance
    """
    obj_id = obj.get('object_id', obj.get('id', 'unknown'))
    obj_features = set(obj.get('features', []))

    # L0: Raw features
    L0_features = sorted(list(obj_features))

    # L1: Match L1 compounds
    L1_matched = []
    for space, compounds in L1_compounds_by_space.items():
        for compound in compounds:
            if match_compound(obj_features, compound):
                L1_matched.append(compound.id)

    # L2: Match L2 compounds
    L2_matched = []
    for compound in L2_compounds:
        if match_compound(obj_features, compound):
            L2_matched.append(compound.id)

    return ObjectRepresentation(
        object_id=obj_id,
        L0_features=L0_features,
        L1_compounds=L1_matched,
        L2_compounds=L2_matched
    )


def represent_all_objects(
    objects: List[Dict[str, Any]],
    L1_compounds_by_space: Dict[str, List[Compound]],
    L2_compounds: List[Compound],
    verbose: bool = True
) -> List[ObjectRepresentation]:
    """
    Create hierarchical representations for all objects.

    Args:
        objects: List of object dictionaries
        L1_compounds_by_space: L1 compounds by space
        L2_compounds: List of L2 compounds
        verbose: Print progress

    Returns:
        List of ObjectRepresentation instances
    """
    if verbose:
        print("\n" + "=" * 60)
        print("OBJECT REPRESENTATION")
        print("=" * 60)

    representations = []
    for obj in objects:
        rep = represent_object(obj, L1_compounds_by_space, L2_compounds)
        representations.append(rep)

    if verbose:
        print(f"\n[Representation] Created representations for {len(representations)} objects")

        # Statistics
        avg_L1 = sum(len(r.L1) for r in representations) / len(representations)
        avg_L2 = sum(len(r.L2) for r in representations) / len(representations)
        print(f"[Representation] Average L1 compounds per object: {avg_L1:.2f}")
        print(f"[Representation] Average L2 compounds per object: {avg_L2:.2f}")

    return representations


def get_compound_vector(
    obj: Dict[str, Any],
    all_compounds: List[Compound]
) -> List[int]:
    """
    Convert object to binary vector indicating which compounds it has.

    Used for machine learning integration.

    Args:
        obj: Object dictionary
        all_compounds: Ordered list of all compounds (L1 + L2)

    Returns:
        Binary vector [0, 1, 1, 0, ...] where 1 means object has that compound
    """
    obj_features = set(obj.get('features', []))
    vector = []

    for compound in all_compounds:
        if match_compound(obj_features, compound):
            vector.append(1)
        else:
            vector.append(0)

    return vector


def encode_objects_as_vectors(
    objects: List[Dict[str, Any]],
    all_compounds: List[Compound]
) -> List[List[int]]:
    """
    Encode all objects as compound binary vectors.

    Args:
        objects: List of objects
        all_compounds: Ordered list of all compounds

    Returns:
        2D list where each row is an object's compound vector
    """
    return [get_compound_vector(obj, all_compounds) for obj in objects]


def get_all_compounds_flat(
    L1_compounds_by_space: Dict[str, List[Compound]],
    L2_compounds: List[Compound]
) -> List[Compound]:
    """
    Flatten all compounds into a single ordered list.

    Args:
        L1_compounds_by_space: L1 compounds by space
        L2_compounds: L2 compounds

    Returns:
        Ordered list: [all L1 compounds, all L2 compounds]
    """
    all_compounds = []

    # Add all L1 compounds (sorted by space name for consistency)
    for space in sorted(L1_compounds_by_space.keys()):
        all_compounds.extend(L1_compounds_by_space[space])

    # Add all L2 compounds
    all_compounds.extend(L2_compounds)

    return all_compounds
