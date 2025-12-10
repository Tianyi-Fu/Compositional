#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ml_integration.py

Integrate Random Forest and Decision Trees to:
1. Evaluate compound importance
2. Generate interpretable rules
3. Provide hybrid predictions
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import cross_val_score
from collections import defaultdict

from src.compound_learning import Compound
from src.representation import encode_objects_as_vectors


class CompoundImportanceAnalyzer:
    """Analyze compound importance using Random Forest."""

    def __init__(
        self,
        all_compounds: List[Compound],
        n_estimators: int = 100,
        random_state: int = 42
    ):
        """
        Args:
            all_compounds: List of all compounds
            n_estimators: Number of trees in forest
            random_state: Random seed
        """
        self.all_compounds = all_compounds
        self.compound_names = [c.id for c in all_compounds]
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.models = {}  # function -> RandomForestClassifier

    def train_for_function(
        self,
        objects: List[Dict[str, Any]],
        target_function: str,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Train Random Forest for a specific function and return feature importances.

        Args:
            objects: Training objects
            target_function: Function to predict
            verbose: Print information

        Returns:
            Dictionary mapping compound_id to importance score
        """
        # Encode objects as compound vectors
        X = np.array(encode_objects_as_vectors(objects, self.all_compounds))

        # Extract labels
        y = np.array([
            1 if target_function in obj.get('functions', []) else 0
            for obj in objects
        ])

        # Check if we have both classes
        if len(np.unique(y)) < 2:
            if verbose:
                print(f"[Warning] Function '{target_function}' has only one class, skipping")
            return {}

        # Train Random Forest
        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            max_depth=5
        )
        rf.fit(X, y)

        # Cross-validation score
        cv_scores = cross_val_score(rf, X, y, cv=min(5, len(objects) // 2))
        avg_cv_score = cv_scores.mean()

        if verbose:
            print(f"\n[RF] Function '{target_function}':")
            print(f"  Samples: {len(objects)} (pos={y.sum()}, neg={(1-y).sum()})")
            print(f"  CV Accuracy: {avg_cv_score:.3f}")

        # Get feature importances
        importances = rf.feature_importances_
        importance_dict = {
            self.compound_names[i]: importances[i]
            for i in range(len(self.compound_names))
        }

        # Store model
        self.models[target_function] = rf

        return importance_dict

    def train_all_functions(
        self,
        objects: List[Dict[str, Any]],
        verbose: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        Train Random Forest for all functions.

        Args:
            objects: Training objects
            verbose: Print information

        Returns:
            Dictionary: function -> {compound_id: importance}
        """
        if verbose:
            print("\n" + "=" * 60)
            print("RANDOM FOREST IMPORTANCE ANALYSIS")
            print("=" * 60)

        # Collect all functions
        all_functions = set()
        for obj in objects:
            all_functions.update(obj.get('functions', []))

        # Train for each function
        all_importances = {}
        for func in sorted(all_functions):
            importances = self.train_for_function(objects, func, verbose=verbose)
            if importances:
                all_importances[func] = importances

        return all_importances

    def get_top_compounds_for_function(
        self,
        function: str,
        importances: Dict[str, float],
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get top-k most important compounds for a function.

        Args:
            function: Function name
            importances: Importance dictionary for this function
            top_k: Number of top compounds

        Returns:
            List of (compound_id, importance) tuples
        """
        items = [(cid, score) for cid, score in importances.items() if score > 0]
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:top_k]


class DecisionRuleExtractor:
    """Extract interpretable rules using Decision Trees."""

    def __init__(
        self,
        all_compounds: List[Compound],
        max_depth: int = 3,
        random_state: int = 42
    ):
        """
        Args:
            all_compounds: List of all compounds
            max_depth: Maximum depth of decision tree
            random_state: Random seed
        """
        self.all_compounds = all_compounds
        self.compound_names = [c.id for c in all_compounds]
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = {}  # function -> DecisionTreeClassifier

    def train_and_extract_rules(
        self,
        objects: List[Dict[str, Any]],
        target_function: str,
        verbose: bool = True
    ) -> str:
        """
        Train a shallow decision tree and extract rules.

        Args:
            objects: Training objects
            target_function: Function to predict
            verbose: Print information

        Returns:
            String representation of decision rules
        """
        # Encode objects
        X = np.array(encode_objects_as_vectors(objects, self.all_compounds))

        # Extract labels
        y = np.array([
            1 if target_function in obj.get('functions', []) else 0
            for obj in objects
        ])

        if len(np.unique(y)) < 2:
            return f"# Function '{target_function}' has only one class"

        # Train Decision Tree
        dt = DecisionTreeClassifier(
            max_depth=self.max_depth,
            random_state=self.random_state,
            min_samples_split=2
        )
        dt.fit(X, y)

        # Store tree
        self.trees[target_function] = dt

        # Extract rules as text
        rules = export_text(dt, feature_names=self.compound_names)

        if verbose:
            print(f"\n[DT] Decision rules for '{target_function}':")
            print(rules)

        return rules

    def extract_all_rules(
        self,
        objects: List[Dict[str, Any]],
        verbose: bool = True
    ) -> Dict[str, str]:
        """
        Extract rules for all functions.

        Args:
            objects: Training objects
            verbose: Print information

        Returns:
            Dictionary: function -> rules_text
        """
        if verbose:
            print("\n" + "=" * 60)
            print("DECISION TREE RULE EXTRACTION")
            print("=" * 60)

        # Collect all functions
        all_functions = set()
        for obj in objects:
            all_functions.update(obj.get('functions', []))

        # Extract rules for each function
        all_rules = {}
        for func in sorted(all_functions):
            rules = self.train_and_extract_rules(objects, func, verbose=verbose)
            all_rules[func] = rules

        return all_rules


class HybridPredictor:
    """Combine symbolic reasoning with Random Forest predictions."""

    def __init__(
        self,
        symbolic_predictor,
        rf_analyzer: CompoundImportanceAnalyzer,
        confidence_threshold: float = 0.8
    ):
        """
        Args:
            symbolic_predictor: Function that does symbolic prediction
            rf_analyzer: Trained CompoundImportanceAnalyzer
            confidence_threshold: RF confidence threshold to use RF prediction
        """
        self.symbolic_predictor = symbolic_predictor
        self.rf_analyzer = rf_analyzer
        self.confidence_threshold = confidence_threshold

    def predict(
        self,
        novel_object: Dict[str, Any],
        all_compounds: List[Compound]
    ) -> Tuple[List[Any], str]:
        """
        Hybrid prediction: use RF if confident, otherwise symbolic.

        Args:
            novel_object: Object to predict
            all_compounds: All compounds

        Returns:
            Tuple of (predictions, method_used)
        """
        # Encode object
        X = np.array([encode_objects_as_vectors([novel_object], all_compounds)[0]])

        # Try RF prediction for each function
        rf_predictions = {}
        for func, model in self.rf_analyzer.models.items():
            proba = model.predict_proba(X)[0]
            confidence = max(proba)
            prediction = model.predict(X)[0]

            rf_predictions[func] = {
                'prediction': prediction,
                'confidence': confidence
            }

        # Check if any RF prediction is confident enough
        high_confidence_funcs = [
            func for func, pred in rf_predictions.items()
            if pred['confidence'] >= self.confidence_threshold and pred['prediction'] == 1
        ]

        if high_confidence_funcs:
            # Use RF prediction
            return high_confidence_funcs, 'random_forest'
        else:
            # Fall back to symbolic
            symbolic_preds = self.symbolic_predictor(novel_object, all_compounds)
            return symbolic_preds, 'symbolic'
