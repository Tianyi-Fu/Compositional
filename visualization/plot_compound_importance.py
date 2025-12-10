#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_compound_importance.py

Visualize compound importance using matplotlib.
"""

import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def plot_top_compounds(importances, all_compounds, function, top_k=15):
    """
    Plot top-k compounds for a function.

    Args:
        importances: Dictionary of compound_id -> importance
        all_compounds: List of all Compound objects
        function: Function name
        top_k: Number of top compounds to show
    """
    # Get compound map
    compound_map = {c.id: c for c in all_compounds}

    # Sort by importance
    sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_k]

    if not sorted_items:
        print(f"No importance data for function '{function}'")
        return

    # Extract data
    compound_ids = []
    importance_scores = []
    colors = []

    for cid, score in sorted_items:
        compound_ids.append(cid)
        importance_scores.append(score)

        # Color by layer
        if cid in compound_map:
            c = compound_map[cid]
            if c.layer == 1:
                colors.append('steelblue')
            else:
                colors.append('orange')
        else:
            colors.append('gray')

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))

    y_pos = np.arange(len(compound_ids))
    bars = ax.barh(y_pos, importance_scores, color=colors)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(compound_ids)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance (Random Forest)', fontsize=12)
    ax.set_title(f'Top-{top_k} Compounds for Function: {function}', fontsize=14, fontweight='bold')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', label='L1 Compounds'),
        Patch(facecolor='orange', label='L2 Compounds')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    # Add grid
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, importance_scores)):
        if cid in compound_map:
            c = compound_map[compound_ids[i]]
            # Truncate feature list
            features_str = ', '.join(sorted(list(c.features))[:3])
            if len(c.features) > 3:
                features_str += '...'
            ax.text(score + 0.002, bar.get_y() + bar.get_height()/2,
                    f'{score:.3f}',
                    va='center', fontsize=9)

    plt.tight_layout()
    return fig


def plot_compound_distribution(all_compounds):
    """
    Plot distribution of compounds by layer and space.

    Args:
        all_compounds: List of all Compound objects
    """
    # Count by layer
    layer_counts = {'L1': 0, 'L2': 0}
    space_counts = {}

    for c in all_compounds:
        if c.layer == 1:
            layer_counts['L1'] += 1
            space_counts[c.space] = space_counts.get(c.space, 0) + 1
        else:
            layer_counts['L2'] += 1

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Layer distribution
    layers = list(layer_counts.keys())
    counts = list(layer_counts.values())
    colors_layer = ['steelblue', 'orange']

    ax1.bar(layers, counts, color=colors_layer, alpha=0.8)
    ax1.set_ylabel('Number of Compounds', fontsize=12)
    ax1.set_title('Compound Distribution by Layer', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    for i, (layer, count) in enumerate(zip(layers, counts)):
        ax1.text(i, count + max(counts)*0.02, str(count), ha='center', fontsize=12, fontweight='bold')

    # Plot 2: L1 space distribution
    spaces = sorted(space_counts.keys())
    space_count_values = [space_counts[s] for s in spaces]
    colors_space = plt.cm.Set3(np.linspace(0, 1, len(spaces)))

    ax2.bar(spaces, space_count_values, color=colors_space, alpha=0.8)
    ax2.set_ylabel('Number of L1 Compounds', fontsize=12)
    ax2.set_xlabel('Feature Space', fontsize=12)
    ax2.set_title('L1 Compound Distribution by Space', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)

    for i, (space, count) in enumerate(zip(spaces, space_count_values)):
        ax2.text(i, count + max(space_count_values)*0.02, str(count), ha='center', fontsize=10)

    plt.tight_layout()
    return fig


def plot_pmi_distribution(all_compounds):
    """
    Plot PMI score distribution.

    Args:
        all_compounds: List of all Compound objects
    """
    l1_pmis = [c.pmi for c in all_compounds if c.layer == 1]
    l2_pmis = [c.pmi for c in all_compounds if c.layer == 2]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Histogram for L1
    ax1.hist(l1_pmis, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('PMI Score', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title(f'L1 Compound PMI Distribution (n={len(l1_pmis)})', fontsize=14, fontweight='bold')
    ax1.axvline(np.mean(l1_pmis), color='red', linestyle='--', label=f'Mean: {np.mean(l1_pmis):.2f}')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Histogram for L2
    if l2_pmis:
        ax2.hist(l2_pmis, bins=20, color='orange', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('PMI Score', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title(f'L2 Compound PMI Distribution (n={len(l2_pmis)})', fontsize=14, fontweight='bold')
        ax2.axvline(np.mean(l2_pmis), color='red', linestyle='--', label=f'Mean: {np.mean(l2_pmis):.2f}')
        ax2.legend()
        ax2.grid(alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No L2 Compounds', ha='center', va='center', fontsize=16, transform=ax2.transAxes)

    plt.tight_layout()
    return fig


def main():
    """Generate all visualizations."""
    print("Loading model...")
    with open('../output/model.pkl', 'rb') as f:
        model = pickle.load(f)

    all_compounds = model['all_compounds']
    print(f"Loaded {len(all_compounds)} compounds")

    # Plot 1: Compound distribution
    print("\nGenerating compound distribution plot...")
    fig1 = plot_compound_distribution(all_compounds)
    fig1.savefig('compound_distribution.png', dpi=300, bbox_inches='tight')
    print("Saved: compound_distribution.png")

    # Plot 2: PMI distribution
    print("\nGenerating PMI distribution plot...")
    fig2 = plot_pmi_distribution(all_compounds)
    fig2.savefig('pmi_distribution.png', dpi=300, bbox_inches='tight')
    print("Saved: pmi_distribution.png")

    # Plot 3-5: Top compounds for sample functions
    from src.ml_integration import CompoundImportanceAnalyzer
    import json

    print("\nLoading training data...")
    with open('../data/home_objects.json', 'r') as f:
        objects = json.load(f)

    print("Training Random Forest for importance analysis...")
    analyzer = CompoundImportanceAnalyzer(all_compounds)
    all_importances = analyzer.train_all_functions(objects, verbose=False)

    # Plot for top functions
    sample_functions = list(all_importances.keys())[:3]

    for func in sample_functions:
        print(f"\nGenerating importance plot for '{func}'...")
        fig = plot_top_compounds(all_importances[func], all_compounds, func, top_k=15)
        if fig:
            filename = f'importance_{func}.png'
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")

    print("\n" + "="*60)
    print("All visualizations generated successfully!")
    print("="*60)


if __name__ == '__main__':
    main()
