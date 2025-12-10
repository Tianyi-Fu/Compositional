# Hierarchical Feature Learning System

A compositional feature learning system that discovers hierarchical compounds from object features and predicts functions (affordances).

## Overview

This system learns **interpretable, symbolic** representations by:
1. **L1 Compounds**: Discovering feature combinations within single feature spaces using PMI
2. **L2 Compounds**: Finding cross-space patterns by combining L1 compounds
3. **Function Prediction**: Learning compound-function associations for zero-shot prediction
4. **ML Integration**: Using Random Forest to validate compound importance and extract rules

## Project Structure

```
pythonProject/
├── data/
│   └── home_objects.json          # Training data (50+ household objects)
├── src/
│   ├── utils.py                   # PMI calculation and utilities
│   ├── compound_learning.py       # L1/L2 compound discovery
│   ├── representation.py          # Hierarchical object representation
│   ├── association_learning.py   # Compound-function association learning
│   ├── prediction.py              # Function prediction
│   └── ml_integration.py          # Random Forest integration
├── experiments/
│   ├── train.py                   # Training pipeline
│   └── evaluate.py                # Evaluation and comparison
├── output/                        # Generated models (created after training)
│   ├── model.pkl                  # Trained model
│   ├── compounds.json             # Discovered compounds
│   └── associations.json          # Compound-function associations
└── requirements.txt
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the System

```bash
cd pythonProject
python experiments/train.py
```

This will:
- Load objects from `data/home_objects.json`
- Discover L1 compounds within each feature space (structure, material, physical, etc.)
- Discover L2 compounds across feature spaces
- Learn compound-function associations
- Save the trained model to `output/`
- Show sample predictions

### 3. Evaluate the System

```bash
python experiments/evaluate.py
```

This will:
- Compare **Symbolic** vs **Random Forest** methods
- Show compound importance rankings
- Extract interpretable decision tree rules
- Compute accuracy, precision, and recall metrics

## Key Concepts

### Feature Spaces

Features are organized into semantic spaces:
- **Structure**: `geom_*`, `part_*`, `size_*` (e.g., geom_cavity, part_handle)
- **Material**: `mat_*` (e.g., mat_ceramic, mat_glass)
- **Physical**: `phys_*` (e.g., phys_rigid, phys_heat_resistant)
- **Appearance**: `color_*`, `tex_*` (e.g., color_white, tex_smooth)
- **Location**: `loc_*`, `room_*` (e.g., loc_tabletop, room_kitchen)
- **Affordance**: `aff_*` (e.g., aff_drink_from, aff_graspable)

### L1 Compounds

Single-space feature combinations discovered via PMI:

```python
# Example L1 compounds
S1: ['geom_cavity', 'part_handle']           # Structure space, PMI=2.5
M1: ['mat_ceramic', 'phys_heat_resistant']   # Material+Physical, PMI=1.9
```

### L2 Compounds

Cross-space compounds combining L1 compounds:

```python
# Example L2 compound
X1: {
  'components': ['S1', 'M1'],  # Combines structure and material patterns
  'PMI': 2.3
}
```

### Compound-Function Associations

Each compound votes for functions based on training data:

```python
associations = {
  'S1': {'drink': 0.85, 'contain': 0.90},  # 85% of objects with S1 are drinkable
  'M1': {'heat': 0.88, 'cook': 0.75},
}
```

## Configuration

Edit `experiments/train.py` to adjust hyperparameters:

```python
CONFIG = {
    # L1 compound discovery
    'l1_pmi_threshold': 0.8,      # Minimum PMI for L1 compounds
    'l1_min_support': 2,           # Minimum objects with compound
    'l1_max_size': 4,              # Max features per L1 compound

    # L2 compound discovery
    'l2_pmi_threshold': 0.5,       # Minimum PMI for L2 compounds
    'l2_min_support': 2,
    'l2_max_size': 3,              # Max L1 compounds per L2 compound

    # Association learning
    'min_coverage_pos': 0.3,       # Minimum coverage in positive examples

    # Prediction
    'prediction_aggregation': 'max',  # 'max', 'mean', or 'sum'
}
```

## Data Format

Your `home_objects.json` should follow this format:

```json
[
  {
    "object_id": "cup_glass_1",
    "category": "cup",
    "features": [
      "geom_cavity",
      "mat_glass",
      "phys_rigid",
      "color_transparent",
      "loc_tabletop",
      "room_kitchen",
      "aff_drink_from",
      "aff_graspable"
    ],
    "functions": [
      "drink",
      "contain"
    ]
  }
]
```

## Example Output

After training, you'll see output like:

```
[L1 Discovery] Feature space: structure (15 features)
  Size 2: Found 8 compounds
  Size 3: Found 3 compounds
  Total L1 compounds for structure: 11

[L2 Discovery] PMI threshold=0.5, min_support=2
  Total L2 compounds: 5

[Association] Total associations learned: 142
[Association] Average associations per compound: 4.2

[Sample] Object: test_glass_tumbler_1
[Sample] True functions: ['drink', 'contain']
[Sample] Top-3 Predictions:
  1. drink: 0.923
     <- S1 (structure, geom_cavity...): 0.850
  2. contain: 0.901
  3. heat: 0.456
```

## Method Comparison

The system provides three prediction methods:

1. **Symbolic** (baseline): Pure compound-based reasoning
2. **Random Forest**: ML-based prediction using compound vectors
3. **Hybrid**: RF when confident, symbolic otherwise

Example evaluation output:

```
[Symbolic] Top-5 Metrics:
  Accuracy@1: 0.742
  Accuracy@5: 0.891
  Avg Precision: 0.678
  Avg Recall: 0.723

[RF] Top-5 Compounds per Function:
  drink: S1(0.352), S2(0.281), M3(0.124)...
```

## Advanced Usage

### Custom Prediction

```python
from src.compound_learning import discover_all_compounds
from src.prediction import predict_functions
import pickle

# Load model
with open('./output/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Predict for new object
novel_object = {
    'features': ['geom_cavity', 'mat_glass', 'phys_rigid', 'color_transparent']
}

predictions = predict_functions(
    novel_object,
    model['all_compounds'],
    model['compound_function_map'],
    aggregation='max',
    top_k=5
)

for pred in predictions:
    print(f"{pred.function}: {pred.score:.3f}")
```

### Analyze Compound Importance

```python
from src.ml_integration import CompoundImportanceAnalyzer

analyzer = CompoundImportanceAnalyzer(all_compounds)
importances = analyzer.train_for_function(objects, 'drink')

# Get top compounds for 'drink'
top = analyzer.get_top_compounds_for_function('drink', importances, top_k=10)
```

## Key Algorithms

### PMI Calculation

```
PMI(X) = log(P(X) / ∏P(xi))

Where:
- P(X) = frequency of feature set X in training data
- ∏P(xi) = product of individual feature probabilities
- Higher PMI = stronger association (features co-occur more than expected by chance)
```

### Compound Discovery

1. Enumerate all k-combinations of features (k=2,3,4)
2. Compute support count and PMI for each
3. Keep combinations with:
   - Support ≥ threshold
   - PMI ≥ threshold
4. For L2, ensure components are from different spaces

### Association Learning

For each (compound, function) pair:
- Compute coverage_pos = P(compound | function=True)
- Compute coverage_neg = P(compound | function=False)
- Association score = coverage_pos

### Prediction

1. Find all compounds matching the novel object
2. Collect function votes from matched compounds
3. Aggregate votes (max/mean/sum)
4. Return top-k functions

## Citation

If you use this system, please cite:

```
Hierarchical Compositional Feature Learning for Object Affordance Prediction
```

## License

MIT License

## Contact

For questions or issues, please open a GitHub issue.
