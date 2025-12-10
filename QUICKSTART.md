# Quick Start Guide

## 安装和运行

### 1. 激活虚拟环境并安装依赖

```bash
cd pythonProject

# Windows
.venv\Scripts\activate
pip install -r requirements.txt

# Linux/Mac
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 运行训练

```bash
python experiments/train.py
```

**预期输出:**
```
============================================================
COMPOUND DISCOVERY
============================================================
[L1 Discovery] Feature space: structure (15 features)
  Size 2: Found 9 compounds
  Size 3: Found 19 compounds
  Total L1 compounds for structure: 33

[L2 Discovery] Total L2 compounds: 877

[Summary] Total L1 compounds: 85
[Summary] Total L2 compounds: 877

============================================================
ASSOCIATION LEARNING
============================================================
[Association] Total associations learned: 688

Model saved to: ./output
```

### 3. 运行评估

```bash
python experiments/evaluate.py
```

**预期输出:**
```
============================================================
SYMBOLIC METHOD EVALUATION
============================================================
[Symbolic] Top-5 Metrics:
  Accuracy@1: 0.333
  Accuracy@5: 0.750
  Avg Precision: 0.250
  Avg Recall: 0.542

============================================================
RANDOM FOREST EVALUATION
============================================================
[RF] Function 'drink':
  Samples: 48 (pos=10, neg=38)
  CV Accuracy: 0.853

[RF] Top-5 Compounds per Function:
  drink: A23(0.352), S1(0.281), ...
```

## 核心功能演示

### 查看发现的Compounds

训练后,查看 `output/compounds.json`:

```json
{
  "L1_compounds": {
    "structure": [
      {
        "id": "S1",
        "features": ["geom_cavity", "part_handle"],
        "layer": 1,
        "space": "structure",
        "pmi": 2.5,
        "support": 8
      }
    ]
  },
  "L2_compounds": [
    {
      "id": "X1",
      "components": ["S1", "M1"],
      "layer": 2,
      "space": "cross",
      "pmi": 1.8,
      "support": 5
    }
  ]
}
```

### 查看Compound-Function关联

查看 `output/associations.json`:

```json
[
  {
    "compound_id": "S1",
    "function": "drink",
    "support_pos": 8,
    "coverage_pos": 0.85,
    "association_score": 0.85
  }
]
```

### 使用训练好的模型预测

创建测试脚本 `test_prediction.py`:

```python
import pickle
from src.prediction import predict_functions

# 加载模型
with open('./output/model.pkl', 'rb') as f:
    model = pickle.load(f)

# 新对象
novel_object = {
    'features': [
        'geom_cavity',
        'mat_glass',
        'phys_rigid',
        'color_transparent',
        'loc_tabletop',
        'room_kitchen'
    ]
}

# 预测
predictions = predict_functions(
    novel_object,
    model['all_compounds'],
    model['compound_function_map'],
    aggregation='max',
    top_k=5
)

# 打印结果
print("Predicted functions:")
for pred in predictions:
    print(f"  {pred.function}: {pred.score:.3f}")
    # 显示支持的compounds
    for cid, score in pred.supporting_compounds[:2]:
        print(f"    <- Compound {cid}: {score:.3f}")
```

运行:
```bash
python test_prediction.py
```

输出:
```
Predicted functions:
  drink: 0.923
    <- Compound S1: 0.850
    <- Compound A23: 0.923
  contain: 0.901
    <- Compound S1: 0.850
```

## 调整参数

编辑 `experiments/train.py` 中的 CONFIG:

```python
CONFIG = {
    # 降低PMI阈值可以发现更多compounds(但可能质量降低)
    'l1_pmi_threshold': 0.5,  # 默认: 0.8

    # 增加min_support使compounds更稳健
    'l1_min_support': 3,      # 默认: 2

    # 控制compound大小
    'l1_max_size': 3,         # 默认: 4 (更小=更简单)

    # 预测聚合方式
    'prediction_aggregation': 'mean',  # 'max', 'mean', 'sum'
}
```

## 常见问题

### Q: Compounds太多(>1000个),训练很慢?

**A:** 提高PMI阈值和min_support:
```python
'l1_pmi_threshold': 1.5,  # 更严格
'l2_pmi_threshold': 1.0,
'l1_min_support': 3,
'l2_min_support': 3,
```

### Q: 预测准确率低(<50%)?

**A:** 可能原因:
1. 训练数据太少(至少需要30+对象)
2. PMI阈值太高,compounds太少
3. 试试不同的aggregation方法('max', 'mean', 'sum')

### Q: 如何理解某个预测?

**A:** 使用 `explain_prediction`:
```python
from src.prediction import explain_prediction

# 获取预测
predictions = predict_functions(...)

# 解释第一个预测
compound_map = {c.id: c for c in model['all_compounds']}
explanation = explain_prediction(
    predictions[0],
    compound_map,
    max_compounds=5
)
print(explanation)
```

输出:
```
Function: drink (score: 0.923)
Supporting evidence:
  1. L1[S1] (structure): geom_cavity, part_handle (contribution: 0.850)
  2. L1[A23] (affordance): aff_drink_from, aff_graspable... (contribution: 0.923)
```

## 下一步

1. **可视化Compounds**: 实现 `visualization/compound_network.py` 绘制compound层次结构
2. **混合预测**: 实现 `src/hybrid_prediction.py` 结合Symbolic和RF
3. **增量学习**: 添加新对象后重新训练特定compounds
4. **特征重要性**: 分析哪些原始特征最重要

## 获取帮助

- 查看完整文档: `README.md`
- 检查代码注释: 每个模块都有详细的docstrings
- 调试模式: 在函数调用中设置 `verbose=True`
