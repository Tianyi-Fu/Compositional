#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
learn_home_signatures.py

从一个“人工标注”的小型居家对象数据集中，
为每个功能标签学习：
  1) Positive core: 在正例中高度稳定出现的单特征集合
  2) Negative killers: 在负例中常出现、在正例中几乎不出现的强排除特征

同时，打印详细调试信息：
  - 每个功能的正/负例数量
  - L0 特征数量
  - 对 2/3/4 元特征组合：
      * 总共枚举了多少个组合
      * 有多少组合满足正例支持数阈值
      * 有多少组合满足 lift >= LIFT_THRESHOLD
      * 有多少组合同时满足正/负例对比阈值 (contrast >= CONTRAST_THRESHOLD)

数据格式 (home_objects.json):
[
  {
    "object_id": "mug_ceramic_1",
    "category": "mug",
    "super_category": "kitchenware",   # 可选，没有就默认 GLOBAL
    "features": [...],
    "functions": [...]
  },
  ...
]

输出 (home_signatures_all.json):
[
  {
    "func": "drink",
    "type": "positive_core",
    "features": [...],
    "support_pos": 3,
    "coverage_pos": 0.75,
    "per_feat_stats": [
      {"feat": "geom_cavity", "p_pos": 1.0, "p_neg": 0.2},
      ...
    ]
  },
  {
    "func": "drink",
    "type": "negative_killer",
    "feat": "geom_flat",
    "p_pos": 0.0,
    "p_neg": 0.5,
    "kill_score": 500001.0
  },
  ...
]
"""

import json
import itertools
from collections import defaultdict
from typing import List, Dict, Any, Set, Tuple

# ========= 配置区：你只需要改这里 =========

# 训练数据 & 输出文件
OBJECTS_JSON = "./data/home_objects.json"
OUT_JSON = "./data/home_signatures_all.json"

# 负例范围模式：
#   "global"          : 所有对象里不含该 func 的都是负例
#   "super_category"  : 只在相同 super_category 内找负例（数据量大时更合理）
NEG_SCOPE_MODE = "global"

# combo 调试相关：
MIN_SUPPORT_COMBO = 2          # 组合在正例里至少出现多少次才算
LIFT_THRESHOLD = 2.0           # 你要的 P(S) / ∏P(x) 阈值
CONTRAST_THRESHOLD = 2.0       # P_pos(S) / P_neg(S) 的对比阈值
DEBUG_MAX_K = 4                # 最大组合维度（2 元、3 元、4 元）
ENABLE_DEBUG_COMBOS = True     # 是否打印 2/3/4 元统计信息

# Positive core 相关：
CORE_MIN_COVERAGE = 0.7        # 单特征进入 core 的最低 p_pos（比如 >=0.7）
MIN_POS_EXAMPLES = 1           # 至少多少个正例才考虑这个 func

EPS = 1e-6                     # 防止除 0

# =========================================


def load_objects(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("objects_json must be a JSON list.")
    return data


def build_func_index(objects: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """构建 function -> [objects...] 的索引。"""
    func_to_objs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for obj in objects:
        for func in obj.get("functions", []):
            func_to_objs[func].append(obj)
    return func_to_objs


def get_super_category(obj: Dict[str, Any]) -> str:
    """统一取 super_category 没有就用 'GLOBAL'。"""
    return obj.get("super_category") or "GLOBAL"


def collect_feat_stats(
    objects: List[Dict[str, Any]]
) -> Tuple[List[Set[str]], Dict[str, int], Dict[str, float]]:
    """
    对给定对象集合：
      - 返回每个对象的特征集合列表 feat_sets
      - 每个特征的计数 feat_counts
      - 每个特征的概率 feat_probs = count / len(objects)
    """
    feat_sets: List[Set[str]] = []
    feat_counts: Dict[str, int] = defaultdict(int)

    for obj in objects:
        feats = set(obj.get("features", []))
        feat_sets.append(feats)
        for f in feats:
            feat_counts[f] += 1

    n = len(objects)
    feat_probs: Dict[str, float] = {}
    if n > 0:
        feat_probs = {f: c / n for f, c in feat_counts.items()}

    return feat_sets, feat_counts, feat_probs


def support_count(S: Set[str], feat_sets: List[Set[str]]) -> int:
    """统计集合 S 在 feat_sets 里出现（作为子集）的次数。"""
    return sum(1 for feats in feat_sets if S.issubset(feats))


# ============ 组合调试部分：2/3/4 元统计 ============

def debug_combo_flow_for_func(
    func: str,
    pos_objects: List[Dict[str, Any]],
    neg_objects: List[Dict[str, Any]],
):
    """
    只做调试打印，不影响最终 core/killer 结果。

    对某个 func：
      - 在正例上计算 L0 特征和 p_pos
      - 在负例上计算 p_neg
      - 对 k=2..DEBUG_MAX_K：
          * 枚举所有 C(|F|, k) 组合
          * 统计：
              total          : 总组合数
              sup_ok         : 正例支持数 >= MIN_SUPPORT_COMBO 的组合数
              lift_ok        : 同时满足 lift >= LIFT_THRESHOLD 的组合数
              contrast_ok    : 进一步满足对比约束的组合数
    """
    if not ENABLE_DEBUG_COMBOS:
        return

    num_pos = len(pos_objects)
    num_neg = len(neg_objects)

    # 正例特征统计
    pos_feat_sets, _, feat_probs_pos = collect_feat_stats(pos_objects)
    # 负例特征统计（只要 feat_sets 就行，prob 用 sup_neg / num_neg 即可）
    neg_feat_sets, _, _ = collect_feat_stats(neg_objects)

    all_feats: Set[str] = set()
    for feats in pos_feat_sets:
        all_feats.update(feats)
    all_feats = set(sorted(all_feats))

    print(f"\n[DEBUG] ===== Function '{func}' combo mining =====")
    print(f"[DEBUG] Positives = {num_pos}, Negatives in scope = {num_neg}")
    print(f"[DEBUG] L0 feature count = {len(all_feats)}")

    if num_pos == 0 or len(all_feats) == 0:
        print("[DEBUG] No positive examples or features, skip combo debug.")
        return

    for k in range(2, DEBUG_MAX_K + 1):
        if len(all_feats) < k:
            break

        total = 0
        sup_ok = 0
        lift_ok = 0
        contrast_ok = 0

        for combo in itertools.combinations(sorted(all_feats), k):
            total += 1
            S = set(combo)

            sup_pos = support_count(S, pos_feat_sets)
            if sup_pos < MIN_SUPPORT_COMBO:
                continue
            sup_ok += 1

            p_pos_S = sup_pos / num_pos

            # 计算分母 ∏ P_pos(x)
            denom = 1.0
            valid = True
            for f_name in combo:
                p_f = feat_probs_pos.get(f_name, 0.0)
                if p_f <= 0.0:
                    valid = False
                    break
                denom *= p_f
            if not valid or denom == 0.0:
                continue

            lift = p_pos_S / denom
            if lift < LIFT_THRESHOLD:
                continue
            lift_ok += 1

            if num_neg > 0:
                sup_neg = support_count(S, neg_feat_sets)
                p_neg_S = sup_neg / num_neg
                contrast = (p_pos_S + EPS) / (p_neg_S + EPS)
                if contrast < CONTRAST_THRESHOLD:
                    continue
                contrast_ok += 1
            else:
                # 没有负例时，就把 contrast_ok 和 lift_ok 视为一样
                contrast_ok += 1

        print(
            f"[DEBUG]  k={k} : total={total}, "
            f"sup>= {MIN_SUPPORT_COMBO} -> {sup_ok}, "
            f"lift>= {LIFT_THRESHOLD} -> {lift_ok}, "
            f"lift & contrast>= {CONTRAST_THRESHOLD} -> {contrast_ok}"
        )


# ============ core + killer 学习部分 ============

def learn_core_and_killers_for_func(
    func: str,
    pos_objects: List[Dict[str, Any]],
    neg_objects: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    根据当前数据，为单个 func 学习：
      - 一个 positive_core 项（如果存在）
      - 若干 negative_killer 项
    """
    records: List[Dict[str, Any]] = []

    if len(pos_objects) < MIN_POS_EXAMPLES:
        return records

    # 正例统计
    pos_feat_sets, pos_counts, pos_probs = collect_feat_stats(pos_objects)
    num_pos = len(pos_objects)

    # 负例统计
    neg_feat_sets, neg_counts, neg_probs = collect_feat_stats(neg_objects)
    num_neg = len(neg_objects)

    # ---------- positive core: 高覆盖度的单特征 ----------
    all_feats = sorted(set(pos_counts.keys()) | set(neg_counts.keys()))

    core_feats: List[str] = []
    per_feat_stats: List[Dict[str, Any]] = []

    for feat in all_feats:
        p_pos = pos_probs.get(feat, 0.0)
        p_neg = neg_probs.get(feat, 0.0)
        per_feat_stats.append({"feat": feat, "p_pos": p_pos, "p_neg": p_neg})

        # 进入 core 的条件：在正例中出现概率足够高
        if p_pos >= CORE_MIN_COVERAGE:
            core_feats.append(feat)

    # 核心覆盖度取 core 中最小的 p_pos（如果没有 core，则为 0）
    if core_feats:
        coverage_pos = min(
            pos_probs.get(f, 0.0) for f in core_feats
        )
        # 支持数 = 满足所有 core_feats 的正例数量
        core_S = set(core_feats)
        support_pos = support_count(core_S, pos_feat_sets)

        records.append(
            {
                "func": func,
                "type": "positive_core",
                "features": core_feats,
                "layer": len(core_feats),  # 这里 layer 只是个参考值
                "support_pos": support_pos,
                "coverage_pos": coverage_pos,
                "per_feat_stats": per_feat_stats,
            }
        )

    # ---------- negative killers: 在负例中频繁、正例极少的特征 ----------
    for feat in all_feats:
        p_pos = pos_probs.get(feat, 0.0)
        p_neg = neg_probs.get(feat, 0.0)

        # killer 直觉：正例里几乎从不出现，但负例里不算太罕见
        if p_pos <= 0.0 and p_neg > 0.0:
            kill_score = (p_neg + EPS) / (p_pos + EPS)  # p_pos=0 时会非常大
            records.append(
                {
                    "func": func,
                    "type": "negative_killer",
                    "feat": feat,
                    "p_pos": p_pos,
                    "p_neg": p_neg,
                    "kill_score": kill_score,
                }
            )

    return records


def mine_all_signatures(objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    func_to_objs = build_func_index(objects)
    all_records: List[Dict[str, Any]] = []

    print(f"[INFO] Total objects: {len(objects)}")
    print(f"[INFO] Distinct functions: {len(func_to_objs)}")
    print(f"[INFO] NEG_SCOPE_MODE: {NEG_SCOPE_MODE}")

    for func in sorted(func_to_objs.keys()):
        pos_objs = func_to_objs[func]

        # 负例范围选择
        if NEG_SCOPE_MODE == "super_category":
            pos_scats = {get_super_category(o) for o in pos_objs}
            scope = [o for o in objects if get_super_category(o) in pos_scats]
        else:
            scope = list(objects)

        neg_objs = [o for o in scope if func not in o.get("functions", [])]

        print(
            f"[INFO] Func '{func}': {len(pos_objs)} positives, "
            f"{len(neg_objs)} negatives in scope."
        )

        # 先跑组合调试（只打印，不写入 JSON）
        debug_combo_flow_for_func(func, pos_objs, neg_objs)

        # 再学习 core + killers（写入 JSON）
        recs = learn_core_and_killers_for_func(func, pos_objs, neg_objs)
        all_records.extend(recs)

    print(f"[INFO] Total signatures (core + killers): {len(all_records)}")
    return all_records


def main():
    print(f"[INFO] Loading objects from {OBJECTS_JSON}")
    objects = load_objects(OBJECTS_JSON)
    records = mine_all_signatures(objects)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"[OK] Signatures written to {OUT_JSON}")


if __name__ == "__main__":
    main()
