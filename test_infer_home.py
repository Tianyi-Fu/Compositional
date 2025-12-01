#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_infer_home.py

使用 learn_home_signatures.py 生成的 home_signatures_all.json，
做一个基于“多特征空间 + 多层级 function signature”的推理测试。

流程：
  1) 读取 signatures：
       - type == "positive_core" 的记录，里面有 features 列表
       - type == "negative_killer" 的记录，里面有 feat
  2) 按特征前缀把这些特征分到不同 feature space：
       - structure  : geom_*, part_*, size_*
       - appearance : color_*, tex_*, appearance_*
       - location   : loc_*, room_*
       - material   : mat_*
       - physical   : phys_*
       - affordance : aff_*
       - other      : 其它（暂时不用于匹配）
  3) 对测试物体（只给 features）做推理：
       - 对每个功能 func，在每个 space 上算一个匹配度：
           score_space = |core_space ∩ feats_obj| / |core_space|
       - 若该 func 有结构 core 且 structure_score < STRUCTURE_GATE，
         直接丢弃这个 func（结构 gate，一票否决）。
       - 通过所有 killers：若物体包含某 func 的 killer 特征，则直接丢弃该 func。
       - 对剩下的 func，把所有参与匹配的 space 分数取平均，得到总分 score_total。
       - 按总分从高到低排序输出，并打印每个 space 的分数，构成“多层级、多视角”的解释。

注意：
  - 这里不再使用 raw objects 的频率统计，完全依赖 signatures。
  - affordance 空间目前只从 signatures 里读出来，用于解释，不参与打分（因为你说 affordance 在真实场景下是要推出来的）。
"""

import json
from collections import defaultdict
from typing import Dict, List, Any, Set, Tuple

# ====== 配置 ======

# 从 learn_home_signatures.py 输出的签名文件
SIGNATURE_JSON = "./data/home_signatures_all.json"

# feature space 的名字顺序（打印时用）
FEATURE_SPACES_ORDER = [
    "structure",
    "appearance",
    "location",
    "material",
    "physical",
    "affordance",
    "other",
]

# 结构 gate：如果某功能在 structure space 有 core 特征，
# 且在该空间的匹配度低于这个阈值，就直接否掉这个功能。
STRUCTURE_GATE = 0.5  # 你可以改成 0.6 / 0.7 做敏感性测试

# 总分阈值（非常宽，主要是方便你看排序和 per-space scores）
GLOBAL_SCORE_GATE = 0.0  # 如果想更严格可以设成 0.4 / 0.5 等

# 测试物体：可以按需修改
TEST_OBJECTS = [
    {
        "id": "paper_cup_1",
        "desc": "disposable paper cup (drink & contain, cannot heat)",
        "features": [
            "geom_cavity",
            "phys_rigid",
            "size_small",
            "loc_tabletop",
            "mat_paper",
            "tex_smooth",
            "room_kitchen",
        ],
    },
    {
        "id": "shallow_plate_1",
        "desc": "shallow plate (eat, not drink/contain/heat/cook)",
        "features": [
            "geom_flat",
            "mat_ceramic",
            "phys_rigid",
            "phys_heat_resistant",
            "size_medium",
            "loc_tabletop",
            "tex_smooth",
            "room_kitchen",
        ],
    },
    {
        "id": "thermos_1",
        "desc": "thermos (drink + contain + heat)",
        "features": [
            "geom_cavity",
            "mat_metal",
            "part_handle",
            "phys_rigid",
            "phys_heat_resistant",
            "size_small",
            "loc_tabletop",
            "tex_smooth",
            "room_kitchen",
        ],
    },
]


# ====== 工具函数：feature space 分类 ======

def feature_space_of(feat: str) -> str:
    """
    根据特征名前缀粗略划分 feature space。
    这跟你 home_objects.json 里的命名方案保持一致。
    """
    if feat.startswith("geom_") or feat.startswith("part_") or feat.startswith("size_"):
        return "structure"
    if feat.startswith("color_") or feat.startswith("tex_") or feat.startswith("appearance_"):
        return "appearance"
    if feat.startswith("loc_") or feat.startswith("room_"):
        return "location"
    if feat.startswith("mat_"):
        return "material"
    if feat.startswith("phys_"):
        return "physical"
    if feat.startswith("aff_"):
        return "affordance"
    return "other"


# ====== 读取 signatures，并按 space 重组 ======

def load_signatures(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Signature JSON must be a list.")
    return data


def build_core_and_killers_by_space(
    sig_records: List[Dict[str, Any]]
) -> Tuple[Dict[str, Dict[str, Set[str]]], Dict[str, Dict[str, Set[str]]]]:
    """
    从 home_signatures_all.json 里构建：
      - core_by_space[func][space]   = set(features)
      - killers_by_space[func][space] = set(features)
    """
    core_by_space: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    killers_by_space: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))

    for rec in sig_records:
        func = rec.get("func")
        if not func:
            continue

        rec_type = rec.get("type")
        if rec_type == "positive_core":
            feats = rec.get("features", []) or []
            for feat in feats:
                space = feature_space_of(feat)
                core_by_space[func][space].add(feat)

        elif rec_type == "negative_killer":
            feat = rec.get("feat")
            if not feat:
                continue
            space = feature_space_of(feat)
            killers_by_space[func][space].add(feat)

    return core_by_space, killers_by_space


# ====== 推理：按 space 匹配 + 结构 gate + 总分 ======

def compute_space_scores_for_func(
    feats_obj: Set[str],
    func: str,
    core_spaces: Dict[str, Set[str]],
    killers_spaces: Dict[str, Set[str]],
) -> Tuple[float, Dict[str, float]]:
    """
    对单个功能 func：
      - 先检查 killer：如有击中则直接返回 (score=-1, {}) 表示不考虑该功能
      - 再按空间计算匹配度：
          score_space = |core_space ∩ feats_obj| / |core_space|
      - 如果存在 structure core 且匹配度 < STRUCTURE_GATE，则返回 (score=-1, {})
      - 否则对所有有 core 的空间取平均作为 total_score
    """
    # 1) killer 检查
    for space, kset in killers_spaces.items():
        if feats_obj & kset:
            # 有 killer 特征命中，直接否掉
            return -1.0, {}

    per_space_scores: Dict[str, float] = {}
    has_any_core = False

    # 2) 先处理 structure（可能有 gate）
    struct_core = core_spaces.get("structure", set())
    if struct_core:
        has_any_core = True
        hits = len(struct_core & feats_obj)
        struct_score = hits / len(struct_core)
        per_space_scores["structure"] = struct_score

        if struct_score < STRUCTURE_GATE:
            # 结构匹配度太低，一票否决
            return -1.0, {}

    # 3) 再处理其它空间
    for space in FEATURE_SPACES_ORDER:
        if space == "structure":
            continue
        # 现在我们对 affordance 仅做解释，不参与打分，
        # 所以可以跳过 affordance；如果以后要用，删掉这行即可。
        if space == "affordance":
            continue

        core_feats = core_spaces.get(space, set())
        if not core_feats:
            continue

        has_any_core = True
        hits = len(core_feats & feats_obj)
        score_space = hits / len(core_feats)
        per_space_scores[space] = score_space

    if not has_any_core or not per_space_scores:
        return -1.0, {}

    # 4) 平均所有已参与的 space 得分
    total_score = sum(per_space_scores.values()) / len(per_space_scores)
    return total_score, per_space_scores


def predict_functions_for_features(
    feat_list: List[str],
    core_by_space: Dict[str, Dict[str, Set[str]]],
    killers_by_space: Dict[str, Dict[str, Set[str]]],
) -> List[Tuple[str, float, Dict[str, float]]]:
    """
    对一个物体（只给 features），用 multi-space function signatures 做推理：
      - 对每个 func 计算 total_score 和 per-space scores
      - 过滤掉 total_score <= GLOBAL_SCORE_GATE 的
      - 按 total_score 从高到低排序，返回列表：
          (func, total_score, per_space_scores)
    """
    feats_obj = set(feat_list)
    results: List[Tuple[str, float, Dict[str, float]]] = []

    for func, core_spaces in core_by_space.items():
        killers_spaces = killers_by_space.get(func, {})
        total_score, per_space_scores = compute_space_scores_for_func(
            feats_obj, func, core_spaces, killers_spaces
        )
        if total_score <= GLOBAL_SCORE_GATE:
            continue
        results.append((func, total_score, per_space_scores))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


# ====== main：打印 signatures 概览 + 测试推理 ======

def main():
    print(f"[INFO] Loading signatures from {SIGNATURE_JSON}")
    sig_records = load_signatures(SIGNATURE_JSON)

    core_by_space, killers_by_space = build_core_and_killers_by_space(sig_records)

    # 打印每个功能的多-space core/killer 概览
    print("\n[INFO] Learned core & killers by feature space:\n")

    for func in sorted(core_by_space.keys()):
        spaces = core_by_space[func]
        killers = killers_by_space.get(func, {})

        num_core_feats = sum(len(s) for s in spaces.values())
        print(f"  Function: {func}")
        print(f"    num_pos_core_feats: {num_core_feats}")

        for space in FEATURE_SPACES_ORDER:
            core_feats = sorted(spaces.get(space, set()))
            if core_feats:
                print(f"    core[{space}] ({len(core_feats)}): {core_feats}")

        if killers:
            print("    killers by space:")
            for space in FEATURE_SPACES_ORDER:
                kset = sorted(killers.get(space, set()))
                if kset:
                    print(f"      killers[{space}] ({len(kset)}): {kset}")
        else:
            print("    killers: (none)")

        print()

    # 对测试物体做推理
    print("==============================")
    print("Predictions for test objects")
    print("==============================\n")

    for obj in TEST_OBJECTS:
        oid = obj["id"]
        desc = obj.get("desc", "")
        feats = obj["features"]

        preds = predict_functions_for_features(feats, core_by_space, killers_by_space)

        print(f"Object: {oid}")
        if desc:
            print(f"  Desc     : {desc}")
        print(f"  Features : {feats}")
        if not preds:
            print("  Predicted functions: []")
        else:
            print("  Predicted functions (func, score, per-space scores):")
            for func, score, pspace in preds:
                # 为了输出稳定一点，按 FEATURE_SPACES_ORDER 排一下
                ordered_parts = []
                for space in FEATURE_SPACES_ORDER:
                    if space in pspace:
                        ordered_parts.append(f"{space}={pspace[space]:.2f}")
                per_space_str = ", ".join(ordered_parts)
                print(f"    - {func}: score={score:.2f} ({per_space_str})")
        print("----------------------------------------")


if __name__ == "__main__":
    main()
