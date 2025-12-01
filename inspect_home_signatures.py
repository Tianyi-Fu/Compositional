#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
inspect_home_signatures.py

查看 learn_home_signatures.py 生成的 home_signatures_all.json，
按功能打印：
  - 正例核心特征 (positive_core)
  - 杀手特征 (negative_killer)
"""

import json
from collections import defaultdict
from typing import List, Dict, Any

SIG_PATH = "./data/home_signatures_all.json"


def load_signatures(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("signatures json must be a list")
    return data


def main():
    sigs = load_signatures(SIG_PATH)

    # 按 func 分组
    by_func: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in sigs:
        by_func[s["func"]].append(s)

    # 每个 func 打一段
    for func in sorted(by_func.keys()):
        items = by_func[func]
        print("=" * 40)
        print(f"Function: {func}")
        print("-" * 40)

        core_list = [s for s in items if s.get("type") == "positive_core"]
        killer_list = [s for s in items if s.get("type") == "negative_killer"]

        # 1) 正例核心
        if core_list:
            core = core_list[0]  # 每个 func 现在只会有一个 core
            feats = core["features"]
            support_pos = core["support_pos"]
            coverage_pos = core.get("coverage_pos", 0.0)

            print("  [Positive core]")
            print(f"    features      : {feats}")
            print(f"    support_pos   : {support_pos}")
            print(f"    coverage_pos  : {coverage_pos:.2f}")
            print(f"    per_feat_stats:")
            for st in core.get("per_feat_stats", []):
                f_name = st["feat"]
                p_pos = st.get("p_pos", 0.0)
                p_neg = st.get("p_neg", 0.0)
                print(
                    f"      - {f_name}: p_pos={p_pos:.2f}, p_neg={p_neg:.2f}"
                )
        else:
            print("  [Positive core]  (none)")

        # 2) 杀手特征
        if killer_list:
            print("  [Negative killers]")
            # 已经按 kill_score 排好，这里再按分数从大到小打印一下
            killer_list_sorted = sorted(
                killer_list, key=lambda d: -d.get("kill_score", 0.0)
            )
            for k in killer_list_sorted:
                feats = k["features"]
                p_pos = k.get("p_pos", 0.0)
                p_neg = k.get("p_neg", 0.0)
                score = k.get("kill_score", 0.0)
                print(
                    f"    - {feats[0]}: p_pos={p_pos:.2f}, p_neg={p_neg:.2f}, "
                    f"kill_score={score:.2f}"
                )
        else:
            print("  [Negative killers]  (none)")

        print()

    print("=" * 40)
    print("Done.")


if __name__ == "__main__":
    main()
