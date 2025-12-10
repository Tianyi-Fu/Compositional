#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conservatively enrich home_objects with basic structure/material/size tokens.
- If no size_*: add size_medium
- If no mat_*: add mat_unknown
- Heuristic geom/part tokens from object_id keywords:
    cup/bowl/bottle/glass/mug -> geom_cavity
    box/cabinet/drawer -> geom_box
    plate/tray -> geom_flat
    handle keyword -> part_handle
Output: data/home_objects_augmented.json (original is untouched).
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "home_objects.json"
OUT = ROOT / "data" / "home_objects_augmented.json"


def has_prefix(feats, prefix):
    return any(f.startswith(prefix) for f in feats)


def augment_obj(obj):
    feats = set(obj.get("features", []))
    oid = (obj.get("object_id") or obj.get("id") or "").lower()

    # size
    if not any(f.startswith("size_") for f in feats):
        # heuristic by name
        if any(k in oid for k in ["mini", "small"]):
            feats.add("size_small")
        elif any(k in oid for k in ["large", "big"]):
            feats.add("size_large")
        else:
            feats.add("size_medium")

    # material
    if not any(f.startswith("mat_") for f in feats):
        feats.add("mat_unknown")

    # geom heuristics
    if any(k in oid for k in ["cup", "bowl", "bottle", "glass", "mug"]):
        feats.add("geom_cavity")
        if "tall" in oid or "long" in oid:
            feats.add("geom_long")
    if any(k in oid for k in ["box", "cabinet", "drawer"]):
        feats.add("geom_box")
    if any(k in oid for k in ["plate", "tray"]):
        feats.add("geom_flat")

    # part
    if "handle" in oid or any("handle" in f for f in feats):
        feats.add("part_handle")

    obj["features"] = sorted(feats)
    return obj


def main():
    objs = json.load(open(SRC, "r", encoding="utf-8"))
    out = [augment_obj(dict(o)) for o in objs]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Augmented {len(out)} objects -> {OUT}")


if __name__ == "__main__":
    main()
