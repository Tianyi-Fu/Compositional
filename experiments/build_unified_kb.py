#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Normalize and fuse OCL + YCB into a unified object KB.
- OCL: attributes -> features, affordance -> aff_*, also used as functions
- YCB: heuristic + templated phys/geom/material tokens from object id
- Merge: by canonical name (strip numeric prefix in YCB), keep YCB-only objects to boost structural coverage
- Filtering: keep aff_ in whitelist; keep geom/phys/part/size/mat_* (no support filter to retain rare but meaningful structural tokens); drop appearance.

Outputs:
- data/unified_object_kb.json
- data/unified_object_kb.csv
"""

import csv
import json
import os
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OCL_RES = ROOT / "data" / "ocl_raw" / "data" / "resources"
YCB_ROOT = ROOT / "data" / "ycb_raw"
OUT_JSON = ROOT / "data" / "unified_object_kb.json"
OUT_CSV = ROOT / "data" / "unified_object_kb.csv"

FUNCTION_WHITELIST = {
    "drink",
    "eat",
    "cook",
    "clean",
    "cut",
    "hold",
    "carry",
    "lift",
    "push",
    "pull",
    "open",
    "close",
    "move",
    "pick",
    "put",
    "sit",
    "lie",
    "stir",
}

AFF_FEATURES_ALLOWED = {f"aff_{f}" for f in FUNCTION_WHITELIST} | {"aff_grasp"}
STRUCT_PREFIXES = ("geom_", "phys_", "part_", "size_", "mat_")


def canonical_name_from_ycb(ycb_id: str) -> str:
    """Convert YCB id like '025_mug' -> 'mug'; '070-b_colored_wood_blocks' -> 'colored_wood_blocks'."""
    name_part = ycb_id
    if "_" in ycb_id:
        parts = ycb_id.split("_", 1)
        if re.match(r"^[0-9]+(-[a-z])?$", parts[0]):
            name_part = parts[1]
    return name_part.replace("-", "_").lower()


def canonical_name_from_ocl(obj_name: str) -> str:
    return obj_name.strip().lower()


def load_ocl_objects():
    """Decode OCL pkl annotations into features/functions."""
    attr_list = json.load(open(OCL_RES / "OCL_class_attribute.json", "r", encoding="utf-8"))
    aff_list_raw = json.load(open(OCL_RES / "OCL_class_affordance.json", "r", encoding="utf-8"))
    aff_list = [entry["word"][0].replace(" ", "_").lower() for entry in aff_list_raw]

    data_files = [
        OCL_RES / "OCL_annot_train.pkl",
        OCL_RES / "OCL_annot_val.pkl",
        OCL_RES / "OCL_annot_test.pkl",
    ]

    agg = defaultdict(lambda: {"features": set(), "functions": set(), "sources": set()})
    for pkl_path in data_files:
        if not pkl_path.exists():
            continue
        with open(pkl_path, "rb") as f:
            items = pickle.load(f)
        for item in items:
            for obj in item.get("objects", []):
                cname = canonical_name_from_ocl(obj.get("obj", ""))
                if not cname:
                    continue
                rec = agg[cname]
                rec["sources"].add("ocl")
                for idx in obj.get("attr", []):
                    if 0 <= idx < len(attr_list):
                        attr_name = attr_list[idx].replace(" ", "_")
                        rec["features"].add(f"appearance_attr_{attr_name}")
                for idx in obj.get("aff", []):
                    if 0 <= idx < len(aff_list):
                        aff_name = aff_list[idx]
                        rec["features"].add(f"aff_{aff_name}")
                        if aff_name in FUNCTION_WHITELIST:
                            rec["functions"].add(aff_name)
    # drop objects without whitelisted functions
    filtered = {k: v for k, v in agg.items() if v["functions"]}
    return filtered


def template_tokens(name: str) -> set:
    """Template-based tokens to enrich YCB physical/structural features."""
    t = set()
    if any(k in name for k in ["mug", "cup", "glass"]):
        t.update({"geom_cavity", "part_handle", "phys_shape_cylindrical", "mat_ceramic"})
    if any(k in name for k in ["bowl"]):
        t.update({"geom_cavity", "phys_shape_bowl", "phys_shape_round", "mat_ceramic"})
    if "plate" in name:
        t.update({"phys_shape_flat", "phys_shape_round", "mat_ceramic"})
    if any(k in name for k in ["bottle", "pitcher", "can"]):
        t.update({"geom_cavity", "phys_shape_cylindrical", "mat_metal"})
    if any(k in name for k in ["knife", "fork", "spoon", "spatula", "wrench", "screwdriver"]):
        t.update({"phys_shape_long", "mat_metal", "size_small"})
    if any(k in name for k in ["hammer", "drill"]):
        t.update({"phys_shape_long", "mat_metal"})
    if any(k in name for k in ["block", "brick"]):
        t.update({"phys_shape_boxy", "mat_wood"})
    if any(k in name for k in ["ball", "marble"]):
        t.update({"phys_shape_round", "size_small"})
    if "clamp" in name:
        t.add("part_clamp")
    if any(k in name for k in ["foam", "sponge"]):
        t.add("mat_foam")
    return t


def heuristic_ycb_tokens(ycb_id: str) -> set:
    """Generate phys/geom tokens from YCB id keywords + templates."""
    name = canonical_name_from_ycb(ycb_id)
    tokens = set()
    # base size guess
    if any(k in name for k in ["mini", "small", "dice", "golf_ball", "marble"]):
        tokens.add("size_small")
    elif any(k in name for k in ["large", "extra_large"]):
        tokens.add("size_large")
    else:
        tokens.add("size_medium")
    # heuristic patterns
    if any(k in name for k in ["mug", "cup", "can", "bottle", "pitcher", "glass"]):
        tokens.update({"geom_cavity", "phys_shape_cylindrical"})
    if "bowl" in name:
        tokens.update({"geom_cavity", "phys_shape_bowl"})
    if "plate" in name:
        tokens.update({"phys_shape_flat", "phys_shape_round"})
    if any(k in name for k in ["knife", "spatula", "fork", "wrench", "screwdriver"]):
        tokens.add("phys_shape_long")
    if "marker" in name or "drill" in name:
        tokens.add("phys_shape_cylindrical")
    if "hammer" in name or "block" in name or "brick" in name:
        tokens.add("phys_shape_boxy")
    if "clamp" in name:
        tokens.add("part_clamp")
    if any(k in name for k in ["mug", "pitcher", "skillet", "saucepan"]):
        tokens.add("part_handle")
    if any(k in name for k in ["glass", "bottle", "wine"]):
        tokens.add("mat_glass")
    if any(k in name for k in ["can", "knife", "wrench", "screwdriver", "drill", "hammer"]):
        tokens.add("mat_metal")
    if any(k in name for k in ["bowl", "plate", "cup", "mug", "pitcher"]):
        tokens.add("mat_ceramic")
    if any(k in name for k in ["block", "wood"]):
        tokens.add("mat_wood")
    if any(k in name for k in ["sponge", "foam"]):
        tokens.add("mat_foam")
    tokens.update(template_tokens(name))
    return tokens


def load_ycb_objects():
    """Scan YCB directories and add heuristic phys/geom tokens."""
    agg = defaultdict(lambda: {"features": set(), "functions": set(), "sources": set()})
    for obj_dir in sorted(YCB_ROOT.iterdir()):
        if not obj_dir.is_dir():
            continue
        ycb_id = obj_dir.name
        cname = canonical_name_from_ycb(ycb_id)
        rec = agg[cname]
        rec["sources"].add("ycb")
        rec["features"].update(heuristic_ycb_tokens(ycb_id))
    return agg


def merge_sources(ocl_data, ycb_data):
    """Merge OCL (with functions) + YCB (feature-only)."""
    merged = defaultdict(lambda: {"features": set(), "functions": set(), "sources": set()})
    for cname, rec in ocl_data.items():
        out = merged[cname]
        out["features"].update(rec["features"])
        out["functions"].update(rec["functions"])
        out["sources"].update(rec["sources"])
    for cname, rec in ycb_data.items():
        out = merged[cname]
        out["features"].update(rec["features"])
        out["functions"].update(rec["functions"])
        out["sources"].update(rec["sources"])
    return merged


def filter_features_global(merged):
    for rec in merged.values():
        rec["features"] = {
            f for f in rec["features"] if (f in AFF_FEATURES_ALLOWED) or f.startswith(STRUCT_PREFIXES)
        }
    return merged


def save_outputs(merged):
    merged = filter_features_global(merged)
    records = []
    for cname, rec in sorted(merged.items()):
        records.append(
            {
                "object_id": cname,
                "features": sorted(rec["features"]),
                "functions": sorted(rec["functions"]),
                "sources": sorted(rec["sources"]),
            }
        )
    os.makedirs(OUT_JSON.parent, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["object_id", "features", "functions", "sources"])
        for r in records:
            writer.writerow(
                [
                    r["object_id"],
                    " ".join(r["features"]),
                    " ".join(r["functions"]),
                    " ".join(r["sources"]),
                ]
            )
    print(f"[INFO] Saved {len(records)} objects to {OUT_JSON}")
    print(f"[INFO] Saved CSV to {OUT_CSV}")


def main():
    print("[INFO] Loading OCL annotations...")
    ocl_data = load_ocl_objects()
    print(f"[INFO] OCL objects: {len(ocl_data)}")

    print("[INFO] Loading YCB objects...")
    ycb_data = load_ycb_objects()
    print(f"[INFO] YCB objects: {len(ycb_data)} (heuristic tokens)")

    merged = merge_sources(ocl_data, ycb_data)
    print(f"[INFO] Merged objects: {len(merged)}")
    save_outputs(merged)


if __name__ == "__main__":
    main()
