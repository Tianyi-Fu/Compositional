#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge OCL annotations with home_objects.json into a unified KB.
- OCL: attributes -> appearance_attr_*; affordances -> aff_* (also functions)
- Home objects: keep features/functions as-is
- Merge key: lowercase object name (ocl 'obj' field, home_objects 'object_id')

Output:
- data/unified_ocl_home.json
"""

import json
import pickle
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OCL_RES = ROOT / "data" / "ocl_raw" / "data" / "resources"
HOME_PATH = ROOT / "data" / "home_objects.json"
OUT_JSON = ROOT / "data" / "unified_ocl_home.json"


def canonical(name: str) -> str:
    return (name or "").strip().lower()


def load_home_objects():
    objs = json.load(open(HOME_PATH, "r", encoding="utf-8"))
    out = {}
    for obj in objs:
        cname = canonical(obj.get("object_id") or obj.get("id") or "")
        if not cname:
            continue
        out[cname] = {
            "object_id": cname,
            "features": set(obj.get("features", [])),
            "functions": set(obj.get("functions", [])),
            "sources": {"home"},
        }
    return out


def load_ocl():
    attr_list = json.load(open(OCL_RES / "OCL_class_attribute.json", "r", encoding="utf-8"))
    aff_list_raw = json.load(open(OCL_RES / "OCL_class_affordance.json", "r", encoding="utf-8"))
    aff_list = [entry["word"][0].replace(" ", "_").lower() for entry in aff_list_raw]

    data_files = [
        OCL_RES / "OCL_annot_train.pkl",
        OCL_RES / "OCL_annot_val.pkl",
        OCL_RES / "OCL_annot_test.pkl",
    ]

    agg = {}
    for pkl_path in data_files:
        if not pkl_path.exists():
            continue
        with open(pkl_path, "rb") as f:
            items = pickle.load(f)
        for item in items:
            for obj in item.get("objects", []):
                cname = canonical(obj.get("obj", ""))
                if not cname:
                    continue
                rec = agg.setdefault(cname, {"object_id": cname, "features": set(), "functions": set(), "sources": {"ocl"}})
                for idx in obj.get("attr", []):
                    if 0 <= idx < len(attr_list):
                        attr_name = attr_list[idx].replace(" ", "_")
                        rec["features"].add(f"appearance_attr_{attr_name}")
                for idx in obj.get("aff", []):
                    if 0 <= idx < len(aff_list):
                        aff_name = aff_list[idx]
                        rec["features"].add(f"aff_{aff_name}")
                        rec["functions"].add(aff_name)
    return agg


def merge(home_data, ocl_data):
    merged = defaultdict(lambda: {"object_id": None, "features": set(), "functions": set(), "sources": set()})
    for cname, rec in home_data.items():
        m = merged[cname]
        m["object_id"] = cname
        m["features"].update(rec["features"])
        m["functions"].update(rec["functions"])
        m["sources"].update(rec["sources"])
    for cname, rec in ocl_data.items():
        m = merged[cname]
        m["object_id"] = cname
        m["features"].update(rec["features"])
        m["functions"].update(rec["functions"])
        m["sources"].update(rec["sources"])
    return merged


def main():
    home = load_home_objects()
    ocl = load_ocl()
    merged = merge(home, ocl)
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
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved {len(records)} objects to {OUT_JSON}")


if __name__ == "__main__":
    main()
