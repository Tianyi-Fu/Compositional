#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge home_objects (with functions) + ROCS object_instance_knowledge (property symbols as features).
ROCS features prefixed with prop_*. ROCS has no functions; functions come from home_objects only.

Output: data/unified_home_rocs.json
"""

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOME_PATH = ROOT / "data" / "home_objects.json"
ROCS_PATH = ROOT / "rocs-dataset-master" / "conceptual_knowledge" / "object_instance_knowledge.json"
OUT_JSON = ROOT / "data" / "unified_home_rocs.json"


def canonical(name: str) -> str:
    return (name or "").strip().lower()


def load_home():
    objs = json.load(open(HOME_PATH, "r", encoding="utf-8"))
    out = {}
    for obj in objs:
        cname = canonical(obj.get("object_id") or obj.get("id") or "")
        if not cname:
            continue
        out[cname] = {
            "features": set(obj.get("features", [])),
            "functions": set(obj.get("functions", [])),
            "sources": {"home"},
        }
    return out


def load_rocs():
    data = json.load(open(ROCS_PATH, "r", encoding="utf-8"))
    out = {}
    for name, feats in data.items():
        cname = canonical(name)
        out[cname] = {
            "features": {f"prop_{f}" for f in feats},
            "functions": set(),
            "sources": {"rocs"},
        }
    return out


def merge(home, rocs):
    merged = defaultdict(lambda: {"features": set(), "functions": set(), "sources": set()})
    for cname, rec in home.items():
        m = merged[cname]
        m["features"] |= rec["features"]
        m["functions"] |= rec["functions"]
        m["sources"] |= rec["sources"]
    for cname, rec in rocs.items():
        m = merged[cname]
        m["features"] |= rec["features"]
        m["functions"] |= rec["functions"]
        m["sources"] |= rec["sources"]
    return merged


def main():
    home = load_home()
    rocs = load_rocs()
    merged = merge(home, rocs)
    records = []
    for cname in sorted(merged.keys()):
        rec = merged[cname]
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
