#!/usr/bin/env python3
"""
Export GraphRAG community_reports.parquet → community_reports.json

Run this after build_graphrag.sb completes:
    python3 scripts/export_graphrag_summaries.py

Output: graphrag_workspace/output/community_reports.json
Upload to HF Dataset repo so the HF Space can load it at startup.
"""

import json
import sys
from pathlib import Path

PROJECT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT / "graphrag_workspace" / "output"
OUT_JSON = OUTPUT_DIR / "community_reports.json"

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas not found — activate bean_llm conda env first")


def main():
    parquet_path = OUTPUT_DIR / "community_reports.parquet"
    if not parquet_path.exists():
        sys.exit(f"community_reports.parquet not found at {parquet_path}\n"
                 "Run build_graphrag.sb first.")

    print(f"Reading {parquet_path} ...")
    df = pd.read_parquet(parquet_path)
    print(f"  {len(df)} community reports, columns: {list(df.columns)}")

    # Keep only the columns needed for search + answering
    keep = ["id", "human_readable_id", "community", "level",
            "title", "summary", "findings", "full_content", "rank"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    # Sort by rank descending so the most important communities come first
    if "rank" in df.columns:
        df = df.sort_values("rank", ascending=False)

    # Normalize findings: some versions store as list-of-dicts, some as JSON string
    if "findings" in df.columns:
        def _normalize_findings(val):
            if val is None:
                return []
            if isinstance(val, str):
                try:
                    return json.loads(val)
                except Exception:
                    return [{"summary": val, "explanation": ""}]
            if isinstance(val, list):
                return val
            return []
        df["findings"] = df["findings"].apply(_normalize_findings)

    records = df.to_dict(orient="records")

    # Build a compact search index: id → {title, summary, rank, level}
    # plus the full list for LLM context
    export = {
        "meta": {
            "n_communities": len(records),
            "levels": sorted(df["level"].unique().tolist()) if "level" in df.columns else [],
        },
        "reports": records,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False, indent=2, default=str)

    size_mb = OUT_JSON.stat().st_size / 1e6
    print(f"\n✓ Exported {len(records)} reports → {OUT_JSON}  ({size_mb:.1f} MB)")
    print("\nNext: upload to HF Dataset repo")
    print("  huggingface-cli upload Parimalanath/bean-lab-vector-db \\")
    print(f"    {OUT_JSON} community_reports.json --repo-type dataset")


if __name__ == "__main__":
    main()
