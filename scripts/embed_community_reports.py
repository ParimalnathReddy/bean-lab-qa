#!/usr/bin/env python3
"""
Precompute community report embeddings for GraphRAG "global" mode search.

Why this exists: hf_space/app.py used to call embedder.encode() over every
community report's (title + summary) text at Space *startup*, using
BAAI/bge-large-en-v1.5 (335M params). On the free HF Space's CPU-only
container this hung indefinitely — confirmed directly from the Space's
container logs, which stalled inside that encode() call and never printed
the following "reports loaded" line, eventually hitting HF's 30-minute
unhealthy-workload timeout. Moving the encoding step here (run once, on
HPCC, with GPU) turns a 30+ minute startup hang into a file the Space just
downloads and loads.

IMPORTANT: the embedding text built below (title + " " + summary) must stay
byte-for-byte identical to how hf_space/app.py's _search_communities()
embeds the QUERY side, and to how this same construction is used elsewhere
— otherwise the precomputed vectors and future queries won't be comparable.
If you change the text construction in one place, update the other.

Run on HPCC (GPU node, bean_llm conda env) after export_graphrag_summaries.py:
    python3 scripts/embed_community_reports.py

Output: graphrag_workspace/output/community_embeddings.npy
Upload alongside community_reports.json to the HF Dataset repo:
    huggingface-cli upload Parimalanath/bean-lab-vector-db \\
        graphrag_workspace/output/community_embeddings.npy \\
        community_embeddings.npy --repo-type dataset
"""

import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT / "graphrag_workspace" / "output"
REPORTS_JSON = OUTPUT_DIR / "community_reports.json"
OUT_NPY = OUTPUT_DIR / "community_embeddings.npy"

EMBED_MODEL = "BAAI/bge-large-en-v1.5"


def main():
    if not REPORTS_JSON.exists():
        sys.exit(f"{REPORTS_JSON} not found — run export_graphrag_summaries.py first.")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit("sentence-transformers not found — activate bean_llm conda env first")

    print(f"Reading {REPORTS_JSON} ...")
    with open(REPORTS_JSON, encoding="utf-8") as f:
        data = json.load(f)
    reports = data.get("reports", [])
    if not reports:
        sys.exit("No reports found in community_reports.json")
    print(f"  {len(reports)} community reports")

    # Must match hf_space/app.py's text construction exactly — see module docstring.
    texts = [f"{r.get('title','')} {r.get('summary','')}" for r in reports]

    print(f"Loading {EMBED_MODEL} ...")
    model = SentenceTransformer(EMBED_MODEL)

    print(f"Encoding {len(texts)} report summaries ...")
    embeddings = model.encode(
        texts, normalize_embeddings=True, show_progress_bar=True, batch_size=32,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    assert embeddings.shape[0] == len(reports), (
        f"embedding count {embeddings.shape[0]} != report count {len(reports)}"
    )

    OUT_NPY.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_NPY, embeddings)

    size_mb = OUT_NPY.stat().st_size / 1e6
    print(f"\n✓ Saved {embeddings.shape} → {OUT_NPY}  ({size_mb:.1f} MB)")
    print("\nNext: upload to HF Dataset repo, alongside community_reports.json")
    print("  huggingface-cli upload Parimalanath/bean-lab-vector-db \\")
    print(f"    {OUT_NPY} community_embeddings.npy --repo-type dataset")


if __name__ == "__main__":
    main()
