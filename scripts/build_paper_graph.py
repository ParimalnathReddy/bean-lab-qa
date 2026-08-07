#!/usr/bin/env python3
"""
Build paper-to-paper similarity graph (Connected Papers style).

Reads GraphRAG output parquets, computes:
  - Paper nodes (964 papers) with community color, year, title
  - Paper-paper edges weighted by shared entities
  - Community metadata for the legend

Output: graphrag_workspace/output/paper_graph.json
Upload to HF Dataset so the Space can load it at startup.

Run:
    python3 scripts/build_paper_graph.py
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix

PROJECT = Path(__file__).parent.parent
OUT_DIR = PROJECT / "graphrag_workspace" / "output"

# ── Palette for top-level communities ────────────────────────────────────────
_PALETTE = [
    "#e63946","#2a9d8f","#e9c46a","#f4a261","#264653",
    "#a8dadc","#457b9d","#1d3557","#8ecae6","#219ebc",
    "#023047","#ffb703","#fb8500","#606c38","#283618",
    "#dda15e","#bc6c25","#9b2226","#ae2012","#bb3e03",
    "#ca6702","#ee9b00","#94d2bd","#0a9396","#001219",
    "#6d6875","#b5838d","#e5989b","#ffb4a2","#ffcdb2",
]

def _year_from_title(title: str) -> int | None:
    m = re.search(r"(19|20)\d{2}", title)
    return int(m.group()) if m else None


def _doi_from_id(did: str) -> str:
    """
    GraphRAG document id -> DOI, same reconstruction convention used
    everywhere else in this project (retriever.py's _filename_to_doi(),
    prepare_graphrag_input.py's SOURCE_DOI header):
    "10.2135_cropsci1971.0011183X001100030039x.txt" ->
    "10.2135/cropsci1971.0011183X001100030039x"
    """
    name_no_ext = did[:-4] if did.endswith(".txt") else did
    parts = name_no_ext.split("_", 1)
    if len(parts) == 2 and parts[0].startswith("10."):
        return f"{parts[0]}/{parts[1]}"
    return name_no_ext


def main():
    print("Loading parquet files...")
    docs  = pd.read_parquet(OUT_DIR / "documents.parquet")
    tunits = pd.read_parquet(OUT_DIR / "text_units.parquet")
    comms = pd.read_parquet(OUT_DIR / "communities.parquet")
    ents  = pd.read_parquet(OUT_DIR / "entities.parquet")

    print(f"  {len(docs)} documents, {len(ents)} entities, {len(comms)} communities")

    # ── Map entity_id → set of document ids ──────────────────────────────────
    # text_units links documents ↔ entities
    print("Building entity→document mapping...")
    entity_to_docs = defaultdict(set)
    doc_to_entities = defaultdict(set)

    for _, row in tunits.iterrows():
        doc_ids = row.get("document_ids") if row.get("document_ids") is not None else []
        ent_ids = row.get("entity_ids") if row.get("entity_ids") is not None else []
        if not isinstance(doc_ids, (list, np.ndarray)): doc_ids = []
        if not isinstance(ent_ids, (list, np.ndarray)): ent_ids = []
        for did in doc_ids:
            for eid in ent_ids:
                entity_to_docs[eid].add(did)
                doc_to_entities[did].add(eid)

    # ── Assign each document a primary community (level 0) ───────────────────
    print("Assigning papers to level-0 communities...")
    level0 = comms[comms["level"] == 0].copy()

    doc_to_community = {}
    doc_to_community_title = {}

    for _, crow in level0.iterrows():
        cid    = crow["community"]
        ctitle = crow.get("title", f"Community {cid}")
        eids   = crow.get("entity_ids")
        if not isinstance(eids, (list, np.ndarray)): eids = []
        # find docs that have entities in this community
        for eid in eids:
            for did in entity_to_docs.get(eid, set()):
                if did not in doc_to_community:
                    doc_to_community[did] = cid
                    doc_to_community_title[did] = ctitle

    # Build community color map
    comm_ids = sorted(level0["community"].unique())
    comm_color = {cid: _PALETTE[i % len(_PALETTE)] for i, cid in enumerate(comm_ids)}

    # ── Real paper titles (scripts/extract_paper_titles.py) ──────────────────
    # GraphRAG's own "title" field is just the ingested filename (see that
    # script's docstring for why a real title needs an LLM, not a heuristic).
    # Falls back to the filename-derived title if this file doesn't exist yet
    # or doesn't have an entry for a given paper — never a hard dependency.
    titles_path = PROJECT / "data" / "paper_titles.json"
    real_titles: dict = {}
    if titles_path.exists():
        with open(titles_path, encoding="utf-8") as f:
            real_titles = json.load(f)
        print(f"Loaded {len(real_titles)} extracted paper titles from {titles_path}")
    else:
        print(f"NOTE: {titles_path} not found — node labels will fall back to "
              f"filenames until scripts/extract_paper_titles.py has been run")

    # ── Build paper nodes ─────────────────────────────────────────────────────
    print("Building paper nodes...")
    doc_id_to_idx = {did: i for i, did in enumerate(docs["id"])}

    nodes = []
    for _, row in docs.iterrows():
        did          = row["id"]  # GraphRAG's own internal content-hash id —
                                   # NOT the filename (verified directly: this
                                   # is a 128-char hex string). The filename
                                   # only ever lived in the "title" column
                                   # (prepare_graphrag_input.py never set a
                                   # real title, so GraphRAG's title defaulted
                                   # to the source filename) — every filename-
                                   # derived value below (source_file for the
                                   # title lookup, year, DOI) MUST come from
                                   # fallback_title, not did, or the lookups
                                   # silently never match anything.
        fallback_title = str(row.get("title", "Unknown"))  # filename-based
        # real_titles is keyed by source_file (the original .pdf name from
        # processed_chunks.json) — fallback_title is that same name with
        # .txt instead of .pdf (see prepare_graphrag_input.py's
        # out_filename convention).
        source_file  = fallback_title[:-4] + ".pdf" if fallback_title.endswith(".txt") else fallback_title
        title        = real_titles.get(source_file) or fallback_title
        # Year and DOI are pulled from the FILENAME (fallback_title), not
        # the real extracted title or the hash id — real paper titles
        # essentially never contain their own publication year, and did is
        # a content hash with no DOI information in it at all.
        year  = _year_from_title(fallback_title)
        doi   = _doi_from_id(fallback_title)
        comm  = doc_to_community.get(did, -1)
        color = comm_color.get(comm, "#aaaaaa")
        n_entities = len(doc_to_entities.get(did, set()))

        nodes.append({
            "id":        did,
            "label":     title[:60] + ("…" if len(title) > 60 else ""),
            "title":     title,
            "doi":       doi,
            "year":      year,
            "community": comm,
            "comm_title":doc_to_community_title.get(did, "Uncategorised"),
            "color":     color,
            "n_entities": n_entities,
        })

    # ── Build paper-paper edges via shared entities ───────────────────────────
    print("Computing paper-paper shared entity counts (sparse matrix)...")
    n_docs = len(docs)
    n_ents = len(ents)
    ent_id_to_idx = {eid: i for i, eid in enumerate(ents["id"])}

    # doc × entity binary matrix
    mat = lil_matrix((n_docs, n_ents), dtype=np.float32)
    for _, row in docs.iterrows():
        di = doc_id_to_idx.get(row["id"])
        if di is None:
            continue
        for eid in doc_to_entities.get(row["id"], set()):
            ei = ent_id_to_idx.get(eid)
            if ei is not None:
                mat[di, ei] = 1.0

    mat = mat.tocsr()
    print("  Matrix built, computing similarities...")
    sim = (mat @ mat.T).toarray()  # doc × doc shared entity count

    # Normalise: Jaccard = intersection / union
    row_sums = np.array(mat.sum(axis=1)).flatten()
    for i in range(n_docs):
        for j in range(i + 1, n_docs):
            union = row_sums[i] + row_sums[j] - sim[i, j]
            sim[i, j] = sim[i, j] / union if union > 0 else 0
            sim[j, i] = sim[i, j]

    # Keep top-5 neighbours per paper + threshold ≥ 0.05
    print("  Filtering edges...")
    edges = []
    doc_ids = list(docs["id"])
    for i in range(n_docs):
        row_sim = sim[i].copy()
        row_sim[i] = 0  # no self-loops
        top5 = np.argsort(row_sim)[-5:][::-1]
        for j in top5:
            w = float(row_sim[j])
            if w >= 0.04 and i < j:
                edges.append({
                    "source": doc_ids[i],
                    "target": doc_ids[j],
                    "weight": round(w, 4),
                })

    print(f"  {len(edges)} edges kept")

    # ── Degree for node sizing ────────────────────────────────────────────────
    degree = defaultdict(int)
    for e in edges:
        degree[e["source"]] += 1
        degree[e["target"]] += 1
    for n in nodes:
        n["degree"] = degree[n["id"]]

    # ── Community legend ──────────────────────────────────────────────────────
    communities_meta = []
    for _, crow in level0.iterrows():
        cid = crow["community"]
        communities_meta.append({
            "id":    cid,
            "title": crow.get("title", f"Community {cid}"),
            "color": comm_color.get(cid, "#aaaaaa"),
            "size":  int(crow.get("size", 0)),
        })
    communities_meta.sort(key=lambda x: -x["size"])

    # ── Export ────────────────────────────────────────────────────────────────
    out = {
        "meta": {
            "n_papers":      len(nodes),
            "n_edges":       len(edges),
            "n_communities": len(communities_meta),
        },
        "nodes":       nodes,
        "edges":       edges,
        "communities": communities_meta,
    }

    out_path = OUT_DIR / "paper_graph.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, default=str)

    size_mb = out_path.stat().st_size / 1e6
    print(f"\n✓ Saved {out_path}  ({size_mb:.1f} MB)")
    print(f"  {len(nodes)} paper nodes, {len(edges)} edges, {len(communities_meta)} level-0 communities")
    print("\nNext: upload to HF Dataset")
    print("  python3 -c \"")
    print("  from huggingface_hub import HfApi; api = HfApi()")
    print(f"  api.upload_file(path_or_fileobj='{out_path}', path_in_repo='paper_graph.json',")
    print("  repo_id='Parimalanath/bean-lab-vector-db', repo_type='dataset')\"")


if __name__ == "__main__":
    main()
