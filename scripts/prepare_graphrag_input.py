#!/usr/bin/env python3
"""
Prepare GraphRAG input from processed_chunks.json.

GraphRAG expects one text file per document.
We group chunks by source paper and reconstruct full text
sorted by chunk_id (which is per-paper sequential order).

Output: graphrag_workspace/input/<doi_encoded>.txt  (one per paper)
"""

import json
from pathlib import Path
from collections import defaultdict

CHUNKS_FILE = "data/processed_chunks.json"
OUTPUT_DIR  = Path("graphrag_workspace/input")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading chunks from {CHUNKS_FILE} ...")
with open(CHUNKS_FILE) as f:
    chunks = json.load(f)

print(f"Total chunks: {len(chunks)}")

# Group chunks by source paper, preserving list order (chunk_id is per-paper)
papers: dict = defaultdict(list)
for idx, c in enumerate(chunks):
    # Store (list_index, chunk_id, text, section) so we can sort correctly
    papers[c["source_file"]].append((idx, c["chunk_id"], c["text"], c.get("section", "")))

print(f"Unique papers: {len(papers)}")

written = 0
skipped = 0

for src, entries in papers.items():
    # Sort by chunk_id within each paper to restore reading order
    entries_sorted = sorted(entries, key=lambda x: x[1])

    # Build full document text, inserting section headers when section changes
    parts = []
    current_section = None
    for _, _, text, section in entries_sorted:
        if section and section != current_section:
            parts.append(f"\n[SECTION: {section.upper()}]\n")
            current_section = section
        parts.append(text)

    full_text = "\n\n".join(parts).strip()

    if not full_text:
        skipped += 1
        continue

    # Reconstruct DOI from filename: "10.2135_cropsci2004.1799.pdf" -> "10.2135/cropsci2004.1799"
    name_no_ext = src.replace(".pdf", "")
    parts_doi   = name_no_ext.split("_", 1)
    if len(parts_doi) == 2 and parts_doi[0].startswith("10."):
        doi = f"{parts_doi[0]}/{parts_doi[1]}"
    else:
        doi = name_no_ext

    # Output filename uses the original encoded name (safe for filesystem)
    out_filename = name_no_ext + ".txt"
    out_path     = OUTPUT_DIR / out_filename

    with open(out_path, "w", encoding="utf-8") as f:
        # Prepend DOI so GraphRAG can pick it up as document metadata
        f.write(f"SOURCE_DOI: {doi}\n")
        f.write(f"SOURCE_FILE: {src}\n\n")
        f.write(full_text)

    written += 1

print(f"\nDone.")
print(f"  Written : {written} document files -> {OUTPUT_DIR}/")
print(f"  Skipped : {skipped} empty documents")

# Sanity check: show a few output files
samples = list(OUTPUT_DIR.iterdir())[:3]
for s in samples:
    size_kb = s.stat().st_size / 1024
    print(f"  {s.name}  ({size_kb:.1f} KB)")
