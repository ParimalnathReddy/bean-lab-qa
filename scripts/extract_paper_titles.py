#!/usr/bin/env python3
"""
Extract real paper titles for the GraphRAG paper-graph visualization.

WHY THIS EXISTS: no stage of this pipeline has ever extracted a human-
readable paper title anywhere — pdf_processor.py's chunk metadata has
source_file/page_number/section but no title field, and GraphRAG's own
"title" field (used by scripts/build_paper_graph.py) is literally just the
ingested filename, e.g. "10.2135_cropsci1971.0011183X001100030039x.txt".
That's what the graph's node labels showed before this script existed.

A plain heuristic (e.g. "first line of the first chunk") does not work —
checked directly, not assumed: real chunk_id=0 text looks like
"Received: 15 November 2022 Accepted: 31 January 2023 Published online:
18 April 2023 DOI: 10.1002/plr2.20289 REGISTRATION Cultivar Registration
of 'USDA Rattler' pinto bean Phillip N. Miklas1..." — the actual title is
run together with dates/DOI/author metadata, not on its own line. Hence
an LLM call per paper (Ollama, matching this project's established
HPCC batch-processing pattern from extract_structured_data.py), not a
regex.

Pipeline:
  1. Load data/processed_chunks.json, group by source_file
  2. Take the LOWEST chunk_id chunk per paper (chunk_id is a per-paper
     counter starting at 0 — see pdf_processor.py's own docs on this) —
     that's the chunk containing the title/authors/abstract opening
  3. Ask Ollama to extract just the title from the first ~800 chars
  4. Save {source_file: title} to data/paper_titles.json

Usage (on HPCC with Ollama running):
    python3 scripts/extract_paper_titles.py \\
        --chunks-file data/processed_chunks.json \\
        --output data/paper_titles.json \\
        --log-file logs/extract_paper_titles.log

Optional flags:
    --model        Ollama model (default: llama3.1:8b, matches the rest
                    of this project's Ollama-based extraction scripts)
    --ollama-host  default: localhost:11434
    --limit        Cap number of papers processed (smoke test)
    --dry-run      Just report how many unique papers would be processed

Next step after running this: re-run scripts/build_paper_graph.py (it
picks up data/paper_titles.json automatically if present) and re-upload
paper_graph.json to the HF dataset repo.
"""

import argparse
import json
import logging
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

_TITLE_PROMPT = """Below is the raw extracted text from the start of an academic paper \
(it may include received/accepted dates, a DOI, author names, and affiliations mixed \
in with the title — this is normal PDF-extraction noise, not an error).

Extract ONLY the paper's title. Return just the title text on its own, with no quotes, \
no explanation, no author names, no journal name, and no trailing period unless the \
title itself ends with one. If you genuinely cannot find a title in this text, return \
exactly: UNKNOWN

TEXT:
{text}

TITLE:"""


def setup_logging(log_file: str) -> logging.Logger:
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def check_ollama_server(host: str) -> bool:
    try:
        return requests.get(f"http://{host}/api/tags", timeout=5).status_code == 200
    except Exception:
        return False


def _query_ollama(prompt: str, model: str, host: str) -> str:
    r = requests.post(
        f"http://{host}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False,
              "options": {"temperature": 0.0, "num_predict": 80}},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["response"].strip()


def _clean_title(raw: str) -> str:
    t = raw.strip().strip('"').strip("'").strip()
    t = re.sub(r"^(title\s*:\s*)", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t)
    return t[:200]


def extract_title(first_chunk_text: str, model: str, host: str, logger: logging.Logger) -> str:
    excerpt = first_chunk_text[:800]
    prompt = _TITLE_PROMPT.format(text=excerpt)
    try:
        raw = _query_ollama(prompt, model, host)
    except Exception as e:
        logger.warning(f"  LLM call failed: {e}")
        return ""
    title = _clean_title(raw)
    if not title or title.upper() == "UNKNOWN":
        return ""
    return title


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chunks-file", default="data/processed_chunks.json")
    parser.add_argument("--output", default="data/paper_titles.json")
    parser.add_argument("--log-file", default="logs/extract_paper_titles.log")
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--ollama-host", default="localhost:11434")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logger = setup_logging(args.log_file)

    logger.info(f"Loading {args.chunks_file} ...")
    with open(args.chunks_file, encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info(f"  {len(chunks)} chunks total")

    by_paper = defaultdict(list)
    for c in chunks:
        by_paper[c["source_file"]].append(c)

    first_chunks = {}
    for src, clist in by_paper.items():
        clist.sort(key=lambda c: c["chunk_id"])
        first_chunks[src] = clist[0]["text"]

    papers = sorted(first_chunks.keys())
    if args.limit:
        papers = papers[: args.limit]
    logger.info(f"  {len(papers)} unique papers to process" + (" (--limit applied)" if args.limit else ""))

    if args.dry_run:
        logger.info("--dry-run: stopping before any LLM calls")
        return

    if not check_ollama_server(args.ollama_host):
        raise SystemExit(f"Ollama not reachable at {args.ollama_host}")

    titles = {}
    failed = []
    start = time.time()
    for i, src in enumerate(papers):
        title = extract_title(first_chunks[src], args.model, args.ollama_host, logger)
        if title:
            titles[src] = title
        else:
            failed.append(src)
        if (i + 1) % 50 == 0 or i == len(papers) - 1:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            logger.info(f"  [{i+1}/{len(papers)}] {rate:.2f} papers/s, {len(failed)} failed so far")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(titles, f, ensure_ascii=False, indent=2)

    logger.info(f"\n✓ Extracted {len(titles)}/{len(papers)} titles -> {args.output}")
    if failed:
        logger.info(f"  {len(failed)} papers had no extractable title (kept as filename fallback downstream)")
    logger.info("\nSample:")
    for src in list(titles.keys())[:5]:
        logger.info(f"  {src} -> {titles[src]!r}")
    logger.info("\nNext: python3 scripts/build_paper_graph.py  (picks this file up automatically)")


if __name__ == "__main__":
    main()
