#!/usr/bin/env python3
"""
Structured trial-data extraction from the Bean Lab paper corpus (Change 6).

No standardized trial-data spreadsheet exists for this lab (checked directly
against the shared BeanLab filesystem before writing this — nothing under
/mnt/research/BeanLab/ resembling yield/trial records). This is the fallback:
extract numeric trial data (yield, disease scores, agronomic traits) straight
out of the 1,067-paper corpus, using the table-preserving chunks
pdf_processor.py already produces.

HONESTY NOTE — read before trusting the output database: this is a much
harder and lower-confidence problem than querying real trial data would be.
pdf_processor.py's table detection is a whitespace/pipe heuristic that
preserves a table's original line layout as one text chunk — it does NOT
know column headers, cell boundaries, or which numbers belong to which
cultivar. This script asks an LLM (Ollama, matching the rest of this
project's HPCC batch-processing pattern) to read each candidate chunk and
report back structured rows, which carries real extraction-error and
hallucination risk of its own — every extracted row is LLM-derived, not a
direct parse. Spot-check a sample of data/trial_data.db against the source
PDFs before treating it as authoritative. This does NOT have the same
trustworthiness guarantee a real lab-maintained spreadsheet would.

Pipeline:
  1. Load data/processed_chunks.json (from pdf_processor.py)
  2. Flag candidate chunks: table-shaped (mirrors pdf_processor.py's
     is_table_row heuristic) OR prose with high digit density + a trait
     keyword nearby (catches "cv. Rex yielded 2400 kg/ha at Elora in 2019"
     sentences that aren't formatted as tables at all)
  3. For each candidate, send it to Ollama with its immediate neighboring
     chunks as context (a table's caption/location/year is often in the
     TEXT chunk immediately before or after it, split apart by chunking —
     a real, known gap in the source data this script works around)
  4. Parse the LLM's JSON response, insert validated rows into SQLite
  5. Every row keeps source_doi/page/chunk_id and the raw source text, so
     you can trace any extracted number back to exactly where it came from

Usage (on HPCC with Ollama running):
    python3 extract_structured_data.py \\
        --chunks-file data/processed_chunks.json \\
        --output data/trial_data.db \\
        --log-file logs/extract_structured_data.log

Optional flags:
    --model            Ollama model (default: llama3.1:8b)
    --ollama-host      default: localhost:11434
    --limit            Cap number of candidate chunks (smoke test)
    --dry-run          Detect candidates and print counts, skip LLM calls
"""

import os
import sys
import re
import json
import time
import sqlite3
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import requests

sys.path.insert(0, str(Path(__file__).parent))
from retriever import _filename_to_doi  # same helper used by build_vector_store.py


# ── Candidate detection ─────────────────────────────────────────────────────────
# Mirrors pdf_processor.py's is_table_row() heuristic (pipe chars or 3+ space
# gaps), since table chunks keep their original per-line layout intact.

def _looks_like_table_row(line: str) -> bool:
    if "|" in line:
        return True
    return bool(re.search(r"\S\s{3,}\S", line)) and len(line.strip()) > 10


def _is_table_shaped(text: str) -> bool:
    lines = text.split("\n")
    if len(lines) < 2:
        return False
    table_like = sum(1 for l in lines if _looks_like_table_row(l))
    return table_like >= max(2, int(len(lines) * 0.4))


_TRAIT_KEYWORDS_RE = re.compile(
    r"\b(yield|kg\s*/?\s*ha|t\s*/?\s*ha|tonnes?|seed\s+weight|100-?seed\s+weight|"
    r"days?\s+to\s+(maturity|flowering)|plant\s+height|disease\s+(severity|score|rating|index)|"
    r"resistance\s+rating|protein\s+content|harvest\s+index|lodging\s+score)\b",
    re.IGNORECASE,
)


def _digit_density(text: str) -> float:
    return sum(ch.isdigit() for ch in text) / len(text) if text else 0.0


def is_extraction_candidate(text: str) -> bool:
    """
    Favors recall here (unlike gene_validator.py's extraction, which favors
    precision) — a missed candidate loses real data silently, while an
    over-included candidate just costs one wasted LLM call that returns an
    empty array. The asymmetry between the two failure modes is why this
    threshold is looser than gene_validator's.
    """
    if _is_table_shaped(text):
        return True
    return _digit_density(text) > 0.03 and bool(_TRAIT_KEYWORDS_RE.search(text))


# ── Ollama extraction ────────────────────────────────────────────────────────────

_EXTRACTION_PROMPT = """You are extracting structured trial data from a passage of an \
agricultural research paper about common bean (Phaseolus vulgaris).

{context_before}
PASSAGE TO EXTRACT FROM:
{target_text}
{context_after}

Extract every distinct data point EXPLICITLY stated in the PASSAGE TO EXTRACT FROM section \
(the context passages are provided only to help you identify the location/year/study — do \
not extract values described only in them). A valid data point needs a specific numeric \
value tied to an identifiable cultivar/variety name.

Rules:
- Only extract values explicitly stated in the text. Do not infer, average, calculate, or guess.
- If no cultivar/variety name is associated with a number, do not include it.
- "trait" should be a short label, e.g. "yield", "disease_severity", "days_to_maturity", \
"seed_weight", "plant_height", "protein_content".
- "unit" should capture the unit exactly as written (e.g. "kg/ha", "t/ha", "1-9 scale", "%", "days").
- "location" and "year" may be null if genuinely not stated anywhere in the passages above.
- If nothing qualifies, return an empty array.

Return ONLY a JSON array, no other text, no markdown code fences:
[{{"cultivar": "...", "location": "...", "year": ..., "trait": "...", "value": ..., "unit": "..."}}]"""


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
              "options": {"temperature": 0.0, "num_predict": 1024}},
        timeout=180,
    )
    r.raise_for_status()
    return r.json()["response"].strip()


def extract_rows_from_chunk(
    target: Dict, prev_chunk: Optional[Dict], next_chunk: Optional[Dict],
    model: str, host: str, logger: logging.Logger,
) -> List[Dict]:
    context_before = f"CONTEXT (passage immediately before, for location/year/study only):\n{prev_chunk['text']}\n" if prev_chunk else ""
    context_after = f"\nCONTEXT (passage immediately after, for location/year/study only):\n{next_chunk['text']}" if next_chunk else ""
    prompt = _EXTRACTION_PROMPT.format(
        context_before=context_before, target_text=target["text"], context_after=context_after,
    )

    try:
        raw = _query_ollama(prompt, model, host)
    except Exception as e:
        logger.warning(f"  LLM call failed for chunk {target.get('chunk_id')}: {e}")
        return []

    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return []
    try:
        rows = json.loads(match.group())
    except Exception as e:
        logger.warning(f"  JSON parse failed for chunk {target.get('chunk_id')}: {e}")
        return []

    if not isinstance(rows, list):
        return []

    valid = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if not row.get("cultivar") or row.get("value") is None:
            continue  # cultivar + value are the minimum for a usable row
        valid.append(row)
    return valid


# ── SQLite storage ───────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trial_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cultivar TEXT NOT NULL,
    location TEXT,
    year INTEGER,
    trait TEXT,
    value REAL,
    unit TEXT,
    source_doi TEXT,
    source_page INTEGER,
    source_chunk_id INTEGER,
    raw_context TEXT
);
CREATE INDEX IF NOT EXISTS idx_cultivar ON trial_records(cultivar);
CREATE INDEX IF NOT EXISTS idx_location ON trial_records(location);
CREATE INDEX IF NOT EXISTS idx_year ON trial_records(year);
CREATE INDEX IF NOT EXISTS idx_trait ON trial_records(trait);
"""


def init_db(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def insert_rows(conn: sqlite3.Connection, rows: List[Dict], chunk: Dict) -> int:
    doi = _filename_to_doi(chunk.get("source_file", ""))
    inserted = 0
    for row in rows:
        year = row.get("year")
        try:
            year = int(year) if year is not None else None
        except (ValueError, TypeError):
            year = None
        try:
            value = float(row["value"])
        except (ValueError, TypeError):
            continue  # unusable without a numeric value
        conn.execute(
            "INSERT INTO trial_records "
            "(cultivar, location, year, trait, value, unit, source_doi, source_page, "
            " source_chunk_id, raw_context) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                str(row["cultivar"]).strip(),
                (row.get("location") or None),
                year,
                (row.get("trait") or None),
                value,
                (row.get("unit") or None),
                doi,
                chunk.get("page_number"),
                chunk.get("chunk_id"),
                chunk.get("text", "")[:2000],
            ),
        )
        inserted += 1
    conn.commit()
    return inserted


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract structured trial data from the Bean Lab corpus")
    parser.add_argument("--chunks-file", default="data/processed_chunks.json")
    parser.add_argument("--output", default="data/trial_data.db")
    parser.add_argument("--log-file", default="logs/extract_structured_data.log")
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--ollama-host", default="localhost:11434")
    parser.add_argument("--limit", type=int, help="Cap number of candidate chunks (smoke test)")
    parser.add_argument("--dry-run", action="store_true", help="Detect candidates only, skip LLM calls")
    args = parser.parse_args()

    logger = setup_logging(args.log_file)
    logger.info("=" * 70)
    logger.info("Bean Lab Structured Data Extraction")
    logger.info(f"Chunks:  {args.chunks_file}")
    logger.info(f"Output:  {args.output}")
    logger.info(f"Model:   {args.model}")
    logger.info("=" * 70)

    with open(args.chunks_file, encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info(f"Loaded {len(chunks)} chunks")

    # Index chunks by (source_file, chunk_id) per source, in original order,
    # so we can look up each candidate's immediate neighbors for context.
    by_source: Dict[str, List[Dict]] = {}
    for c in chunks:
        by_source.setdefault(c.get("source_file", ""), []).append(c)
    for source in by_source:
        by_source[source].sort(key=lambda c: c.get("chunk_id", 0))

    candidates = []
    for source, source_chunks in by_source.items():
        for i, c in enumerate(source_chunks):
            if is_extraction_candidate(c.get("text", "")):
                prev_chunk = source_chunks[i - 1] if i > 0 else None
                next_chunk = source_chunks[i + 1] if i + 1 < len(source_chunks) else None
                candidates.append((c, prev_chunk, next_chunk))

    logger.info(f"✓ {len(candidates)} candidate chunks out of {len(chunks)} total "
                f"({len(candidates) / max(1, len(chunks)):.1%})")

    if args.limit:
        candidates = candidates[: args.limit]
        logger.info(f"Limited to {len(candidates)} candidates (--limit)")

    if args.dry_run:
        logger.info("--dry-run: skipping LLM extraction and database write")
        print(f"\n{len(candidates)} candidate chunks identified (dry run, no extraction performed)")
        return

    if not check_ollama_server(args.ollama_host):
        raise SystemExit(f"Ollama not reachable at {args.ollama_host}")

    conn = init_db(args.output)

    total_rows = 0
    chunks_with_data = 0
    for i, (target, prev_chunk, next_chunk) in enumerate(candidates, 1):
        logger.info(f"[{i}/{len(candidates)}] {target.get('source_file')} "
                    f"chunk {target.get('chunk_id')} (p.{target.get('page_number')})")
        t0 = time.time()
        rows = extract_rows_from_chunk(target, prev_chunk, next_chunk, args.model, args.ollama_host, logger)
        if rows:
            n = insert_rows(conn, rows, target)
            total_rows += n
            chunks_with_data += 1
            logger.info(f"  → {n} row(s) extracted in {time.time() - t0:.1f}s")

    conn.close()

    logger.info("=" * 70)
    logger.info(f"✓ Extraction complete: {total_rows} rows from {chunks_with_data}/"
                f"{len(candidates)} candidate chunks -> {args.output}")
    logger.info("Spot-check a sample of rows against the source PDFs before trusting "
                "this database — see this script's module docstring for why.")
    logger.info("=" * 70)

    print(f"\n✓ {total_rows} trial-data rows extracted from {chunks_with_data} chunks -> {args.output}")
    print("  Recommended: spot-check a sample against the source PDFs before relying on this data.")


if __name__ == "__main__":
    main()
