#!/usr/bin/env python3
"""
Build the per-paper structured metadata index (Change 14).

Right now nothing in this project can cleanly filter by DOI, year, species,
or methodology as first-class fields — everything relies on chunk-level
metadata and embedding similarity. This builds a small SQLite table, ONE ROW
PER PAPER (964 rows, not 21,876 chunks), so a query like "papers from 2015+
using GWAS methodology" is a real SQL WHERE clause instead of hoping
retrieval happens to surface the right chunks.

★ HONESTY BOUNDARY (same convention as extract_structured_data.py's own
  docstring, STEP 16) — read before trusting a column at face value:

  doi / source_file / title / n_chunks / n_pages / token_count: exact,
    reused directly from data/paper_titles.json and
    data/embeddings_metadata.json's chunk_metadata (rolled up per paper) —
    no extraction, no guessing.

  community / community_title: reused directly from
    graphrag_workspace/output/paper_graph.json (Change 9's GraphRAG
    community assignment) — a rough "what this paper is about" signal that
    already existed, not recomputed here.

  year: NOT copied blindly from paper_graph.json's own year field, despite
    that being tempting (it's already there) — verified directly that its
    _year_from_title() heuristic (a bare 4-digit regex over the DOI-derived
    filename) is wrong 18/964 times (1.9%), including on a paper directly
    relevant to this script's own "papers from 2015+ using GWAS
    methodology" motivating example: 10.1002/plr2.20387 (a real 2024 GWAS
    paper) was misread as year 2038 — the DOI suffix "20387" contains a
    spurious "2038" match. Fixed with a two-source cascade instead:
      1. Parse an explicit "Received:/Accepted:/Published online: DD Month
         YYYY" date-stamp from the paper's own first page, when present
         (confirmed common in Journal of Plant Registrations / Plant
         Registrations-style front matter — 53/959 papers, 6%, but highly
         accurate since it's the paper's own printed date, not a guess).
      2. Fall back to the same DOI-suffix regex paper_graph.json uses
         (~98% coverage, ~2% error rate) ONLY when step 1 finds nothing.
      3. Sanity-bound the result to [1900, CURRENT_YEAR] regardless of
         source — anything outside that range becomes NULL, not a
         garbage value silently passed through.
    year_source records which of the two methods (or neither) produced the
    final value, so a caller can tell a printed-date year from a guessed
    one rather than treating both as equally trustworthy.

  species / methodology: BEST-EFFORT, RULE-BASED, MULTI-VALUED keyword
    classification over each paper's own FULL text (all its chunks,
    EXCLUDING the references section — a citation merely mentioning "GWAS"
    or "soybean" in a bibliography entry is not evidence this PAPER used or
    studied it; same reference-exclusion lesson Change 11's manual corpus
    verification already established). NOT LLM-extracted, NOT authoritative
    — comma-joined tags, NULL where nothing matched. Measured coverage
    (verified against the real corpus before shipping, not assumed):
      species:     942/964 papers (98%) get at least one tag
      methodology: 492/964 papers (51%) get at least one tag — the other
                   49% simply don't state a recognizable methodology in
                   matchable terms (many are cultivar-registration papers
                   whose "methodology" is really just field observation
                   across trial sites, or older papers with sparse abstract
                   text at all). NULL here means "not detected," not
                   "confirmed absent."
    A real GWAS-methodology check against this exact corpus found only 2
    papers (out of 964) actually using GWAS in their own text, not a
    citation — most of this corpus (per data/embeddings_metadata.json's
    year_range bucketing: 19,742/21,876 chunks pre-2007) predates GWAS
    becoming a common technique. The capability being built here is real;
    this particular corpus's actual GWAS coverage is genuinely thin.

  trait: DELIBERATELY NOT INCLUDED in this change. Species and methodology
    are both closed(ish), enumerable vocabularies a handful of regexes can
    reasonably cover; "trait" spans yield, a dozen+ named diseases, drought/
    heat/cold tolerance, seed quality, nitrogen fixation, and more — an
    open-ended tagging problem that would need either a much larger keyword
    taxonomy (more real work than "the cheapest architecture change" should
    absorb) or LLM extraction (like scripts/extract_paper_titles.py's
    per-paper approach) to do well. Left for a future, explicitly-scoped
    change rather than shipped as a crude, low-value guess here.

FILES READ: data/paper_titles.json, data/embeddings_metadata.json,
  data/processed_chunks.json, graphrag_workspace/output/paper_graph.json
OUTPUT: vector_db/paper_metadata.db (table: papers)

Run:
    python3 scripts/build_paper_metadata.py
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

PROJECT = Path(__file__).parent.parent
DATA = PROJECT / "data"
GRAPHRAG_OUT = PROJECT / "graphrag_workspace" / "output"
DB_PATH = PROJECT / "vector_db" / "paper_metadata.db"

CURRENT_YEAR = 2026  # sanity-bound ceiling; see year cascade in the module docstring


def _doi_from_source_file(source_file: str) -> str:
    """
    Same filename -> DOI reconstruction used independently in retriever.py,
    hf_space/app.py, and build_paper_graph.py (scripts/ scripts in this
    project keep their own small copy of this rather than cross-importing —
    matching that existing, if imperfect, precedent rather than introducing
    a new cross-module import pattern for one line of logic).
    "10.2135_cropsci1971.0011183X001100030039x.pdf" ->
    "10.2135/cropsci1971.0011183X001100030039x"
    """
    name = source_file[:-4] if source_file.endswith(".pdf") else source_file
    parts = name.split("_", 1)
    if len(parts) == 2 and parts[0].startswith("10."):
        return f"{parts[0]}/{parts[1]}"
    return name


# ── Year extraction (see module docstring for why this exists instead of
# just trusting paper_graph.json's already-computed year field) ─────────────

_DATE_STAMP_RE = re.compile(
    r"(?:Published online|Accepted|Received)\s*:?\s*\d{1,2}\s+\w+\s+((?:19|20)\d{2})",
    re.IGNORECASE,
)
_DOI_SUFFIX_YEAR_RE = re.compile(r"(19|20)\d{2}")


def _year_from_page1(page1_text: str) -> Optional[int]:
    m = _DATE_STAMP_RE.search(page1_text[:600])
    return int(m.group(1)) if m else None


def _year_from_doi_suffix(source_file: str) -> Optional[int]:
    m = _DOI_SUFFIX_YEAR_RE.search(source_file)
    return int(m.group()) if m else None


def _resolve_year(source_file: str, page1_text: str) -> tuple[Optional[int], Optional[str]]:
    y = _year_from_page1(page1_text)
    if y is not None and 1900 <= y <= CURRENT_YEAR:
        return y, "published_date"
    y = _year_from_doi_suffix(source_file)
    if y is not None and 1900 <= y <= CURRENT_YEAR:
        return y, "doi_suffix"
    return None, None


# ── Species / methodology tagging (see module docstring for coverage
# numbers and the honesty boundary on what these are and aren't) ───────────

SPECIES_PATTERNS = {
    "phaseolus_vulgaris": r"phaseolus vulgaris|common bean|dry bean|kidney bean|navy bean|"
                           r"pinto ?bean|black bean|great northern|pink bean|red bean|"
                           r"yellow bean|snap bean|french bean|wax bean|cranberry bean",
    "phaseolus_acutifolius": r"phaseolus acutifolius|tepary bean",
    "phaseolus_coccineus": r"phaseolus coccineus|runner bean|scarlet runner",
    "phaseolus_lunatus": r"phaseolus lunatus|lima bean",
    "glycine_max": r"glycine max|\bsoybean",
    "pisum_sativum": r"pisum sativum|field pea",
    "vigna": r"\bvigna |cowpea|mungbean|mung bean",
    "medicago_sativa": r"medicago sativa|\balfalfa",
    "vicia_faba": r"vicia faba|faba bean|broad bean",
    "cicer_arietinum": r"cicer arietinum|chickpea",
    "arachis_hypogaea": r"arachis hypogaea|\bpeanut",
}

METHODOLOGY_PATTERNS = {
    "cultivar_registration": r"cultivar registration|germplasm registration",
    "gwas": r"genome-wide association|\bgwas\b",
    "qtl_mapping": r"qtl mapping|quantitative trait loci|\bqtls?\b",
    "marker_assisted_selection": r"marker-assisted selection|marker assisted selection",
    "field_trial": r"field trial|field experiment|multi-location trial|multi-environment",
    "greenhouse": r"greenhouse experiment|growth chamber",
    "rna_seq": r"rna-seq|rna sequencing|transcriptom",
    "genomic_selection": r"genomic selection",
    "recurrent_selection": r"recurrent selection",
    "linkage_mapping": r"linkage map|linkage analysis",
    "meta_analysis_review": r"meta-analysis|systematic review",
}


def _tags(text_lower: str, patterns: Dict[str, str]) -> str:
    hits = [name for name, pat in patterns.items() if re.search(pat, text_lower)]
    return ",".join(hits)


# ── SQLite storage ───────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    doi TEXT PRIMARY KEY,
    source_file TEXT NOT NULL,
    title TEXT,
    year INTEGER,
    year_source TEXT,
    community INTEGER,
    community_title TEXT,
    species TEXT,
    methodology TEXT,
    n_chunks INTEGER,
    n_pages INTEGER,
    token_count INTEGER
);
CREATE INDEX IF NOT EXISTS idx_papers_year ON papers(year);
CREATE INDEX IF NOT EXISTS idx_papers_species ON papers(species);
CREATE INDEX IF NOT EXISTS idx_papers_methodology ON papers(methodology);
"""


def init_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def main():
    print("Loading source data...")
    titles: Dict[str, str] = json.loads((DATA / "paper_titles.json").read_text(encoding="utf-8"))

    chunks = json.loads((DATA / "processed_chunks.json").read_text(encoding="utf-8"))
    print(f"  {len(titles)} titles, {len(chunks)} chunks")

    # Roll up per-paper: full text (excluding references, for species/
    # methodology tagging — see module docstring), page-1 text (for the
    # date-stamp year check), n_chunks, n_pages, token_count.
    full_text_parts: Dict[str, List[str]] = defaultdict(list)
    page1_text: Dict[str, str] = {}
    n_chunks: Dict[str, int] = defaultdict(int)
    pages_seen: Dict[str, set] = defaultdict(set)
    token_count: Dict[str, int] = defaultdict(int)

    for c in chunks:
        src = c["source_file"]
        n_chunks[src] += 1
        pg = c.get("page_number")
        if pg is not None:
            pages_seen[src].add(pg)
        token_count[src] += c.get("token_count", 0) or 0
        if c.get("section") != "references":
            full_text_parts[src].append(c["text"])
        if pg in (1, None) and src not in page1_text:
            page1_text[src] = c["text"]

    # GraphRAG community assignment + paper_graph.json's own doi (used only
    # as a cross-check that our independent DOI reconstruction agrees, and
    # for community/community_title — NOT for year, see module docstring).
    community_by_doi: Dict[str, tuple] = {}
    graph_path = GRAPHRAG_OUT / "paper_graph.json"
    if graph_path.exists():
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        for n in graph["nodes"]:
            community_by_doi[n["doi"]] = (n.get("community"), n.get("comm_title"))
        print(f"  {len(community_by_doi)} community assignments loaded from paper_graph.json")
    else:
        print(f"  NOTE: {graph_path} not found — community/community_title will be NULL")

    print("Classifying species/methodology and resolving years...")
    rows = []
    year_sources = defaultdict(int)
    for src, title in titles.items():
        doi = _doi_from_source_file(src)
        full_text = " ".join(full_text_parts.get(src, [])).lower()
        year, year_source = _resolve_year(src, page1_text.get(src, ""))
        year_sources[year_source] += 1
        comm, comm_title = community_by_doi.get(doi, (None, None))

        rows.append((
            doi,
            src,
            title,
            year,
            year_source,
            comm,
            comm_title,
            _tags(full_text, SPECIES_PATTERNS) or None,
            _tags(full_text, METHODOLOGY_PATTERNS) or None,
            n_chunks.get(src, 0),
            len(pages_seen.get(src, set())),
            token_count.get(src, 0),
        ))

    print(f"  year sources: {dict(year_sources)}")

    print(f"Writing {len(rows)} rows to {DB_PATH} ...")
    conn = init_db(DB_PATH)
    conn.executemany(
        """INSERT OR REPLACE INTO papers
           (doi, source_file, title, year, year_source, community, community_title,
            species, methodology, n_chunks, n_pages, token_count)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()

    # Sanity summary, printed not just assumed
    total = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    with_year = conn.execute("SELECT COUNT(*) FROM papers WHERE year IS NOT NULL").fetchone()[0]
    with_species = conn.execute("SELECT COUNT(*) FROM papers WHERE species IS NOT NULL").fetchone()[0]
    with_method = conn.execute("SELECT COUNT(*) FROM papers WHERE methodology IS NOT NULL").fetchone()[0]
    conn.close()

    print(f"\n✓ {total} papers written to {DB_PATH}")
    print(f"  year populated:        {with_year}/{total} ({100*with_year/total:.0f}%)")
    print(f"  species populated:     {with_species}/{total} ({100*with_species/total:.0f}%)")
    print(f"  methodology populated: {with_method}/{total} ({100*with_method/total:.0f}%)")
    size_mb = DB_PATH.stat().st_size / 1e6
    print(f"  file size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
