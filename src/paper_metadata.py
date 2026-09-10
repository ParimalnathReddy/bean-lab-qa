#!/usr/bin/env python3
"""
Runtime query interface for the per-paper structured metadata index (Change 14).

Queries the SQLite database built by scripts/build_paper_metadata.py — see
that script's module docstring for the honesty boundary on each column
(which are exact/reused vs. best-effort rule-based tags, and why "trait"
isn't a column at all) before treating a result as authoritative.

One row per PAPER (962), not per chunk (21,876) — this is what makes
filtered queries like "papers from 2015+ using GWAS methodology" a real SQL
WHERE clause instead of hoping retrieval happens to surface the right
chunks. This module only provides query functions; deciding WHEN to use
them (a future query-router integration, analogous to how Change 7 wired
Change 6's trial-data layer into routing) is out of scope here — same
scope boundary Change 6's own structured_data.py drew for itself.

Lazy-loads the database connection the same way structured_data.py,
BeanRetriever._load_bm25, and gene_validator._load_index all do: tries an
explicit path, then a BEAN_PAPER_METADATA_DB env var, then the conventional
vector_db/ locations, so this travels for free alongside trial_data.db/
bm25_index.pkl/gene_index.pkl once uploaded to the same HF Dataset repo —
no changes needed to app.py's download logic.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

_DB_CANDIDATE_PATHS = ["vector_db/paper_metadata.db", "/tmp/vector_db/paper_metadata.db"]

_conn: Optional[sqlite3.Connection] = None
_load_attempted = False


def _load_db(explicit_path: Optional[str] = None) -> None:
    global _conn, _load_attempted
    if _load_attempted:
        return
    _load_attempted = True

    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
    env_path = os.environ.get("BEAN_PAPER_METADATA_DB")
    if env_path:
        candidates.append(env_path)
    candidates.extend(_DB_CANDIDATE_PATHS)

    for candidate in candidates:
        path = Path(candidate)
        if not path.exists():
            continue
        try:
            # Read-only connection: this module only ever queries, never writes.
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            count = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
            _conn = conn
            print(f"✓ Paper metadata index loaded ({path}, {count} papers)")
            return
        except Exception as e:
            print(f"⚠ Failed to load paper metadata database from {path}: {e}")

    print("Paper metadata database not found; query_papers() will return "
          "an empty list until vector_db/paper_metadata.db is built and deployed")


def is_available(db_path: Optional[str] = None) -> bool:
    """Whether the paper metadata index has a database loaded."""
    _load_db(db_path)
    return _conn is not None


def query_papers(
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    species: Optional[str] = None,
    methodology: Optional[str] = None,
    doi: Optional[str] = None,
    community: Optional[int] = None,
    db_path: Optional[str] = None,
    limit: int = 200,
) -> List[Dict]:
    """
    Query per-paper metadata with any combination of filters, ANDed together.

    species/methodology do a case-insensitive partial match (SQL LIKE)
    against the comma-joined tag string (e.g. species="vulgaris" matches a
    row tagged "phaseolus_vulgaris,glycine_max"), since callers shouldn't
    need to know the exact tag vocabulary in scripts/build_paper_metadata.py
    to filter on it. All filter values are bound as parameters — never
    string-formatted into the SQL — regardless of what the caller passes in.

    Args:
        year_min/year_max: inclusive bounds; either or both may be given
        species:      partial match against the species tag string
        methodology:  partial match against the methodology tag string
        doi:          exact match
        community:    exact match against the GraphRAG community id (Change 9)
        limit:        max rows returned (default 200)

    Returns:
        List of dicts, one per matching paper.
    """
    _load_db(db_path)
    if _conn is None:
        return []

    where = []
    params: List = []

    if year_min is not None:
        where.append("year >= ?")
        params.append(int(year_min))
    if year_max is not None:
        where.append("year <= ?")
        params.append(int(year_max))
    if species:
        where.append("species LIKE ? COLLATE NOCASE")
        params.append(f"%{species}%")
    if methodology:
        where.append("methodology LIKE ? COLLATE NOCASE")
        params.append(f"%{methodology}%")
    if doi:
        where.append("doi = ?")
        params.append(doi)
    if community is not None:
        where.append("community = ?")
        params.append(int(community))

    sql = "SELECT * FROM papers"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY year DESC, title LIMIT ?"
    params.append(limit)

    cursor = _conn.execute(sql, params)
    return [dict(row) for row in cursor.fetchall()]


def get_available_species(db_path: Optional[str] = None) -> List[str]:
    """Distinct species tags in the database (each row can have several,
    comma-joined — this returns the individual tag vocabulary, not the raw
    joined strings)."""
    _load_db(db_path)
    if _conn is None:
        return []
    rows = _conn.execute("SELECT DISTINCT species FROM papers WHERE species IS NOT NULL").fetchall()
    tags = set()
    for r in rows:
        tags.update(r["species"].split(","))
    return sorted(tags)


def get_available_methodologies(db_path: Optional[str] = None) -> List[str]:
    _load_db(db_path)
    if _conn is None:
        return []
    rows = _conn.execute("SELECT DISTINCT methodology FROM papers WHERE methodology IS NOT NULL").fetchall()
    tags = set()
    for r in rows:
        tags.update(r["methodology"].split(","))
    return sorted(tags)


def format_paper_results(rows: List[Dict]) -> str:
    """
    Format query results into a citation-style markdown snippet, mirroring
    prompts.py's format_references() and structured_data.py's
    format_trial_results() so a future caller can plug this straight into
    an answer without reinventing formatting.
    """
    if not rows:
        return ""
    lines = ["**Papers matching filter:**"]
    for r in rows:
        parts = [r["title"] or r["source_file"]]
        if r.get("year"):
            parts.append(str(r["year"]))
        if r.get("methodology"):
            parts.append(r["methodology"].replace(",", ", "))
        citation = f"(doi:{r['doi']})" if r.get("doi") else ""
        lines.append(f"- {' | '.join(parts)} {citation}".rstrip())
    return "\n".join(lines)
