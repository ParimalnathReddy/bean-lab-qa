#!/usr/bin/env python3
"""
Runtime query interface for the structured trial-data layer (Change 6).

Queries the SQLite database built by extract_structured_data.py — see that
script's module docstring for the important honesty caveat about how this
data was produced (LLM-extracted from paper text, not a real lab spreadsheet)
before treating results as ground truth.

Intended as the data-access layer a future query router (Change 7) calls
when a question mentions a specific cultivar, location, year, or trait —
querying real (if imperfectly extracted) numbers instead of hoping a
retrieved paper chunk happens to mention them. This module only provides the
query functions; routing/decision logic is out of scope here.

Lazy-loads the database connection the same way BeanRetriever._load_bm25 and
gene_validator._load_index do: tries an explicit path, then the
BEAN_TRIAL_DB env var, then the conventional vector_db/ locations, so this
travels for free alongside bm25_index.pkl and gene_index.pkl once uploaded
to the same HF Dataset repo — no changes needed to app.py's download logic.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

_DB_CANDIDATE_PATHS = ["vector_db/trial_data.db", "/tmp/vector_db/trial_data.db"]

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
    env_path = os.environ.get("BEAN_TRIAL_DB")
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
            count = conn.execute("SELECT COUNT(*) FROM trial_records").fetchone()[0]
            _conn = conn
            print(f"✓ Structured trial data loaded ({path}, {count} records)")
            return
        except Exception as e:
            print(f"⚠ Failed to load trial database from {path}: {e}")

    print("Structured trial database not found; query_trial_data() will return "
          "an empty list until data/trial_data.db is built and deployed")


def is_available(db_path: Optional[str] = None) -> bool:
    """Whether the structured data layer has a database loaded."""
    _load_db(db_path)
    return _conn is not None


def query_trial_data(
    cultivar: Optional[str] = None,
    location: Optional[str] = None,
    year: Optional[int] = None,
    year_range: Optional[str] = None,
    trait: Optional[str] = None,
    db_path: Optional[str] = None,
    limit: int = 200,
) -> List[Dict]:
    """
    Query extracted trial records with any combination of filters. All
    string filters do a case-insensitive partial match (SQL LIKE) since
    LLM-extracted cultivar/location spelling won't always exactly match a
    user's query. All filter values are bound as parameters — never
    string-formatted into the SQL — regardless of what the caller passes in.

    Args:
        cultivar:   partial match, e.g. "Rex" matches "Rex", "cv. Rex"
        location:   partial match, e.g. "Elora"
        year:       exact match
        year_range: "1961-2006" style — matches any year within [start, end]
        trait:      partial match, e.g. "yield"
        limit:      max rows returned (default 200, prevents an overly broad
                    filter combination from returning the entire table)

    Returns:
        List of dicts, one per matching record, each including source_doi/
        source_page/raw_context so results can be cited and audited.
    """
    _load_db(db_path)
    if _conn is None:
        return []

    where = []
    params: List = []

    if cultivar:
        where.append("cultivar LIKE ? COLLATE NOCASE")
        params.append(f"%{cultivar}%")
    if location:
        where.append("location LIKE ? COLLATE NOCASE")
        params.append(f"%{location}%")
    if trait:
        where.append("trait LIKE ? COLLATE NOCASE")
        params.append(f"%{trait}%")
    if year is not None:
        where.append("year = ?")
        params.append(int(year))
    elif year_range and "-" in year_range:
        try:
            y_start, y_end = (int(x) for x in year_range.split("-", 1))
            where.append("year BETWEEN ? AND ?")
            params.extend([y_start, y_end])
        except ValueError:
            pass

    sql = "SELECT * FROM trial_records"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY year DESC, cultivar LIMIT ?"
    params.append(limit)

    cursor = _conn.execute(sql, params)
    return [dict(row) for row in cursor.fetchall()]


def get_available_cultivars(db_path: Optional[str] = None) -> List[str]:
    """Distinct cultivar names in the database, for validating/suggesting a query."""
    _load_db(db_path)
    if _conn is None:
        return []
    rows = _conn.execute(
        "SELECT DISTINCT cultivar FROM trial_records WHERE cultivar IS NOT NULL ORDER BY cultivar"
    ).fetchall()
    return [r["cultivar"] for r in rows]


def get_available_locations(db_path: Optional[str] = None) -> List[str]:
    _load_db(db_path)
    if _conn is None:
        return []
    rows = _conn.execute(
        "SELECT DISTINCT location FROM trial_records WHERE location IS NOT NULL ORDER BY location"
    ).fetchall()
    return [r["location"] for r in rows]


def get_available_traits(db_path: Optional[str] = None) -> List[str]:
    _load_db(db_path)
    if _conn is None:
        return []
    rows = _conn.execute(
        "SELECT DISTINCT trait FROM trial_records WHERE trait IS NOT NULL ORDER BY trait"
    ).fetchall()
    return [r["trait"] for r in rows]


def format_trial_results(rows: List[Dict]) -> str:
    """
    Format query results into a citation-style markdown snippet, mirroring
    prompts.py's format_references() so a future query router can plug this
    straight into an answer without reinventing formatting.
    """
    if not rows:
        return ""
    lines = ["**Trial data:**"]
    for r in rows:
        parts = [r["cultivar"]]
        if r.get("trait"):
            value = f"{r['value']}"
            if r.get("unit"):
                value += f" {r['unit']}"
            parts.append(f"{r['trait']}: {value}")
        if r.get("location"):
            parts.append(r["location"])
        if r.get("year"):
            parts.append(str(r["year"]))
        citation = f"(doi:{r['source_doi']}, p.{r['source_page']})" if r.get("source_doi") else ""
        lines.append(f"- {' | '.join(parts)} {citation}".rstrip())
    return "\n".join(lines)
