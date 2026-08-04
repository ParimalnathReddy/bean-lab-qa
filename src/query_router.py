#!/usr/bin/env python3
"""
Query router for the Bean Lab RAG QA system (Change 7).

Decides which of three independent subsystems a question should activate:
  use_literature      — vector + BM25 retrieval over the paper corpus
                        (Changes 1-2). Effectively always True: even a pure
                        trial-data question needs literature to interpret the
                        numbers, compare them to broader findings, and cite
                        supporting studies.
  use_structured_data — query structured_data.py's SQLite trial database
                        (Change 6) for exact cultivar/location/year/trait
                        numbers, instead of hoping a retrieved chunk
                        mentions them.
  run_gene_validation — run gene_validator.py (Change 5) on the generated
                        answer to catch fabricated gene/locus claims.

These are flags, not a single category (unlike a GPT-4o-classifier design
that picks one bucket): a question like "What is the yield of navy beans
with Co-1 resistance in Michigan trials?" is simultaneously a literature
question, a structured-data question, and a genetics question, and needs
all three active at once.

Design: rule-based detection only, no LLM call — this runs before every
single question (including the 80%+ that are plain literature queries), so
adding a classifier round-trip here would tax the common case to handle the
rare one. See the module-level NOTE below on why an LLM disambiguation
fallback isn't implemented yet.

Detection strategy, per subsystem:
  - Gene mentions: reuse gene_validator.extract_gene_mentions() directly —
    NOT reimplemented here. Those regexes are already hardened (including
    negative-context guards like the "Cobalt-1"/"Co-1" collision fix from
    Change 5); duplicating them would just create a second copy to drift out
    of sync, the same failure mode as the retriever.py/hf_space/retriever.py
    split earlier in this project.
  - Cultivar/location/trait: checked against the REAL values currently in
    the structured-data database (structured_data.get_available_cultivars/
    locations/traits()), cached once per process — not a hardcoded keyword
    list. A hardcoded list of common cultivar/location words would false-
    trigger constantly ("Black" and "Red" are both market classes AND
    ordinary English words; "Elora" could appear in a paper's author
    affiliation rather than as a trial site). Checking against what's
    ACTUALLY in the database only fires when the term genuinely resolves to
    something, and stays correct automatically as more trial data is added
    — no keyword list to maintain in parallel.

One deliberate deviation from "cache as frozensets, check tokens for O(1)
membership": real cultivar names can be multi-word ("Black Turtle Soup") or
hyphenated, and Change 6's extracted trait labels use underscores
("days_to_maturity") that never appear in natural phrasing ("days to
maturity"). Pure per-token set membership silently misses both. Matching is
therefore done as a word-boundary regex search per cached value (still
backed by a frozenset, still loaded once, still fast at the realistic scale
of a lab's trial data — tens to low hundreds of distinct values, not
thousands) rather than a hash lookup per token. This is the one place this
module intentionally does a bit more work than specified, and it's a
one-line fix, not new machinery.

NOTE on the LLM fallback: the original design allowed an LLM disambiguation
pass "for ambiguous cases." That's not implemented here — "ambiguous" was
never given a crisp definition, and a query that matches none of the rules
above already has a well-defined, safe default: use_literature stays True,
the other two stay False. Building a fuzzy secondary classifier for an
undefined trigger condition risks either never firing (dead code) or firing
unpredictably (paying the latency cost with no clear benefit). route() is
structured so this can be added later without changing its signature.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import FrozenSet, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))
from gene_validator import extract_gene_mentions
from structured_data import get_available_cultivars, get_available_locations, get_available_traits


@dataclass
class RoutingDecision:
    use_literature: bool = True
    use_structured_data: bool = False
    run_gene_validation: bool = False
    detected_cultivars: List[str] = field(default_factory=list)
    detected_locations: List[str] = field(default_factory=list)
    detected_traits: List[str] = field(default_factory=list)
    detected_genes: List[str] = field(default_factory=list)
    detected_year: Optional[int] = None
    detected_year_range: Optional[str] = None


# ── Genetics vocabulary (distinct from gene_validator's gene-NAME patterns —
# this flags the query as being ABOUT genetics generally, not extracting a
# specific gene symbol) ─────────────────────────────────────────────────────
_GENETICS_KEYWORDS_RE = re.compile(
    r"\b(qtl|allele|alleles|marker|markers|chromosome|locus|loci|genotype|genotypes|"
    r"genomic|gene expression|linkage map|genetic map)\b",
    re.IGNORECASE,
)

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

# Values shorter than this are excluded from cultivar/location/trait matching
# — Change 6's LLM extraction can produce noisy short codes, and a 1-2
# character "cultivar" would match almost any query as a substring.
_MIN_MATCH_LEN = 3


def _extract_year(query: str) -> Tuple[Optional[int], Optional[str]]:
    years = sorted({int(m.group(0)) for m in _YEAR_RE.finditer(query)})
    if not years:
        return None, None
    if len(years) == 1:
        return years[0], None
    return None, f"{years[0]}-{years[-1]}"


def _match_against_cache(query_lower: str, candidates: FrozenSet[str]) -> List[str]:
    """
    Word-boundary match each cached value (normalizing underscores to spaces,
    so Change 6's "days_to_maturity" trait label matches natural "days to
    maturity" phrasing) against the query. Returns original (un-normalized)
    values, in the order they appear in `candidates` (frozenset order is
    arbitrary but stable within a process).
    """
    matched = []
    for candidate in candidates:
        normalized = candidate.replace("_", " ").strip().lower()
        if len(normalized) < _MIN_MATCH_LEN:
            continue
        if re.search(rf"\b{re.escape(normalized)}\b", query_lower):
            matched.append(candidate)
    return matched


# ── Cached structured-data vocabulary (lazy, loaded once per process) ────────

_cultivars_cache: Optional[FrozenSet[str]] = None
_locations_cache: Optional[FrozenSet[str]] = None
_traits_cache: Optional[FrozenSet[str]] = None
_cache_load_attempted = False


def _load_cache() -> Tuple[FrozenSet[str], FrozenSet[str], FrozenSet[str]]:
    global _cultivars_cache, _locations_cache, _traits_cache, _cache_load_attempted
    if not _cache_load_attempted:
        _cache_load_attempted = True
        try:
            _cultivars_cache = frozenset(get_available_cultivars())
            _locations_cache = frozenset(get_available_locations())
            _traits_cache = frozenset(get_available_traits())
        except Exception as e:
            print(f"⚠ Could not load structured-data vocabulary for routing: {e}")
            _cultivars_cache = frozenset()
            _locations_cache = frozenset()
            _traits_cache = frozenset()
    return _cultivars_cache, _locations_cache, _traits_cache


def route(query: str) -> RoutingDecision:
    """
    Rule-based-only routing decision for a single user question. Cheap: one
    regex pass, one call into gene_validator's extractor, and a handful of
    word-boundary searches against a cached, process-lifetime vocabulary set
    — no network or LLM call, safe to run on every question.
    """
    decision = RoutingDecision()
    query_lower = query.lower()

    # ── Gene validation flag ──────────────────────────────────────────────
    try:
        gene_mentions = extract_gene_mentions(query)
    except Exception:
        gene_mentions = []
    if gene_mentions or _GENETICS_KEYWORDS_RE.search(query):
        decision.run_gene_validation = True
        decision.detected_genes = gene_mentions

    # ── Structured data flag ──────────────────────────────────────────────
    cultivars, locations, traits = _load_cache()
    matched_cultivars = _match_against_cache(query_lower, cultivars)
    matched_locations = _match_against_cache(query_lower, locations)
    matched_traits = _match_against_cache(query_lower, traits)
    year, year_range = _extract_year(query)

    if matched_cultivars or matched_locations or matched_traits:
        decision.use_structured_data = True
        decision.detected_cultivars = matched_cultivars
        decision.detected_locations = matched_locations
        decision.detected_traits = matched_traits
        decision.detected_year = year
        decision.detected_year_range = year_range

    return decision
