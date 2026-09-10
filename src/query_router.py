#!/usr/bin/env python3
"""
Query router for the Bean Lab RAG QA system (Change 7; unified with
GraphRAG mode detection in Change 12).

Decides which subsystems a question should activate, as flags on a single
RoutingDecision:
  mode                 — "local" (per-question retrieval, the default) or
                        "global" (GraphRAG community-summary synthesis,
                        Change 9). Moved here from hf_space/app.py's
                        _is_global() in Change 12 — see that section below
                        for why, and for what's deliberately NOT done yet.
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
all three active at once. mode is the one exception that still behaves like
a single category for now — see the Change 12 section below for why.

── CHANGE 12: UNIFYING THE TWO ROUTERS ──────────────────────────────────────
Before this change, two independent systems decided things about the same
incoming query with no coordination: this module (rule-based, entity-aware,
flag-based) and hf_space/app.py's _is_global() (a standalone keyword-list
check that ran BEFORE route() and gated the entire pipeline into "local" or
"global" before route() ever saw the question). A query like "yield trends
across all Michigan navy bean trials" is structurally both a structured-data
question (Change 7 should catch "yield," "Michigan") and a broad-synthesis
question (Change 9 should catch "trends," "across") — but the old two-router
structure made that impossible to express: _is_global() picked one path,
route() only ever ran (and only ever COULD run) inside the "local" branch it
picked.

Fix: _GLOBAL_SIGNALS and the keyword check that used to be _is_global() now
live here, and route() sets decision.mode as one more field on the SAME
RoutingDecision it already returns, computed in the SAME pass as the other
three flags. hf_space/app.py's chat() now calls route() exactly once, at the
top, and branches on decision.mode instead of calling a separate function
first and route() again afterward — one call site, one decision object,
where there used to be two.

Scope deliberately kept narrow: mode is still exactly one of "local" or
"global" — never "both" — even though the Michigan example above clearly
wants both structured-data lookup AND community synthesis at once. Actually
running both concurrently (parallel local+global execution, merging their
answers) is real, separate work — new prompt assembly, new UI handling for
two result sets, deciding how to reconcile/present a combined answer — and
is deferred to a later change (see the project's change log), not smuggled
in here. This change fixes the STRUCTURAL bug (two uncoordinated decision
points) without also trying to solve the harder concurrent-execution
question in the same commit.

use_literature/use_structured_data/run_gene_validation are computed
identically regardless of mode — route() doesn't special-case "global"
queries when detecting genes or structured-data terms, since a
"yield trends across all Michigan navy bean trials"-style question should
still get flagged for structured data even while mode="local" wins the
single-mode decision for now. hf_space/app.py's global-mode branch doesn't
currently consume use_structured_data (see that module for the exact,
still-local-only wiring) — that's the concurrent-execution gap left for
later, not something this change silently papers over.

qa_with_ollama.py and both eval scripts import route() from here and now
receive decision.mode "for free," the same way they gained
use_structured_data/run_gene_validation awareness in Change 7 — but this is
inert for them today: GraphRAG artifacts (community_reports.json,
community_embeddings.npy) are loaded ONLY in hf_space/app.py, not in the
batch/HPCC path, so nothing on the batch side currently acts on
decision.mode == "global". Confirmed by checking qa_with_ollama.py directly
before making this change, not assumed.

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

── CHANGE 17: WIRING paper_metadata.db INTO ROUTING (use_paper_metadata,
detected_species, detected_methodology, detected_paper_year_min/max) ──
Change 14 built the per-paper (DOI/year/species/methodology) SQLite index
and its query interface, but deliberately stopped there — no detection
logic existed anywhere to notice a question wanted it. This closes that
gap, reusing the SAME two mechanisms this module already uses for
structured_data, rather than inventing new ones:
  - Species/methodology: matched via the EXISTING _match_against_cache()
    helper against the REAL tag vocabulary in the live database
    (paper_metadata.get_available_species()/get_available_methodologies())
    — same "check what's actually there, not a hardcoded list" principle
    as cultivar/location/trait above, and the SAME underscore-to-space
    normalization already handles tag slugs like "phaseolus_vulgaris" and
    "qtl_mapping" matching natural phrasing ("Phaseolus vulgaris",
    "QTL mapping") for free — no new matching logic needed, just a new
    cache to match against.
  - Bounded year range: reuses _extract_year() as-is (already used for
    structured_data's year filtering) — two distinct years mentioned
    becomes a (year_min, year_max) pair.
  - Open-ended year ("papers from 2015+", "since 2015", "after 2015",
    "post-2015"): _extract_year() alone can't express this — it only
    detects a single exact year or a two-year bounded range. A NEW
    _extract_open_ended_year() regex was added specifically for this
    phrasing, since it's the exact shape Change 14's own motivating
    example ("papers from 2015+ using GWAS methodology") uses.

TRIGGER, deliberately narrower than "any year mention": a single bare year
with no other signal (e.g. "What happened in 2019?") does NOT set
use_paper_metadata — too easy to confuse with an incidental year reference
inside an ordinary literature question, the same false-trigger risk this
module's docstring already warns about for hardcoded keyword lists.
use_paper_metadata fires only on a species match, a methodology match, OR
an explicitly filter-shaped year phrase (open-ended or bounded-range) —
signals with much less incidental-mention risk than a lone year. All
detected fields (species/methodology/year bounds) are independent and
combine on the same RoutingDecision when more than one is present — e.g.
"papers from 2015+ using GWAS methodology" sets BOTH
detected_paper_year_min=2015 AND detected_methodology=["gwas"] on one call,
exactly the compound case Change 14 was built for.
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
from paper_metadata import get_available_species, get_available_methodologies


@dataclass
class RoutingDecision:
    # "local" or "global" (Change 12) — never "both" yet; see this module's
    # docstring, CHANGE 12 section, for why "both" is out of scope here.
    mode: str = "local"
    use_literature: bool = True
    use_structured_data: bool = False
    run_gene_validation: bool = False
    detected_cultivars: List[str] = field(default_factory=list)
    detected_locations: List[str] = field(default_factory=list)
    detected_traits: List[str] = field(default_factory=list)
    detected_genes: List[str] = field(default_factory=list)
    detected_year: Optional[int] = None
    detected_year_range: Optional[str] = None
    # ── Change 17: per-paper metadata filtering (Change 14's paper_metadata.db,
    # wired into routing here — see this module's CHANGE 17 section below) ──
    use_paper_metadata: bool = False
    detected_species: List[str] = field(default_factory=list)
    detected_methodology: List[str] = field(default_factory=list)
    detected_paper_year_min: Optional[int] = None
    detected_paper_year_max: Optional[int] = None


# ── Genetics vocabulary (distinct from gene_validator's gene-NAME patterns —
# this flags the query as being ABOUT genetics generally, not extracting a
# specific gene symbol) ─────────────────────────────────────────────────────
_GENETICS_KEYWORDS_RE = re.compile(
    r"\b(qtl|allele|alleles|marker|markers|chromosome|locus|loci|genotype|genotypes|"
    r"genomic|gene expression|linkage map|genetic map)\b",
    re.IGNORECASE,
)

# ── GraphRAG global-mode signal (Change 12 — moved verbatim from
# hf_space/app.py's _is_global()/_GLOBAL_SIGNALS; unchanged keyword list,
# only the location moved, per this module's Change 12 docstring section) ──
_GLOBAL_SIGNALS = [
    "overall", "trend", "trends", "across all", "literature",
    "compare", "comparison", "how has", "how have", "what are the main",
    "what are the major", "overview", "summarize", "summarise", "synthesis",
    "broadly", "generally", "in general", "historically", "over time", "evolv",
    "review", "landscape", "state of", "consensus",
]


def _is_global(query: str) -> bool:
    ql = query.lower()
    return any(s in ql for s in _GLOBAL_SIGNALS)


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


# Change 17: "papers from 2015+" style open-ended year phrasing —
# _extract_year() above only expresses a single exact year or a two-year
# bounded range, neither of which fits "2015 or later, no upper bound."
_OPEN_ENDED_YEAR_RE = re.compile(
    r"\b(?:since|after|post[\s-]?)\s*((?:19|20)\d{2})\b"
    r"|\b((?:19|20)\d{2})\s*\+"
    r"|\bfrom\s+((?:19|20)\d{2})\s+(?:onwards?|on)\b",
    re.IGNORECASE,
)


def _extract_open_ended_year(query: str) -> Optional[int]:
    m = _OPEN_ENDED_YEAR_RE.search(query)
    if not m:
        return None
    return int(next(g for g in m.groups() if g))


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


# ── Cached paper-metadata vocabulary (Change 17), same lazy pattern as above ──
_species_cache: Optional[FrozenSet[str]] = None
_methodology_cache: Optional[FrozenSet[str]] = None
_paper_metadata_cache_load_attempted = False


def _load_paper_metadata_cache() -> Tuple[FrozenSet[str], FrozenSet[str]]:
    global _species_cache, _methodology_cache, _paper_metadata_cache_load_attempted
    if not _paper_metadata_cache_load_attempted:
        _paper_metadata_cache_load_attempted = True
        try:
            _species_cache = frozenset(get_available_species())
            _methodology_cache = frozenset(get_available_methodologies())
        except Exception as e:
            print(f"⚠ Could not load paper-metadata vocabulary for routing: {e}")
            _species_cache = frozenset()
            _methodology_cache = frozenset()
    return _species_cache, _methodology_cache


def route(query: str) -> RoutingDecision:
    """
    Rule-based-only routing decision for a single user question — the single
    call site for mode + all three flags as of Change 12 (see this module's
    docstring). Cheap: one extra keyword-list scan for mode, one regex pass,
    one call into gene_validator's extractor, and a handful of word-boundary
    searches against a cached, process-lifetime vocabulary set — no network
    or LLM call, safe to run on every question.
    """
    decision = RoutingDecision()
    query_lower = query.lower()

    # ── Mode (Change 12) ────────────────────────────────────────────────────
    decision.mode = "global" if _is_global(query) else "local"

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

    # ── Paper-metadata flag (Change 17) — see this module's CHANGE 17
    # docstring section for the full trigger rationale.
    species, methodology = _load_paper_metadata_cache()
    matched_species = _match_against_cache(query_lower, species)
    matched_methodology = _match_against_cache(query_lower, methodology)
    open_ended_year = _extract_open_ended_year(query)

    if matched_species or matched_methodology or open_ended_year or year_range:
        decision.use_paper_metadata = True
        decision.detected_species = matched_species
        decision.detected_methodology = matched_methodology
        if open_ended_year is not None:
            decision.detected_paper_year_min = open_ended_year
        elif year_range:
            y_start, y_end = year_range.split("-", 1)
            decision.detected_paper_year_min = int(y_start)
            decision.detected_paper_year_max = int(y_end)

    return decision
