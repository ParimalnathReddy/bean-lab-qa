#!/usr/bin/env python3
"""
Gene/locus mention extraction and validation for the Bean Lab RAG QA system
(Change 5).

Extracts gene-like mentions from an LLM-generated answer and checks them
against two things:
  1. An index built from NCBI Gene + UniProt for Phaseolus vulgaris (taxid
     3885), built by build_gene_index.py and pickled to
     <vector_db>/gene_index.pkl (see _load_index() for search locations —
     same lazy-load pattern as BeanRetriever._load_bm25 in retriever.py).
  2. CLASSICAL_GENE_SYMBOLS below — see the important caveat in that
     constant's docstring before trusting it as authoritative.

IMPORTANT FINDING (verified directly against the live NCBI and UniProt APIs
before writing this file, not assumed): classical bean genetics symbols —
Co-1..Co-14 (anthracnose resistance), Phg-1..Phg-4 (angular leaf spot),
bc-1/bc-2/bc-3 (BCMV/BCMNV resistance), fin/Ppd (growth habit/photoperiod) —
are essentially ABSENT from both NCBI Gene and UniProt for P. vulgaris.
Concretely: `Co-1[Gene Name] AND txid3885[Organism]` on NCBI returns 0 of
62,308 P. vulgaris gene records; UniProt `gene:bc-3`, `gene:Fin` for
organism_id:3885 return 0 results; even a full-text search for "anthracnose"
across all P. vulgaris NCBI gene records returns 0 hits. NCBI's P. vulgaris
gene set is almost entirely systematic genome-annotation identifiers
(LOC137819210, PHAVU_010G111100) — these databases don't catalog classical
linkage-mapped genetics nomenclature from breeding literature. Flagging every
mention of a real, decades-documented gene like Co-1 as "not found — treat
with caution" would be actively backwards: it implies the SYSTEM's answer is
untrustworthy when actually the DATABASES just don't cover this vocabulary.
Hence the CLASSICAL_GENE_SYMBOLS allowlist below, and three-way (not
two-way) verified/classical/unverified labeling everywhere in this module.

Also verified directly: NCBI does not store the Phytozome "Phvul.010G111100"
dotted format at all — it cross-references the same locus as
"PHAVU_010G111100" (dot removed, underscore inserted, no case change). See
_normalize_phytozome_id().
"""

from __future__ import annotations

import os
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional

# ── Classical bean genetics symbols ────────────────────────────────────────────
# Manually curated from general knowledge of common bean (Phaseolus vulgaris)
# disease-resistance and phenology genetics literature (the kind of gene
# symbols that show up in reviews like Kelly & Vallejo on bean disease
# resistance breeding). This is NOT sourced from a queryable database — it is
# NOT guaranteed complete or free of errors. Treat "classical_literature" as
# "well-established enough that a domain expert would recognize it," not as
# independently verified the way an NCBI/UniProt hit is. Add/correct entries
# as you find gaps; that's expected and fine.
CLASSICAL_GENE_SYMBOLS: Dict[str, str] = {
    # Anthracnose resistance (Colletotrichum lindemuthianum)
    **{f"Co-{i}": "Anthracnose (Colletotrichum lindemuthianum) resistance locus"
       for i in range(1, 15)},
    # Angular leaf spot resistance (Pseudocercospora / Phaeoisariopsis griseola)
    **{f"Phg-{i}": "Angular leaf spot (Pseudocercospora griseola) resistance locus"
       for i in range(1, 5)},
    # Bean common mosaic virus / necrosis virus resistance
    "bc-1": "Recessive BCMV resistance locus",
    "bc-2": "Recessive BCMV resistance locus",
    "bc-3": "Recessive BCMV/BCMNV resistance locus",
    "bc-u": "BCMV resistance modifier locus",
    "I": "Dominant BCMV resistance / hypersensitive-reaction locus",
    # Rust resistance (Uromyces appendiculatus)
    **{f"Ur-{i}": "Bean rust (Uromyces appendiculatus) resistance locus"
       for i in [3, 4, 5, 6, 7, 9, 11, 12, 13]},
    # Growth habit / phenology
    "fin": "Recessive indeterminate growth habit locus",
    "Fin": "Dominant determinate growth habit allele",
    "Ppd": "Photoperiod sensitivity locus",
}
_CLASSICAL_LOWER: Dict[str, str] = {k.lower(): k for k in CLASSICAL_GENE_SYMBOLS}


# ── Extraction patterns ─────────────────────────────────────────────────────────

_PHVUL_ID_RE = re.compile(r"\bPhvul\.\d{3}G\d{5,6}\b", re.IGNORECASE)  # Phytozome gene models
_PHAVU_ID_RE = re.compile(r"\bPHAVU_\d{3}G\d{5,6}g?\b", re.IGNORECASE)  # NCBI's native format
_CO_RE = re.compile(r"\bCo-(?:1[0-4]|[1-9])\b", re.IGNORECASE)  # anthracnose resistance
_PHG_RE = re.compile(r"\bPhg-[1-4]\b", re.IGNORECASE)  # angular leaf spot resistance
_BC_RE = re.compile(r"\bbc-[123]\b", re.IGNORECASE)  # BCMV/BCMNV resistance
_UR_RE = re.compile(r"\bUr-(?:1[0-9]|[1-9])\b", re.IGNORECASE)  # rust resistance

_GENE_REGEXES = [_PHVUL_ID_RE, _PHAVU_ID_RE, _CO_RE, _PHG_RE, _BC_RE, _UR_RE]

# Short, high-collision-risk symbols: only counted as a mention if a biology
# keyword appears nearby. Bare "P" and bare "I" (the BCMV "I" gene) are
# deliberately NOT matched at all — even context-anchoring isn't reliable
# enough for single-letter symbols in free-flowing prose (every "I" as a
# pronoun, every stray "P" would need filtering).
_SHORT_SYMBOLS = ["Fin", "fin", "Ppd"]
_CONTEXT_WINDOW = 60  # characters each side of the match
_CONTEXT_KEYWORDS_RE = re.compile(
    r"\b(gene|allele|locus|loci|qtl|resistance|resistant|mutation|mutant|"
    r"photoperiod|determinacy|determinate|indeterminate|genotype|phenotype|"
    r"growth habit)\b",
    re.IGNORECASE,
)

# Generic markdown-emphasized token (e.g. "*PvTFL1y*", "_bc-3_") as a fallback
# for gene names not covered by the specific patterns above.
_EMPHASIS_RE = re.compile(r"(?:\*\*|\*|_)([A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)?)(?:\*\*|\*|_)")


def _looks_gene_shaped(token: str) -> bool:
    """
    Heuristic filter for the markdown-emphasis fallback: real gene/protein
    symbols usually contain a digit, a hyphen, or more than one internal
    capital letter (PvDREB2A, bc-3, phyA, ARC5A). Plain italicized Latin
    species/genus names (*Phaseolus*, *vulgaris*) or emphasized English
    words (*however*, *important*) don't have that structure, so this
    excludes them without needing an explicit denylist of species names.
    """
    if any(ch.isdigit() for ch in token):
        return True
    if "-" in token:
        return True
    if re.fullmatch(r"[A-Z][a-z]+", token) or re.fullmatch(r"[a-z]+", token):
        return False  # plain capitalized or lowercase word
    return True


def extract_gene_mentions(text: str) -> List[str]:
    """
    Extract candidate gene/locus mentions from LLM-generated answer text.
    Favors precision over recall deliberately: this feeds a user-facing
    verification footer, so a missed gene mention is far less costly than a
    false one.
    """
    mentions: List[str] = []
    seen = set()

    def _add(raw: str) -> None:
        key = raw.lower()
        if key not in seen:
            seen.add(key)
            mentions.append(raw)

    for regex in _GENE_REGEXES:
        for m in regex.finditer(text):
            # Known ambiguity: "Co-1" through "Co-9" collide with "cobalt"
            # abbreviations in chemistry/nutrient contexts (e.g. "Cobalt-1
            # (Co-1) supplementation"). If "cobalt" appears right next to the
            # match, it's almost certainly not the anthracnose resistance
            # gene — skip it rather than report a likely-wrong verification.
            if regex is _CO_RE:
                window = text[max(0, m.start() - 20): m.end() + 5].lower()
                if "cobalt" in window:
                    continue
            _add(m.group(0))

    for symbol in _SHORT_SYMBOLS:
        for m in re.finditer(rf"\b{re.escape(symbol)}\b", text):
            start, end = max(0, m.start() - _CONTEXT_WINDOW), m.end() + _CONTEXT_WINDOW
            if _CONTEXT_KEYWORDS_RE.search(text[start:end]):
                _add(m.group(0))

    for m in _EMPHASIS_RE.finditer(text):
        token = m.group(1)
        if _looks_gene_shaped(token):
            _add(token)

    return mentions


def _normalize_phytozome_id(s: str) -> str:
    """
    'Phvul.010G111100' -> 'phavu_010g111100' (lowercased), matching how NCBI
    actually cross-references Phytozome gene models — verified directly
    against a live NCBI record (Phvul.010G111100 shows up there as
    otheraliases: "PHAVU_010G111100", not the dotted form).
    """
    if s.lower().startswith("phvul."):
        return ("phavu_" + s[len("phvul."):]).lower()
    return s.lower()


# ── Index loading (lazy, same pattern as BeanRetriever._load_bm25) ────────────

_GENE_INDEX_CANDIDATE_PATHS = ["vector_db/gene_index.pkl", "/tmp/vector_db/gene_index.pkl"]

_gene_id_map: Optional[Dict] = None
_symbol_map: Optional[Dict[str, List]] = None
_index_load_attempted = False


def _load_index(explicit_path: Optional[str] = None) -> None:
    """
    Lazily load the NCBI/UniProt gene index pickle. Tries an explicit path,
    then BEAN_GENE_INDEX env var, then the conventional vector_db/ locations
    (mirrors _BM25_CANDIDATE_PATHS in retriever.py). If nothing is found,
    validate_genes() still works using CLASSICAL_GENE_SYMBOLS alone.
    """
    global _gene_id_map, _symbol_map, _index_load_attempted
    if _index_load_attempted:
        return
    _index_load_attempted = True

    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
    env_path = os.environ.get("BEAN_GENE_INDEX")
    if env_path:
        candidates.append(env_path)
    candidates.extend(_GENE_INDEX_CANDIDATE_PATHS)

    for candidate in candidates:
        path = Path(candidate)
        if not path.exists():
            continue
        try:
            with path.open("rb") as f:
                payload = pickle.load(f)
            _gene_id_map = payload["gene_id_map"]
            _symbol_map = payload["symbol_map"]
            print(f"✓ Gene index loaded ({path}, {len(_gene_id_map)} genes, "
                  f"{len(_symbol_map)} symbols)")
            return
        except Exception as e:
            print(f"⚠ Failed to load gene index from {path}: {e}")

    print("Gene index not found; gene verification will only check the "
          "classical-symbols allowlist (see CLASSICAL_GENE_SYMBOLS)")


def validate_genes(mentions: List[str], gene_index_path: Optional[str] = None) -> List[Dict]:
    """
    Check each extracted mention against the NCBI/UniProt index, then the
    classical-symbols allowlist. Returns one dict per unique mention:
        {"mention": str, "verified": bool, "source": str|None,
         "gene_id": str|None, "description": str|None}
    source is one of "ncbi", "uniprot", "classical_literature", or None.
    """
    _load_index(gene_index_path)

    results: List[Dict] = []
    seen = set()
    for mention in mentions:
        key = mention.lower()
        if key in seen:
            continue
        seen.add(key)

        entry = {
            "mention": mention, "verified": False,
            "source": None, "gene_id": None, "description": None,
        }

        lookup_key = _normalize_phytozome_id(mention)
        gene_ids = _symbol_map.get(lookup_key) if _symbol_map else None
        if not gene_ids and _symbol_map:
            gene_ids = _symbol_map.get(key)

        if gene_ids:
            meta = _gene_id_map.get(gene_ids[0], {}) if _gene_id_map else {}
            entry.update({
                "verified": True,
                "source": meta.get("source", "ncbi"),
                "gene_id": gene_ids[0],
                "description": meta.get("description"),
            })
        elif key in _CLASSICAL_LOWER:
            canonical = _CLASSICAL_LOWER[key]
            entry.update({
                "verified": True,
                "source": "classical_literature",
                "gene_id": None,
                "description": CLASSICAL_GENE_SYMBOLS[canonical],
            })

        results.append(entry)

    return results
