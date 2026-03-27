#!/usr/bin/env python3
"""
BeanRetriever — improved retrieval for the Bean Lab RAG system.

Improvements over the original top-k retrieval:
  1. Scientific synonym / alias expansion before embedding
  2. Over-retrieval then cross-encoder reranking (retrieve 20, return top-k)
  3. Confidence assessment: SUPPORTED / PARTIALLY_SUPPORTED / INFERRED / UNSUPPORTED
  4. Soft failure: always return best-effort results; never hard-refuse on distance alone
  5. Duplicate deduplication by (source_file, page_number)
"""

from __future__ import annotations

import re
from typing import List, Dict, Optional, Tuple

# ── Scientific synonym dictionary ─────────────────────────────────────────────
# Keys are terms likely to appear in user queries.
# Values are aliases that appear in the scientific literature.
# These are used to augment the query string before embedding.

SCIENTIFIC_SYNONYMS: Dict[str, List[str]] = {
    # Disease names and pathogens
    "rust": ["bean rust", "Uromyces appendiculatus", "foliar rust", "leaf rust"],
    "white mold": ["Sclerotinia sclerotiorum", "sclerotinia", "stem rot"],
    "bcmv": ["bean common mosaic virus", "bean common mosaic necrosis virus", "BCMNV", "mosaic virus"],
    "bacterial blight": ["Xanthomonas axonopodis", "common bacterial blight", "CBB", "angular leaf spot"],
    "anthracnose": ["Colletotrichum lindemuthianum", "Colletotrichum"],
    "root rot": ["Fusarium", "Rhizoctonia", "Pythium", "fusarium root rot"],
    "halo blight": ["Pseudomonas syringae", "halo blight bacteria"],
    "bean golden mosaic": ["BGMV", "begomovirus", "whitefly transmitted"],

    # Abiotic stress
    "drought": ["water deficit", "water stress", "drought stress", "water limitation", "soil water"],
    "heat stress": ["high temperature stress", "heat tolerance", "thermal stress", "canopy temperature"],
    "low phosphorus": ["phosphorus deficiency", "P deficiency", "low P availability", "P stress"],
    "aluminum toxicity": ["Al toxicity", "acid soil", "aluminium stress"],
    "low nitrogen": ["nitrogen deficiency", "N stress", "nitrogen limitation"],

    # Biological processes
    "nitrogen fixation": ["N2 fixation", "biological nitrogen fixation", "BNF", "symbiotic fixation",
                          "Rhizobium", "nodulation", "nodule"],
    "photosynthesis": ["carbon assimilation", "net photosynthesis", "Pn", "CO2 assimilation"],
    "transpiration": ["water use efficiency", "WUE", "stomatal conductance"],

    # Crop species / common names
    "bean": ["common bean", "dry bean", "Phaseolus vulgaris", "navy bean", "pinto bean",
             "black bean", "kidney bean", "snap bean", "field bean", "garden bean"],
    "tepary bean": ["Phaseolus acutifolius", "tepary"],
    "lima bean": ["Phaseolus lunatus"],
    "soybean": ["Glycine max", "soya"],

    # Agronomic traits
    "yield": ["seed yield", "grain yield", "crop yield", "productivity", "pods per plant",
              "seeds per pod", "harvest index"],
    "quality": ["seed quality", "nutritional quality", "protein content", "cooking quality"],
    "maturity": ["days to maturity", "flowering date", "days to flowering", "phenology"],
    "plant architecture": ["growth habit", "stem strength", "lodging", "branching"],

    # Breeding / genetics
    "variety": ["cultivar", "genotype", "accession", "line", "germplasm", "landrace"],
    "breeding": ["genetic improvement", "selection", "backcrossing", "recurrent selection"],
    "resistance": ["tolerance", "immunity", "protection", "defense mechanism"],
    "pyramiding": ["gene pyramiding", "stacking genes", "combining genes"],
    "marker-assisted": ["MAS", "molecular markers", "SSR markers", "SNP markers",
                        "marker-assisted selection", "marker-assisted breeding"],
    "qtl": ["quantitative trait loci", "quantitative trait locus", "QTL mapping", "loci"],
    "gwas": ["genome-wide association", "association mapping", "association study"],

    # Growth types
    "indeterminate": ["climbing bean", "vining", "Type III", "Type IV growth habit"],
    "determinate": ["bush bean", "upright", "Type I", "Type II growth habit"],

    # Management
    "intercropping": ["mixed cropping", "polyculture", "maize-bean system", "companion planting"],
    "herbicide": ["weed control", "herbicide application", "S-metolachlor", "2,4-D"],
    "fungicide": ["fungicide application", "disease control spray"],
    "fertilizer": ["nitrogen fertilizer", "phosphorus fertilizer", "fertilization"],
    "irrigation": ["supplemental irrigation", "drip irrigation", "furrow irrigation"],
}

# Distance thresholds interpreted as approximate cosine similarity
# With L2-normalized embeddings: cos_sim ≈ 1 - (dist^2 / 2)
THRESHOLD_SUPPORTED     = 0.45   # cos_sim ≈ 0.90  — very strong match
THRESHOLD_PARTIAL       = 0.65   # cos_sim ≈ 0.79  — good match
THRESHOLD_INFERRED      = 0.85   # cos_sim ≈ 0.64  — moderate match
THRESHOLD_UNSUPPORTED   = 1.10   # cos_sim ≈ 0.40  — weak match (still attempt)


def expand_query(query: str) -> str:
    """
    Augment the query string with scientific synonyms.

    Strategy: for each synonym key found in the query (case-insensitive),
    append up to 2 alias terms to the query string. This single augmented
    string is then embedded, pulling the embedding vector toward the
    scientific vocabulary used in the papers.

    Returns the augmented query string.
    """
    query_lower = query.lower()
    additions: List[str] = []

    for term, aliases in SCIENTIFIC_SYNONYMS.items():
        if term in query_lower:
            # Add first 2 aliases only to avoid token-length explosion
            additions.extend(aliases[:2])

    if additions:
        # Append unique additions
        seen = set(query_lower.split())
        new_terms = [a for a in additions if a.lower() not in seen]
        if new_terms:
            return query + " " + " ".join(new_terms[:6])  # cap at 6 extra terms

    return query


def _deduplicate(chunks: List[Dict]) -> List[Dict]:
    """Remove duplicate chunks by (source_file, page_number) keeping lowest distance."""
    seen: Dict[Tuple, float] = {}
    result: List[Dict] = []
    for chunk in chunks:
        key = (chunk.get("source", ""), chunk.get("page", ""))
        dist = chunk.get("distance", 1.0)
        if key not in seen or dist < seen[key]:
            seen[key] = dist
            result.append(chunk)
    # Re-sort by distance after dedup
    return sorted(result, key=lambda c: c.get("distance", 1.0))


def assess_confidence(chunks: List[Dict]) -> str:
    """
    Assess overall retrieval confidence from the top chunks.

    Returns one of:
      SUPPORTED          — strong evidence found
      PARTIALLY_SUPPORTED — some evidence but incomplete
      INFERRED           — weak evidence; answer requires reasoning
      UNSUPPORTED        — no meaningful evidence found (still attempt)
    """
    if not chunks:
        return "UNSUPPORTED"

    best_dist = chunks[0].get("distance", 1.0)
    relevant_count = sum(1 for c in chunks if c.get("distance", 1.0) <= THRESHOLD_INFERRED)

    if best_dist <= THRESHOLD_SUPPORTED and relevant_count >= 3:
        return "SUPPORTED"
    elif best_dist <= THRESHOLD_PARTIAL and relevant_count >= 2:
        return "PARTIALLY_SUPPORTED"
    elif best_dist <= THRESHOLD_INFERRED:
        return "INFERRED"
    else:
        return "UNSUPPORTED"


class BeanRetriever:
    """
    Retriever for the Bean Lab ChromaDB collection.

    Pipeline:
      1. expand_query()         — add scientific synonyms to query
      2. retrieve_candidates()  — ChromaDB top-20 similarity search
      3. rerank()               — cross-encoder reranking (or distance fallback)
      4. deduplicate()          — remove duplicate page chunks
      5. assess_confidence()    — label evidence strength
    """

    def __init__(self, collection, embedder=None):
        """
        Args:
            collection: ChromaDB collection object
            embedder:   SentenceTransformer instance (for generating query embeddings).
                        If None, uses ChromaDB's query_texts interface.
        """
        self.collection = collection
        self.embedder = embedder
        self._cross_encoder = None
        self._cross_encoder_loaded = False

    def _load_cross_encoder(self):
        """Load cross-encoder model lazily. Falls back gracefully if unavailable."""
        if self._cross_encoder_loaded:
            return
        self._cross_encoder_loaded = True
        try:
            from sentence_transformers import CrossEncoder
            self._cross_encoder = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                max_length=512,
            )
            print("✓ Cross-encoder reranker loaded (ms-marco-MiniLM-L-6-v2)")
        except Exception as e:
            self._cross_encoder = None
            print(f"⚠ Cross-encoder not available ({e}); using distance-based ranking")

    def _chromadb_query(self, query_text: str, n_results: int, year_range: Optional[str]) -> List[Dict]:
        """Run ChromaDB query and return normalized chunk dicts."""
        where = {"year_range": year_range} if year_range else None

        if self.embedder is not None:
            # Generate embedding locally (used in HF Spaces to avoid double-encoding)
            emb = self.embedder.encode([query_text], normalize_embeddings=True).tolist()
            results = self.collection.query(
                query_embeddings=emb,
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        else:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            source = meta.get("source_file", "unknown")
            chunks.append({
                "text": doc,
                "source": source,
                "doi": _filename_to_doi(source),
                "year_range": meta.get("year_range", "unknown"),
                "page": meta.get("page_number", "?"),
                "section": meta.get("section", ""),
                "distance": round(float(dist), 4),
                "rerank_score": None,
            })
        return chunks

    def retrieve_candidates(
        self,
        query: str,
        n_candidates: int = 20,
        year_range: Optional[str] = None,
    ) -> List[Dict]:
        """Retrieve over-sampled candidates with query expansion."""
        expanded = expand_query(query)
        candidates = self._chromadb_query(expanded, n_candidates, year_range)
        return _deduplicate(candidates)

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 10) -> List[Dict]:
        """
        Rerank candidates using cross-encoder.
        Falls back to distance ordering if cross-encoder unavailable.
        """
        self._load_cross_encoder()

        if self._cross_encoder is None or len(candidates) == 0:
            return candidates[:top_k]

        try:
            pairs = [(query, c["text"]) for c in candidates]
            scores = self._cross_encoder.predict(pairs)
            for chunk, score in zip(candidates, scores):
                chunk["rerank_score"] = float(score)
            reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
            return reranked[:top_k]
        except Exception as e:
            print(f"⚠ Reranking failed ({e}); using distance ordering")
            return candidates[:top_k]

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        year_range: Optional[str] = None,
        n_candidates: int = 20,
    ) -> Tuple[List[Dict], str]:
        """
        Full pipeline: expand → retrieve candidates → rerank → assess confidence.

        Returns:
            (chunks, confidence_label)
            confidence_label: SUPPORTED | PARTIALLY_SUPPORTED | INFERRED | UNSUPPORTED
        """
        candidates = self.retrieve_candidates(query, n_candidates=n_candidates, year_range=year_range)
        ranked = self.rerank(query, candidates, top_k=top_k)
        confidence = assess_confidence(ranked)
        return ranked, confidence


def _filename_to_doi(filename: str) -> str:
    """Convert stored filename back to DOI.
    e.g. '10.2135_cropsci2004.1799.pdf' → '10.2135/cropsci2004.1799'
    """
    name = filename.replace(".pdf", "")
    parts = name.split("_", 1)
    if len(parts) == 2 and parts[0].startswith("10."):
        return f"{parts[0]}/{parts[1]}"
    return name
