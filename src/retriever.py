#!/usr/bin/env python3
"""
BeanRetriever — improved retrieval for the Bean Lab RAG system.

Improvements over the original top-k retrieval:
  1. Scientific synonym / alias expansion before embedding
  2. Hybrid retrieval: dense (ChromaDB) + BM25 keyword search, merged via
     Reciprocal Rank Fusion, automatic whenever a BM25 index is present
     (falls back to dense-only otherwise — see _load_bm25())
  3. Over-retrieval then cross-encoder reranking (retrieve ~20-35, return top-k)
  4. Confidence assessment: SUPPORTED / PARTIALLY_SUPPORTED / INFERRED / UNSUPPORTED
  5. Soft failure: always return best-effort results; never hard-refuse on distance alone
  6. Duplicate deduplication by (source_file, page_number)
"""

from __future__ import annotations

import heapq
import os
import pickle
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# ── Embedding model ────────────────────────────────────────────────────────────
# Must match the model used in generate_embeddings.py to build the vector store.
# BGE models are trained with an asymmetric query/passage convention: queries are
# prefixed with this instruction, passages/documents are not.
EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# ── BM25 hybrid retrieval (Reciprocal Rank Fusion) ─────────────────────────────
# The BM25 index is built once, alongside ChromaDB, in build_vector_store.py and
# pickled to <vector_db>/bm25_index.pkl (same directory as chroma.sqlite3, so it
# travels for free wherever vector_db/ already goes — including the HF Dataset
# repo snapshot_download pulls in hf_space/app.py). If the pickle can't be found,
# hybrid retrieval is silently skipped and BeanRetriever falls back to
# dense-only, matching this codebase's soft-failure philosophy everywhere else.

RRF_K = 60  # standard Reciprocal Rank Fusion constant

# Conventional locations every current caller already uses for vector_db/:
# a relative "vector_db/" for HPCC scripts and deploy/app.py (both run with the
# project directory as cwd), and "/tmp/vector_db/" for the HF Space (the
# snapshot_download target hardcoded in hf_space/app.py).
_BM25_CANDIDATE_PATHS = ["vector_db/bm25_index.pkl", "/tmp/vector_db/bm25_index.pkl"]

_BM25_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "in", "to", "for", "on",
    "at", "by", "is", "was", "are", "were", "be", "been", "with",
    "that", "this", "it", "as", "from", "not", "but", "have", "has",
    "had", "do", "did", "does", "will", "would", "could", "should",
    "may", "might", "also", "than", "its", "their", "they", "we",
})

# Matches alphanumeric runs with internal dots preserved (so gene IDs like
# "Phvul.003G108200" and decimals like "6.5" stay as one token) but treats
# hyphens/underscores/other punctuation as separators (so "drought-tolerant"
# splits into "drought" + "tolerant", matching how those words normally
# appear separately in the corpus).
_BM25_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)*")


def tokenize_bm25(text: str) -> List[str]:
    """
    Shared tokenizer for both sides of BM25: used to build the index in
    build_vector_store.py AND to tokenize queries in BeanRetriever. Must stay
    identical on both sides, or BM25 scores become meaningless.
    """
    return [
        t for t in (m.group(0).lower() for m in _BM25_TOKEN_RE.finditer(text or ""))
        if t not in _BM25_STOPWORDS
    ]


def _parse_chroma_id(doc_id: str) -> Optional[int]:
    """
    ChromaDB ids are 'chunk_{i}', where i is the row position in
    embeddings.npy / processed_chunks.json — the same order the BM25 index's
    corpus was built from in build_vector_store.py. This integer is the
    shared identifier used to merge dense and BM25 rankings via RRF.
    """
    try:
        return int(str(doc_id).rsplit("_", 1)[-1])
    except (ValueError, IndexError):
        return None


def _dedupe_keep_order(chunks: List[Dict]) -> List[Dict]:
    """
    Dedupe RRF-fused chunks by (source, page), keeping the first occurrence.

    Unlike _deduplicate() below, this does NOT re-sort by distance: `chunks`
    is expected to already be sorted by fused RRF score, and BM25-only
    chunks have distance=None, which can't be compared against real
    distances.
    """
    seen = set()
    result: List[Dict] = []
    for c in chunks:
        key = (c.get("source", ""), str(c.get("page", "")))
        if key in seen:
            continue
        seen.add(key)
        result.append(c)
    return result


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

    def _dist(c: Dict) -> float:
        # BM25-only chunks (no dense match) have distance=None; treat as
        # worst-case rather than crashing the comparison below.
        d = c.get("distance")
        return d if d is not None else 1.0

    best_dist = _dist(chunks[0])
    relevant_count = sum(1 for c in chunks if _dist(c) <= THRESHOLD_INFERRED)

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
      1. expand_query()         — add scientific synonyms to query (dense side only)
      2. retrieve_candidates()  — ChromaDB top-N dense search, fused via Reciprocal
                                  Rank Fusion with BM25 top-N keyword search if a
                                  BM25 index is available (see _load_bm25);
                                  falls back to dense-only otherwise
      3. rerank()               — cross-encoder reranking (or distance fallback)
      4. deduplicate()          — remove duplicate page chunks
      5. assess_confidence()    — label evidence strength
    """

    def __init__(self, collection, embedder=None, bm25_path: Optional[str] = None):
        """
        Args:
            collection: ChromaDB collection object
            embedder:   SentenceTransformer instance (for generating query embeddings).
                        If None, one is lazily loaded on first query (see
                        _load_embedder) so the query vector always lives in the
                        same space as the stored document vectors.
            bm25_path:  Optional explicit path to the pickled BM25 index. If None,
                        _load_bm25() searches the BEAN_BM25_INDEX env var and a
                        couple of conventional locations (see _BM25_CANDIDATE_PATHS).
        """
        self.collection = collection
        self.embedder = embedder
        self.bm25_path = bm25_path
        self._embedder_load_attempted = embedder is not None
        self._cross_encoder = None
        self._cross_encoder_loaded = False
        self._bm25 = None
        self._bm25_metadata: List[Dict] = []
        self._bm25_load_attempted = False

    def _load_embedder(self):
        """
        Lazily load the query embedder if the caller didn't inject one.

        This matters because ChromaDB's `query_texts=` interface falls back to
        its own default embedding function when no explicit query_embeddings are
        given — a *different* model than whatever built the vector store. That
        mismatch used to go unnoticed because both happened to be
        all-MiniLM-L6-v2 (384-dim). Now that the store is built with
        BAAI/bge-large-en-v1.5 (1024-dim), a mismatched default would raise a
        dimension error, so every caller needs a matching embedder — loaded
        here if one wasn't passed in.
        """
        if self._embedder_load_attempted:
            return
        self._embedder_load_attempted = True
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(EMBED_MODEL_NAME)
            print(f"✓ Query embedder loaded ({EMBED_MODEL_NAME})")
        except Exception as e:
            print(f"⚠ Could not load {EMBED_MODEL_NAME} ({e}); falling back to "
                  f"ChromaDB's default query embedding function, which will NOT "
                  f"match the stored document embedding space.")

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

    def _load_bm25(self):
        """
        Lazily load the BM25 keyword index for hybrid retrieval, if one exists.

        Tries, in order: an explicit bm25_path passed to the constructor, the
        BEAN_BM25_INDEX environment variable, then the conventional vector_db/
        locations every caller in this codebase already uses (see
        _BM25_CANDIDATE_PATHS). If nothing is found, hybrid retrieval is
        skipped entirely and retrieval falls back to dense-only — no caller
        needs to opt in or out explicitly.
        """
        if self._bm25_load_attempted:
            return
        self._bm25_load_attempted = True

        candidates = []
        if self.bm25_path:
            candidates.append(self.bm25_path)
        env_path = os.environ.get("BEAN_BM25_INDEX")
        if env_path:
            candidates.append(env_path)
        candidates.extend(_BM25_CANDIDATE_PATHS)

        for candidate in candidates:
            path = Path(candidate)
            if not path.exists():
                continue
            try:
                with path.open("rb") as f:
                    payload = pickle.load(f)
                self._bm25 = payload["bm25"]
                self._bm25_metadata = payload["metadata"]
                print(f"✓ BM25 index loaded ({path}, {len(self._bm25_metadata)} docs) "
                      f"— hybrid retrieval ON")
                return
            except Exception as e:
                print(f"⚠ Failed to load BM25 index from {path}: {e}")

        print("BM25 index not found; using dense-only retrieval")

    def _hybrid_fuse(self, raw_query: str, dense_chunks: List[Dict], n_candidates: int) -> List[Dict]:
        """
        Merge dense (ChromaDB) and BM25 candidates via Reciprocal Rank Fusion.

        `dense_chunks` were retrieved using the synonym-expanded query — dense
        embeddings handle synonyms well. BM25 gets the raw query here instead:
        expanding "rust" to "Uromyces appendiculatus" for BM25 would add
        keywords that may not appear verbatim in the chunks, defeating the
        point of exact keyword matching.

        rrf_score(doc) = sum(1 / (RRF_K + rank_in_list)) across every ranked
        list the doc appears in (rank is 1-indexed). A doc in both lists sums
        both terms; a doc in only one list gets only that term.
        """
        tokens = tokenize_bm25(raw_query)
        if not tokens:
            return dense_chunks

        scores = self._bm25.get_scores(tokens)
        top_indices = heapq.nlargest(n_candidates, range(len(scores)), key=lambda i: scores[i])
        top_bm25 = [i for i in top_indices if scores[i] > 0]

        rrf_scores: Dict[int, float] = {}
        dense_ranked_ids = [c["global_id"] for c in dense_chunks if c.get("global_id") is not None]
        for rank, gid in enumerate(dense_ranked_ids, start=1):
            rrf_scores[gid] = rrf_scores.get(gid, 0.0) + 1.0 / (RRF_K + rank)
        for rank, idx in enumerate(top_bm25, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank)

        by_id: Dict[int, Dict] = {}
        for c in dense_chunks:
            gid = c.get("global_id")
            if gid is not None:
                entry = dict(c)
                entry.setdefault("bm25_score", None)
                entry["retrieval_channels"] = ["vector"]
                by_id[gid] = entry

        for idx in top_bm25:
            if idx in by_id:
                by_id[idx]["bm25_score"] = round(float(scores[idx]), 4)
                by_id[idx]["retrieval_channels"].append("bm25")
            else:
                meta = self._bm25_metadata[idx]
                entry = dict(meta)
                entry["global_id"] = idx
                entry["distance"] = None
                entry["rerank_score"] = None
                entry["bm25_score"] = round(float(scores[idx]), 4)
                entry["retrieval_channels"] = ["bm25"]
                by_id[idx] = entry

        fused = sorted(by_id.values(), key=lambda c: rrf_scores.get(c["global_id"], 0.0), reverse=True)
        return _dedupe_keep_order(fused)

    def _chromadb_query(self, query_text: str, n_results: int, year_range: Optional[str]) -> List[Dict]:
        """Run ChromaDB query and return normalized chunk dicts."""
        self._load_embedder()
        where = {"year_range": year_range} if year_range else None

        if self.embedder is not None:
            # BGE models are trained asymmetrically: queries get this instruction
            # prefix, documents (embedded in generate_embeddings.py) do not.
            instructed_query = BGE_QUERY_INSTRUCTION + query_text
            emb = self.embedder.encode([instructed_query], normalize_embeddings=True).tolist()
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
        ids = results.get("ids", [[]])[0]
        for doc_id, doc, meta, dist in zip(
            ids,
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
                "chunk_id": meta.get("chunk_id"),
                "distance": round(float(dist), 4),
                "rerank_score": None,
                "bm25_score": None,
                "global_id": _parse_chroma_id(doc_id),
            })
        return chunks

    def retrieve_candidates(
        self,
        query: str,
        n_candidates: int = 20,
        year_range: Optional[str] = None,
    ) -> List[Dict]:
        """
        Retrieve over-sampled candidates.

        Dense retrieval uses the synonym-expanded query; BM25 (if available)
        uses the raw query — see _hybrid_fuse() for why. Falls back to
        dense-only if no BM25 index is loaded.
        """
        self._load_bm25()
        expanded = expand_query(query)
        dense_chunks = self._chromadb_query(expanded, n_candidates, year_range)

        if self._bm25 is None:
            return _deduplicate(dense_chunks)

        try:
            return self._hybrid_fuse(query, dense_chunks, n_candidates)
        except Exception as e:
            print(f"⚠ Hybrid fusion failed ({e}); falling back to dense-only for this query")
            return _deduplicate(dense_chunks)

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
