# Hybrid Retrieval Upgrade (Change 2)

**This document was fully rewritten.** It previously described an early,
opt-in hybrid BM25+vector system (`--hybrid` flag, `data/bm25_index.json`,
`HybridBeanRetriever`, `/hybrid on` CLI command). That system was **deleted
and replaced** by the one described below — every file, flag, and command
the old version of this document referenced no longer exists. If you find a
reference to `src/bm25_index.py`, `src/hybrid_retriever.py`,
`jobs/build_bm25_index.sb`, `jobs/run_ollama_qa_hybrid.sb`, `--hybrid`,
`--bm25-index`, `--trace-file`, or `BEAN_QA_HYBRID` anywhere else (old notes,
memory, a stale branch), it's describing the removed system, not this one.

## Current Status

Hybrid retrieval is **automatic and always on** — there is no flag to enable
it, and no separate class to opt into. `BeanRetriever` (in both
`src/retriever.py` and `hf_space/retriever.py`) fuses dense (ChromaDB) and
BM25 (keyword) candidates via Reciprocal Rank Fusion on every query, and
falls back to dense-only automatically if no BM25 index is present. Every
caller — `hf_space/app.py`, `deploy/app.py`, `qa_with_ollama.py`,
`interactive_qa.py`, `eval_qa.py`, `eval_ragas.py` — gets this for free just
by constructing a `BeanRetriever`; none of them need to know or care whether
hybrid retrieval is active.

## How It Works

1. **BM25 index build** — `build_vector_store.py` builds the BM25 index
   itself, immediately after populating ChromaDB, from the exact same
   in-memory chunk list (guaranteeing identical row ordering between the two
   stores — this is what makes step 4 below possible). Tokenized with
   `tokenize_bm25()` (in `retriever.py`): lowercases, splits on
   hyphens/punctuation, but keeps dots inside alphanumeric runs intact so
   gene IDs (`Phvul.003G108200`) and decimals (`6.5`) survive as one token.
   Saved to `vector_db/bm25_index.pkl` via `rank_bm25.BM25Okapi` — pickled
   alongside `chroma.sqlite3`, so it travels for free through the same HF
   Dataset repo `snapshot_download()` already pulls.
2. **Query-time fusion** — `BeanRetriever.retrieve_candidates()` sends the
   **raw** query to BM25 and the **synonym-expanded** query (`expand_query()`)
   to dense retrieval — expanding "rust" to "Uromyces appendiculatus" helps
   embedding similarity but would make BM25 search for keywords that may
   never appear verbatim in the text, defeating the point of exact-match
   search.
3. **Reciprocal Rank Fusion** — `rrf_score(doc) = Σ 1/(60 + rank)` across
   whichever ranked list(s) the doc appears in (`RRF_K = 60`, the standard
   constant). A chunk found by both channels sums both terms; a chunk found
   by only one gets only that term.
4. **Shared identifier** — the tricky part: each chunk's `chunk_id` field
   resets to 0 for every PDF (it's a per-paper counter from
   `pdf_processor.py`), so it can't identify a chunk globally. RRF instead
   uses ChromaDB's own `"chunk_{i}"` document id, where `i` is the row
   position in `embeddings.npy` — the exact same row order the BM25 corpus
   was built from in step 1. `_parse_chroma_id()` extracts this integer;
   `retriever.py` attaches it to every dense chunk as `global_id`.
5. **Cross-encoder reranking, unchanged** — the fused, deduplicated
   candidate set (typically ~25-35 unique chunks after fusing two ~20-chunk
   lists) goes through the same `ms-marco-MiniLM-L-6-v2` reranker as before
   hybrid retrieval existed. It doesn't know or care which channel(s) found
   each candidate.

## What This Fixed That the Old System Didn't Have

- **Actually worked on HPCC.** The old `HybridBeanRetriever` imported
  `apply_access_control`, `apply_metadata_filters`, `apply_similarity_threshold`,
  and `check_prompt_injection` from `retriever` — functions that only ever
  existed in `hf_space/retriever.py`, not `src/retriever.py`. Every HPCC call
  site wrapped that import in a bare `except Exception`, so `--hybrid` was
  silently a no-op there the entire time it existed.
- **A principled fusion algorithm.** The old system merged candidates by key
  and let the cross-encoder sort out relevance, with a separate ad-hoc
  "hybrid confidence score" formula layered on top. RRF is simpler and is
  the standard, well-understood way to combine two ranked lists.
- **No JSON keyword index to maintain separately.** The old `data/bm25_index.json`
  was a standalone artifact built by its own script (`build_bm25_index.py`),
  which could silently drift out of sync with `processed_chunks.json` if one
  was rebuilt without the other. The new `vector_db/bm25_index.pkl` can't
  drift from ChromaDB because they're built from the same in-memory data in
  the same script run.

## Two Latent Bugs Hybrid Retrieval Surfaced (and fixed)

BM25-only chunks (found by keyword search but never touched by dense
retrieval) have `distance=None`. Several places in the codebase did
`chunk.get("distance", 1.0)` — which returns `None`, not the default, when
the key exists with value `None` — and then compared that against a
threshold, raising `TypeError`. Fixed in:
- `assess_confidence()` and `apply_similarity_threshold()` (retriever.py,
  both copies)
- `build_context()` and `format_references()` (prompts.py, both copies)
- The "out of scope" gate in `hf_space/app.py` and `deploy/app.py` — see
  `_out_of_scope()` in both files, which also now deliberately does NOT
  reject a BM25-only top result just because it lacks a dense distance: if
  BM25 and the cross-encoder both independently ranked a keyword match #1,
  that's its own evidence of relevance.

`_deduplicate()` (distance-sorting) is still used for the pure dense-only
fallback path; a separate `_dedupe_keep_order()` (order-preserving, doesn't
sort by distance) is used whenever fusion actually ran, since sorting
BM25-only entries by a `None` distance isn't meaningful.

## Dependency

`rank-bm25` — pure Python, no compiled dependencies, works fine on HF
Spaces' CPU-only free tier. Pinned in `requirements.txt` and
`hf_space/requirements.txt`.

## Current Limitations

- Confidence scoring (`SUPPORTED`/`PARTIALLY_SUPPORTED`/`INFERRED`/`UNSUPPORTED`)
  is still a heuristic over distance thresholds, and those thresholds were
  tuned for the old `all-MiniLM-L6-v2` embedding space before the Change 1
  model upgrade — see `DOCUMENTATION_INDEX.txt`'s STEP 6 for the open
  follow-up on re-validating them.
- BM25 is a from-scratch `rank_bm25.BM25Okapi` index, not a dedicated search
  engine (Elasticsearch/OpenSearch) — fine at this corpus's scale (~22K
  chunks), but wouldn't necessarily stay simple at a much larger corpus.
- Claim-by-claim citation verification isn't part of this layer — that's a
  separate system (see STEP 13, faithfulness verification, in
  `DOCUMENTATION_INDEX.txt`).
