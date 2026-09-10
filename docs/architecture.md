# Architecture

How the system works, mechanism-level. Not a chronology — see
`docs/CHANGELOG.md` for that. Every claim here was checked against the code
on 2026-08-14; if you change the code, update this file in the same edit.

## Three tiers

```
OFFLINE (HPCC, batch, mostly GPU)
    PDFs → chunks → embeddings → vector store + BM25 index
                                → gene index (separate)
                                → structured trial-data DB (separate)
                                → GraphRAG community reports + paper graph (separate)
                                → per-paper metadata index (separate)
         all pushed to →
STORAGE (one shared HF dataset repo: Parimalanath/bean-lab-vector-db)
         pulled whole, at boot, via snapshot_download() →
RUNTIME (HF Space, hf_space/app.py, live, CPU-only, 24/7)
    query_router.route() → hybrid retrieval → gap-check loop →
    escalation decision → LLM generation waterfall → verification → answer
```

## Offline build pipeline

Each stage is a separate script/job, all independent of each other except
where noted, all writing into `vector_db/` or `graphrag_workspace/output/`:

| Stage | Script | Output |
|---|---|---|
| PDF → chunks | `src/pdf_processor.py` (`jobs/process_pdfs.sb`) | `data/processed_chunks.json` |
| Chunks → embeddings | `src/generate_embeddings.py` (`jobs/generate_embeddings.sb`, GPU) | `data/embeddings.npy`, `data/embeddings_metadata.json` |
| Embeddings → vector store | `src/build_vector_store.py` (`jobs/build_vector_store.sb`) | `vector_db/chroma.sqlite3`, `vector_db/*.bin`, `vector_db/bm25_index.pkl` (same script builds both dense and BM25 in one pass, from the same in-memory chunk list, so they can never drift out of row-order sync) |
| Gene index | `src/build_gene_index.py` (`jobs/build_gene_index.sb`) | `vector_db/gene_index.pkl` — NCBI Gene + UniProt, *P. vulgaris* only |
| Structured trial data | `src/extract_structured_data.py` (`jobs/extract_structured_data.sb`, GPU + Ollama) | `vector_db/trial_data.db` — LLM-extracted from table-shaped chunks |
| GraphRAG | `jobs/build_graphrag.sb` → `scripts/export_graphrag_summaries.py` → `scripts/build_paper_graph.py` → `scripts/embed_community_reports.py` | `graphrag_workspace/output/community_reports.json`, `paper_graph.json`, `community_embeddings.npy` |
| Per-paper metadata | `scripts/build_paper_metadata.py` | `vector_db/paper_metadata.db` — one row per paper (962), DOI/year/species/methodology |

All outputs converge into one flat push to the `Parimalanath/bean-lab-vector-db`
dataset repo (`hf_dataset_upload/`, a local clone). The GraphRAG artifacts
don't physically live in `vector_db/` locally, but end up in the same
dataset repo alongside everything else.

**A chunking gap that affects both structured-data and GraphRAG extraction**:
`pdf_processor.py` keeps a detected table as one self-contained chunk, but a
table's caption — which usually states the location/year/study the numbers
belong to — is ordinary text, so it lands in a *different* chunk than the
table itself. `extract_structured_data.py` works around this by sending each
candidate table chunk **plus its immediately-neighboring chunks** (same
source file, adjacent order) to the LLM as context, with an explicit
instruction to extract values only from the target passage but use the
neighbors to resolve location/year/study identification. Any future code
that processes a table chunk in isolation will hit the same gap.

## Runtime request pipeline (`hf_space/app.py`'s `chat()`)

`chat()` is a **plain synchronous generator** (not `async def`) — status
messages stream to the UI via `yield` at each stage.

### 1. Routing — `query_router.route(message)`

Deterministic, no LLM call, runs once per question on the raw message (not
the conversation-augmented retrieval query). Sets independent boolean flags
(a question can be a literature question, a structured-data question, and a
genetics question simultaneously — never a single mutually-exclusive
category) plus the actual matched entities:

- `use_literature` (~always True)
- `use_structured_data` + matched cultivar/location/trait/year
- `run_gene_validation` + matched gene mentions (reuses
  `gene_validator.extract_gene_mentions()` directly, not reimplemented)
- `use_paper_metadata` + matched species/methodology/year-filter
- `mode` (`"local"`/`"global"`) — **computed but not read** by `chat()`
  since Change 13; see "Escalation" below for what actually decides mode.

Entity matching against cultivar/location/trait/species/methodology checks
the **real current values** in the database (cached as frozensets), never a
hardcoded keyword list — this is a deliberate, repeated pattern in this
codebase (a hardcoded list would false-trigger constantly and drift out of
sync with real data).

### 2. Retrieval — `BeanRetriever.retrieve()` (`src/retriever.py` /
`hf_space/retriever.py`, diverged copies)

Runs **unconditionally** for every non-greeting question, concurrently with
structured-data/paper-metadata lookups via `ThreadPoolExecutor` (both are
I/O-bound — SQLite reads and a ChromaDB query — so threads give real
concurrency despite the GIL, in a generator with no async context to plug
into).

1. `expand_query()` — augments the query with scientific synonyms
   (`rust` → `Uromyces appendiculatus`, etc.) for the **dense side only**;
   BM25 gets the raw query, since expansion would make keyword search look
   for terms unlikely to appear verbatim.
2. Dense: ChromaDB top-20 via cosine similarity (BAAI/bge-large-en-v1.5,
   query side gets the `BGE_QUERY_INSTRUCTION` prefix, document side
   doesn't — asymmetric embedding convention).
3. BM25: `rank_bm25.BM25Okapi` top-N by keyword score (if the index is
   present; silently falls back to dense-only if not).
4. Fusion: Reciprocal Rank Fusion across both lists, keyed by ChromaDB's own
   `chunk_{i}` row-position id (not the per-chunk `chunk_id` metadata field,
   which resets to 0 per PDF and isn't globally unique).
5. Cross-encoder rerank (`ms-marco-MiniLM-L-6-v2`) over the fused,
   deduplicated candidates → top-6.
6. `assess_confidence()` — labels the result `SUPPORTED` /
   `PARTIALLY_SUPPORTED` / `INFERRED` / `UNSUPPORTED` from the best chunk's
   L2 distance (thresholds 0.45/0.65/0.85 — tuned for an earlier, smaller
   embedding model; never re-validated against the current
   `bge-large-en-v1.5` space, see `docs/current-state.md`).

A BM25-only chunk has `distance=None`. Every place that compares distance
numerically treats `None` as worst-case (1.0), never crashes — see
`docs/decisions.md` for why this needed a deliberate fix pattern, not just a
default parameter.

### 3. Gap-check loop (Self-RAG, pre-generation) — `_check_context_sufficiency()`
/ `_gap_check_retrieve()`

Triggers when confidence isn't `SUPPORTED` and no structured-data rows
already answer the question. One Gemini call judges sufficiency and, if
insufficient, suggests a reformulated search query; retrieval reruns with
that query. Capped at 2 iterations. Soft-fails to "sufficient, stop" on any
Gemini error — can only fail to improve an answer, never make one worse.
Lives entirely inside the local-retrieval branch; structurally cannot touch
GraphRAG.

### 4. Escalation decision — `should_escalate` (computed inline in `chat()`)

```
escalate = global_available AND trial_rows is empty AND
           confidence != "SUPPORTED" AND NOT none_distance_artifact
```

Confidence-based, not keyword-based (see `docs/decisions.md` for the full
reasoning and the false-positive rate keyword-based routing produced).
`none_distance_artifact` guards specifically against a `None`-distance top
chunk (which `assess_confidence()` can only ever label `UNSUPPORTED`, never
anything else) being mistaken for a genuine low-confidence signal — a real
BM25-only match with real text isn't necessarily bad evidence, it's just
unmeasured. Escalation is full replacement, not a merge: either the local
answer generates, or the global one does — never both, never combined.

`"local"` → answer from the retrieved chunks directly.
`"global"` (GraphRAG) → `_search_communities()` embeds the question, finds
the top-5 most similar precomputed community-report embeddings
(`community_embeddings.npy`, computed offline — see `docs/decisions.md` for
why this must never be computed at Space-boot time), and synthesizes from
those summaries instead of raw chunks. `_render_graph()` renders a
highlighted subgraph via pyvis, wrapped in `<iframe srcdoc>` (see
`docs/current-state.md`'s constraints).

### 5. Generation — `call_llm()`

Waterfall, first success wins: Gemini → Groq primary → Groq fallback →
OpenRouter → Together. The concrete model IDs are defaults with environment
variable overrides (`GEMINI_MODEL`, `GROQ_MODEL_PRIMARY`,
`GROQ_MODEL_FALLBACK`, `OPENROUTER_MODEL`, `TOGETHER_MODEL`) because
provider-side model availability changes independently of this codebase.
Raw `requests.post()` throughout, never an SDK (see `docs/decisions.md`).
On total failure, logs full per-provider detail server-side and raises a
**sanitized** generic message — never the raw exception text, which can
embed request URLs (and therefore API keys).

`build_messages()` / `build_messages_with_history()` (`prompts.py`, kept
byte-identical between `src/` and `hf_space/`) assemble the prompt: system
prompt + up to 3 prior conversation turns (plain question/answer text only,
no stale retrieved context) + the current turn's retrieved-sources block +
an optional pre-formatted `trial_data_block` string. `prompts.py`
deliberately doesn't know the row shape of what produced that block —
paper-metadata results (Change 17) get concatenated into the same string
alongside trial-data results, with zero changes to `prompts.py` needed.

### 6. Verification

`_verify_faithfulness(context, answer)` — a second, direct Gemini call
(bypassing the waterfall — a cheap secondary check shouldn't compete with
generation for scarcer Groq/OpenRouter/Together quota) checks each claim
against the retrieved context, appends a warning footer for anything
unsupported/contradicted. Runs identically for local and global mode
(`context` is a plain string either way, not a chunk list — this
generalization is *why* the signature takes `context: str`, not `chunks`).

`_verify_gene_mentions(answer)` — free (regex + dict lookup, no LLM call),
runs **unconditionally**, not gated by the router's `run_gene_validation`
flag. A query with no genetics vocabulary can still produce an answer that
names a gene; catching a hallucinated one there costs nothing extra.

Both are soft-fail-only footers — never block or replace the primary answer.

## Conversation memory (Space only)

Server-side `_conversations` dict, keyed by `request.session_hash` — **not**
the Gradio UI `history` list, which contains display-only content (status
placeholders, footer-laden final answers) that shouldn't be replayed to the
model as if it were real conversation content. Last 3 turns, 30-minute TTL,
swept by a background thread every 5 minutes. A short retrieval-side
heuristic (pronoun or ≤6-word message) decides whether to prepend the prior
question to the current retrieval query — not an LLM call, and not applied
unconditionally (would dilute retrieval precision for self-contained
questions).

## What's NOT wired together

- Structured trial-data, paper-metadata, and GraphRAG are three independent
  subsystems. A single query can activate literature + structured-data +
  gene-validation simultaneously (all independent flags), but `mode` is
  still exactly one of `"local"`/`"global"` — there's no concurrent
  local+global execution or answer-merging. Deliberately deferred scope, not
  an oversight (see `docs/decisions.md`).
- `deploy/app.py` has its **own, separately maintained copies** of
  `_call_gemini()`, `_verify_faithfulness()`, `_verify_gene_mentions()`, the
  gap-check functions — verbatim-copied from `hf_space/app.py`, not a shared
  module. Deliberate: refactoring already-live, already-verified production
  code into a shared module purely for DRY would be unrequested risk to the
  live Space. `deploy/app.py` has no `query_router.py`, no GraphRAG, no
  conversation memory, and no paper-metadata wiring at all.
- `qa_with_ollama.py` / `eval_qa.py` / `eval_ragas.py` (the HPCC batch path)
  get routing, structured-data, and gene-validation "for free" since they
  call the same `answer_question()` — but never GraphRAG (Space-only
  artifacts) and never conversation memory (no session concept in a batch
  run).
