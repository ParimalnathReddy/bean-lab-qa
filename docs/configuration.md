# Configuration

Env vars, secrets, CLI options, and important defaults — verified against
the actual code on 2026-08-14, not copied from an older doc unchecked.

## Secrets (HF Space Settings → Variables and Secrets)

| Secret name (case-sensitive) | Service | Used for |
|---|---|---|
| `Googlegeminiapi` | Google Gemini | Generation Tier 1, faithfulness check, gap-check loop, RAGAS judge (HPCC only) |
| `groq` | Groq | Generation Tier 2 (two models) |
| `OPENROUTER_API_KEY` | OpenRouter | Generation Tier 3 |
| `TOGETHER_API_KEY` | Together AI | Generation Tier 4 |
| `GEMINI_MODEL` | optional model override | Gemini generation, faithfulness check, gap-check loop, RAGAS default |
| `GROQ_MODEL_PRIMARY` | optional model override | Groq generation Tier 2a |
| `GROQ_MODEL_FALLBACK` | optional model override | Groq generation Tier 2b |
| `OPENROUTER_MODEL` | optional model override | OpenRouter generation Tier 3 |
| `TOGETHER_MODEL` | optional model override | Together generation Tier 4 |

Read as `os.environ.get("Googlegeminiapi", os.environ.get("GEMINI_API_KEY", "")).strip()`
— **the `.strip()` is load-bearing**: HF Spaces injects a trailing `\n` into
secret values, which otherwise corrupts request URLs with `%0A`. None of
these are stored in any repo file (see `docs/decisions.md` for what happens
when one leaks into a committed file anyway).

`deploy/app.py` additionally needs `HF_TOKEN` (for the Qwen InferenceClient)
and its own `Googlegeminiapi`/`GEMINI_API_KEY` (added in Change 16 — separate
dependency from `hf_space/app.py`'s, not shared).

No secrets are needed for gene validation, structured trial-data, or
paper-metadata lookups — all three are self-contained (NCBI/UniProt need no
auth key; SQLite is local).

## Key runtime constants (`hf_space/app.py`)

| Constant | Value | Meaning |
|---|---|---|
| `TOP_K` | 6 | Final chunks returned per retrieval |
| `N_CANDIDATES` | 20 | Candidates fetched before reranking |
| `RATE_LIMIT_REQUESTS` / `RATE_LIMIT_WINDOW` | 8 / 60 | Per-session-hash rate limit (8 questions/60s) |
| `NO_MATCH_THRESHOLD` | 1.10 | Distance above this → out-of-scope |
| `CONVERSATION_TTL` / `CONVERSATION_TURNS` | 1800s / 3 | Conversation-memory retention |
| `_PAPER_METADATA_PROMPT_CAP` | 15 | Max paper-metadata rows formatted into a prompt (added 2026-08-14 after a broad filter blew a payload past a provider's limit) |

## Confidence thresholds (`src/retriever.py`, `assess_confidence()`)

| Label | Condition |
|---|---|
| `SUPPORTED` | best distance ≤ 0.45 and ≥3 relevant chunks |
| `PARTIALLY_SUPPORTED` | best distance ≤ 0.65 and ≥2 relevant chunks |
| `INFERRED` | best distance ≤ 0.85 |
| `UNSUPPORTED` | best distance > 0.85 (or `None`, treated as 1.0) |

Tuned for the original `all-MiniLM-L6-v2` embedding model. Never
re-validated against the current `BAAI/bge-large-en-v1.5` space (see
`docs/current-state.md`).

## Models in use

| Role | Model |
|---|---|
| Embedding (dense retrieval + GraphRAG community search) | `BAAI/bge-large-en-v1.5`, 1024-dim |
| Cross-encoder rerank | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Generation Tier 1 | `gemini-3.6-flash` by default; overridable with `GEMINI_MODEL` |
| Generation Tier 2a/2b | Groq `openai/gpt-oss-120b` / `openai/gpt-oss-20b` by default; overridable with `GROQ_MODEL_PRIMARY` / `GROQ_MODEL_FALLBACK` |
| Generation Tier 3 | OpenRouter `openrouter/auto` |
| Generation Tier 4 | Together `deepseek-ai/DeepSeek-R1` |
| Faithfulness check / gap-check judge | `gemini-3.6-flash` by default; overridable with `GEMINI_MODEL` (direct call, not the waterfall) |
| RAGAS judge (HPCC only) | `gemini-3.6-flash` by default; overridable with `GEMINI_MODEL` or `--gemini-model` |
| Batch/local QA (`qa_with_ollama.py`) | `llama3.1:8b` via local Ollama |

## Interactive CLI flags (`src/interactive_qa.py`)

`/quit`, `/help`, `/year`, `/sources`, `/strict`. (`/hybrid` was removed in
Change 2 — hybrid retrieval is automatic, not a toggle.)

## `query_papers()` / `query_trial_data()` filter parameters

Both (`src/paper_metadata.py`, `src/structured_data.py`) accept plain string
filters, ANDed together, all bound as SQL parameters (never string-formatted
— verified with a real injection string against a test database). Callers
passing a `list` where a single string is expected (e.g. router-detected
species) must index into it (`detected_species[0] if detected_species else
None`) — passing the list itself silently produces a `LIKE` pattern that
matches nothing, since Python's list-repr gets embedded in the query string.
This bug was checked for and confirmed NOT present in the current
`hf_space/app.py` call site (2026-08-14) — but it's an easy regression to
reintroduce if this call site is ever refactored.

## requirements.txt across the three surfaces

- **Repo root** (`requirements.txt`) — HPCC/offline pipeline. Uses **loose**
  version bounds (`chromadb>=0.4.0`, `torch>=2.0.0`, etc.), plus the tightly
  pinned RAGAS stack (`ragas==0.2.15`, `langchain==0.3.25`,
  `langchain-community==0.3.7`, `langchain-core>=0.3.17,<0.4.0`,
  `langchain-google-genai==2.0.11` — verified to actually resolve together;
  don't bump any of these five individually without re-checking the whole
  set still installs).
- **`hf_space/requirements.txt`** — `gradio==6.10.0` (exact pin),
  `chromadb>=1.5.0,<2.0.0`, `sentence-transformers>=2.7.0,<4.0.0`,
  `huggingface-hub>=0.24.0,<2.0.0`, `requests>=2.31.0`,
  `numpy>=1.24.0,<2.0.0` (pinned `<2.0.0` — chromadb incompatibility),
  `pyvis>=0.3.2`, `rank-bm25>=0.2.2`. No RAGAS/langchain — that stack is
  HPCC-only, never deployed.
- **`deploy/requirements.txt`** — same core set as `hf_space/`, plus
  `rank-bm25` (imports `src/retriever.py` directly) and `requests>=2.31.0`
  (added Change 16, Gemini REST calls).

`gene_validator.py`, `structured_data.py`, `query_router.py`,
`paper_metadata.py`, `build_gene_index.py`, and `extract_structured_data.py`
add **zero** new pip dependencies anywhere — pure standard library
throughout.

## HPCC environment

- Conda env name: `bean_llm`, documented path
  `/mnt/scratch/kodumuru/conda/envs/bean_llm` — **broken as of 2026-08-05**
  (dangling `python3` symlink; see `docs/CHANGELOG.md`). SLURM jobs
  currently activate `/mnt/home/kodumuru/.conda/envs/ptgpu` instead.
- **A third, project-local candidate exists and is unverified**:
  `envs/bean_llm/` inside this repo has a real (non-broken) `python3.10`
  binary. Not referenced in any prior fix. Check it before assuming
  `ptgpu` is the only working option.
- GPU node: `dev-amd20-v100` (Tesla V100S, 32GB VRAM).
- Modules: `CUDA/12.1.1`, `git-lfs/3.5.1`, `Ollama/0.15.5`.
- `sbatch` must be run from a dev node, never the gateway/login node
  (confirmed error, not a guess).

## SLURM job resource envelope (`jobs/`)

| Job | Resources | Needs |
|---|---|---|
| `process_pdfs.sb` | 8 CPU, 32GB, 4h, no GPU | — |
| `generate_embeddings.sb` | 4 CPU, 32GB, 2h, 1 GPU | ptgpu env |
| `build_vector_store.sb` | 4 CPU, 16GB, 1h, no GPU | still points at broken `bean_llm` as of last check — needs the same ptgpu switch |
| `build_gene_index.sb` | 2 CPU, 8GB, 1h, no GPU | outbound HTTPS to NCBI/UniProt |
| `extract_structured_data.sb` | 8 CPU, 32GB, 8h, 1 GPU | Ollama; run `--dry-run` first |
| `run_ollama_qa.sb` | 8 CPU, 32GB, 4h, 1 GPU | Ollama |
| `run_ragas_eval.sb` | 8 CPU, 32GB, 4h, 1 GPU | Ollama + `Googlegeminiapi` exported before `sbatch` |
| `build_graphrag.sb` | 8 CPU, 32GB, up to 4 days, 1 GPU, partition `general-long-gpu` | fully local via Ollama, no API keys |

## `.gitignore` — not tracked, don't expect them in a fresh clone

`data/pdfs/`, `vector_db/`, `models/`, `graphrag_workspace/`, `logs/`.
