# Current State

> Read this first, every session. It describes the project AS IT ACTUALLY IS,
> verified against the code — not as originally designed, not as a changelog
> claims. If anything here conflicts with `DOCUMENTATION_INDEX.txt` or a
> README, THIS FILE WINS; go re-verify and fix the older doc instead of
> trusting it. Last verified against source: 2026-09-10.

## What this project does

A RAG (Retrieval-Augmented Generation) question-answering system over ~1,067
bean/legume crop-science papers (1961–2026) for the MSU Bean Lab. A
researcher asks a question in plain English; the system retrieves the most
relevant passages, cites them by DOI, and checks its own answer for
unsupported claims before showing it. Live at
[Parimalanath/bean-lab-qa](https://huggingface.co/spaces/Parimalanath/bean-lab-qa)
(Hugging Face Spaces, free tier, Gradio UI).

## The three code surfaces — know which one you're in

| Surface | Repo | Status | Notes |
|---|---|---|---|
| `src/` | main GitHub repo (`ParimalnathReddy/bean-lab-qa`) | Batch/offline pipeline + shared logic | Used by `qa_with_ollama.py`, `eval_qa.py`, `eval_ragas.py`. No web UI. |
| `hf_space/` | **its own separate git remote** (`huggingface.co/spaces/Parimalanath/bean-lab-qa`), also mirrored into the main repo | **LIVE — this is the deployed app** | `git -C hf_space push` is what actually ships changes to users. Pushing in the main repo does NOT touch the Space. |
| `deploy/` | main GitHub repo only | **Not deployed anywhere** — confirmed via a Hugging Face Hub search: the account has exactly one Space, and it's `hf_space/`'s | Single-model (Qwen via HF InferenceClient) standalone alternative. Change 16's safety-check port is now committed (`08c6151`), but the app is still not deployed anywhere — this is a code-availability fact, not a "still uncommitted" one anymore. |

`hf_space/query_router.py` and `hf_space/prompts.py` are kept **byte-identical**
to their `src/` counterparts (verified via `diff`, 2026-08-14). `hf_space/retriever.py`
has **diverged** from `src/retriever.py` — it has extra methods
(`apply_access_control`, `apply_metadata_filters`, `check_prompt_injection`)
`src/retriever.py` doesn't have. Both copies get retrieval upgrades applied
independently; don't assume they match without checking.

## Request pipeline (live Space, `hf_space/app.py`, `chat()`)

1. **Route** — `query_router.route(message)`: deterministic, no LLM call. Sets
   flags (`use_literature`, `use_structured_data`, `run_gene_validation`,
   `use_paper_metadata`) and extracts entities (cultivar/location/trait/gene/
   year/species/methodology) against real database vocabulary, never a
   hardcoded list.
2. **Retrieve** — hybrid dense (ChromaDB, BAAI/bge-large-en-v1.5) + BM25,
   fused by Reciprocal Rank Fusion, cross-encoder reranked
   (`ms-marco-MiniLM-L-6-v2`). Runs **unconditionally** for every non-greeting
   question — mode is never pre-decided from keywords. Structured trial-data
   lookup and paper-metadata lookup run alongside it.
3. **Gap-check loop** (Self-RAG, pre-generation) — one Gemini call asks
   whether retrieved evidence is sufficient; if not, reformulates the query
   and retries, capped at 2 iterations.
4. **Escalation decision** — computed from the resulting confidence label
   (`SUPPORTED`/`PARTIALLY_SUPPORTED`/`INFERRED`/`UNSUPPORTED`), **not**
   keywords. Only escalates to GraphRAG ("global" mode) when confidence
   isn't `SUPPORTED`, no structured-data rows rescue it, and GraphRAG data is
   actually available.
5. **Generate** — `call_llm()`: Gemini → Groq primary → Groq fallback →
   OpenRouter → Together, first success wins. Model IDs are overridable
   through Space environment variables (see `docs/configuration.md`) because
   provider-side model availability changes — and has already broken
   production twice in one month (2026-08 and 2026-09, see
   `docs/CHANGELOG.md` Changes 18/19). Treat the current hardcoded defaults
   as provisional, not permanent — see "Known problems" below.
6. **Verify** — a second Gemini call checks every claim against retrieved
   context (faithfulness footer); a separate, free (no LLM) regex+dict check
   validates any gene/locus mentions. Both soft-fail — they only ever append
   a footer, never block or alter the primary answer.

## Configuration highlights

See `docs/configuration.md` for the full list. The two things most likely to
bite you:
- Secrets are read with `.strip()` — HF injects a trailing `\n` into secret
  values that otherwise corrupts request URLs.
- LLM calls use raw `requests.post()`, never the OpenAI/Groq SDKs — HF Spaces
  sets `OPENAI_BASE_URL`, which those SDKs silently honor, redirecting calls
  to HF's paid router (`402 Payment Required`).

## Known problems (verified 2026-09-10, not assumed)

- **The exposed Gemini API key (2026-08-14 incident) still needs rotation.**
  The code fix (sanitized errors) is live and verified, but a code fix is
  not credential rotation — the key was shown in a public chat UI at least
  once and should be treated as compromised. This is a manual step in
  Google AI Studio / Cloud Console that no one has confirmed doing.
- **Hardcoded generation model IDs are a recurring, proven failure mode, not
  a one-off.** Twice in one month (2026-08-14 → Change 18, 2026-09-10 →
  Change 19), a provider retired/deprecated a model this system called by a
  fixed default, and both times it silently broke the entire generation
  waterfall until someone happened to trigger and investigate the sanitized
  "all providers unavailable" message. There's no automated check that
  catches this before a real user does — this is a real, unaddressed gap in
  monitoring, not just a config problem. See `docs/decisions.md` for the
  "verify a replacement model ID against the provider's own source, never a
  general web search" lesson from Change 19 — that discipline matters again
  the next time this happens, and it will happen again.
- **`gene_index.pkl` and `trial_data.db` were never uploaded** to the
  `Parimalanath/bean-lab-vector-db` dataset repo — confirmed against its live
  file listing. Gene validation degrades to allowlist-only (classical
  symbols like Co-1 still work; NCBI/UniProt cross-referencing doesn't).
  Structured trial-data lookup returns empty on every call. Both are fully
  coded and tested — this is a deployment gap, not a code gap. Unchanged
  this session — still open.
- **Query logging is broken** — the `Parimalanath/bean-lab-query-logs`
  dataset repo was never created; `_flush()`'s upload has been silently
  failing since it was written. No usage telemetry exists. Unchanged this
  session — still open.
- **A project-local `envs/bean_llm/` directory exists** (this repo's own
  `envs/bean_llm/bin/python3.10`, real binary, dated May 18 2026) that is
  **not referenced anywhere** in prior documentation or in the incident
  (ERROR 19 in the changelog) that switched SLURM jobs to the `ptgpu`
  environment after the scratch-path `bean_llm` env broke. Unverified whether
  this local copy is a working alternative — found during this
  documentation audit, not investigated further. Check before assuming
  either environment is authoritative.
- **`requirements.txt` (repo root) uses loose version bounds**
  (`chromadb>=0.4.0`, etc.) that don't match the exact pins the old docs
  claimed were "authoritative" (`chromadb==1.5.4`, `torch==2.3.0+cu121`,
  etc.). The loose file is what's actually in the repo; treat any "exact
  pinned version" claim elsewhere as aspirational unless re-verified.
- **RAGAS ground_truth is populated for only 11 of 75 benchmark questions**
  — deliberate (see `docs/decisions.md`), not an oversight, but it means
  `context_recall`/`context_precision` are only meaningful for that subset.
- **`config/` also contains `module_setup.sh`**, not mentioned in the old
  repo-tree documentation (only `HPCC_SETUP_GUIDE.md` was listed there).

## Constraints — do not accidentally revert these

- **Never re-introduce an SDK (OpenAI/Groq) for LLM calls.** Use
  `requests.post()` with hardcoded URLs (see `docs/decisions.md`).
- **Never let a `.get("distance", 1.0)`-style default sneak back in.** A
  BM25-only chunk has `distance` present but explicitly `None`; `.get()`'s
  default only fires when the key is *missing*. Use the two-line
  `d = x.get("distance"); d = 1.0 if d is None else d` pattern everywhere
  distance is compared numerically.
- **Never display a raw caught exception to the user.** `call_llm()` logs
  full provider-failure detail server-side only and raises a sanitized
  message — a `requests.HTTPError`'s `str()` includes the full request URL,
  which for Gemini embeds the API key as a `?key=...` query param. This was a
  real, live security leak (see `docs/CHANGELOG.md`, 2026-08-14 fix), not a
  hypothetical.
- **Never call `gr.HTML()` for anything containing `<script>` tags without
  wrapping it in `<iframe srcdoc="...">`.** Gradio's `gr.HTML()` inserts
  markup inertly; scripts never execute unless framed.
- **Never assume `gr.Textbox(visible=False)` behaves like CSS
  `display:none`.** It removes the element from the DOM entirely. Hidden
  carrier textboxes need `visible=True` + CSS positioning instead.
- **Never read a `gr.Chatbot`'s value via `inputs=[chatbot]` in a `.then()`
  step immediately after that same chatbot was updated in the preceding
  step** — confirmed to silently fail. Yield the value you need as an
  explicit extra output from the same event instead.
- **Keep `hf_space/query_router.py` and `hf_space/prompts.py` byte-identical
  to their `src/` counterparts** after any edit to either — `diff` them
  before considering a change to either file done.
- **Never remove `ssr_mode=False` from `demo.launch()`.** Gradio 6.10's
  server-side rendering hangs Hugging Face's health check indefinitely,
  keeping the Space stuck in "Building." This looks like dead/defensive code
  if you don't know the history — it isn't.
- **PDF filenames encode DOIs**: a slash in a DOI becomes an underscore in
  the filename (`10.2135_cropsci2004.1799.pdf` ↔
  `10.2135/cropsci2004.1799`). Multiple modules reconstruct DOIs from
  filenames this way (`_filename_to_doi()` in `retriever.py`, reused by
  `build_gene_index.py`, `build_vector_store.py`,
  `extract_structured_data.py`) — don't reintroduce a different filename
  convention without updating all of them.
- **Never log only `str(requests.HTTPError))` when diagnosing a provider
  call failure.** It's just `"{status} {reason} for url: {url}"` — never the
  response body, which is usually the only place a provider states *why* a
  call failed. `_sanitize_provider_error()` (`hf_space/app.py`) appends up
  to 300 redacted characters of the body for exactly this reason; don't
  strip that back down to just the status line.
- **Never pick a replacement provider model ID from a general web search
  alone.** Confirm it against the provider's own error response or
  official docs/deprecations page first. A general search returned an
  inconsistent, likely-hallucinated model lineup during the Change 19
  investigation; the provider's own sources were correct and consistent.
- **Never blend retrieval-quality and generation-quality into one
  evaluation score.** `eval_qa.py`/`eval_ragas.py` report them as two
  separate layers on purpose (Change 20, see `docs/decisions.md`) — a
  single blended number hides which half of the pipeline actually needs
  attention.

## Current goals / work in progress

**Recently completed (this session, 2026-09-10):**
- Diagnosed and fixed a live all-providers-failed incident: Change 18's own
  replacement model IDs (`gemini-2.5-flash`, two Groq Llama models) had
  themselves already been deprecated/retired by the time the fix was
  deployed. Root-caused via real Space logs
  (`HfApi().fetch_space_logs()`) plus a diagnostic patch to capture
  provider response bodies (previously only the HTTP status was logged),
  then fixed with model IDs confirmed against each provider's own error
  response / docs page — not guessed. Verified live with the exact
  originally-failing query. See Changes 18–19 in `docs/CHANGELOG.md`.
- Committed six sessions' worth of pending work that had been sitting
  uncommitted in the main repo (Changes 9, 14, 16, 17, the `ptgpu`
  environment fix, and this shared `docs/` system itself) — the main repo
  is now clean except for gitignored-in-spirit `data/`/`Graph_files/`.
- Built this shared `docs/` system (`current-state.md`, `architecture.md`,
  `configuration.md`, `decisions.md`, `CHANGELOG.md`) plus `CLAUDE.md`/
  `AGENTS.md`, consolidating knowledge that previously existed only in
  `DOCUMENTATION_INDEX.txt` and conversation history.
- Separated retrieval-quality and generation-quality scoring in both eval
  scripts into two independent, never-blended layers (Change 20) — see
  `docs/decisions.md`.

**Not started, real next candidates:**
- Rotate the exposed Gemini API key (2026-08-14 incident) — manual,
  human-only step, still outstanding.
- Add a way to catch a provider model deprecation before a real user does —
  no monitoring/alerting exists today; this has now broken production
  twice.
- Upload `gene_index.pkl` and `trial_data.db` to activate those two dormant
  subsystems.
- Re-validate the confidence-label distance thresholds (0.45/0.65/0.85)
  against the current `bge-large-en-v1.5` embedding space — never
  re-validated since the embedding-model upgrade.
- Fix query logging (dataset repo was never created).

## Where to look next

- **How the system works, mechanism-level**: `docs/architecture.md`
- **Every env var, secret, config default**: `docs/configuration.md`
- **Why something is built the way it is**: `docs/decisions.md`
- **What changed, when, and why**: `docs/CHANGELOG.md`
- **The full narrative, step-by-step build log**: `DOCUMENTATION_INDEX.txt`
  (3,800+ lines — the original, exhaustive record this file and its siblings
  were distilled from; still useful for deep historical detail, but this
  file is the one to trust for current state)
