# Changelog

Meaningful historical changes and incidents, not a record of every edit.
Full narrative detail for any entry lives in `DOCUMENTATION_INDEX.txt`.
Numbering matches the "Change N" convention already established in this
project's history and commit messages — keep using it for new entries.

## Changes

**1 — Embedding model upgrade.** `all-MiniLM-L6-v2` (384-dim) →
`BAAI/bge-large-en-v1.5` (1024-dim). Surfaced a latent query-embedder
mismatch (see Incidents) and a stale-data outage months later (Incidents,
Aug 5).

**2 — Hybrid BM25 + RRF retrieval.** Replaced an entirely separate, silently
non-functional earlier hybrid system. Dense + BM25 fused by Reciprocal Rank
Fusion; automatic, no flag. Surfaced the `None`-distance bug pattern across
five call sites (see `docs/decisions.md`).

**3 — Faithfulness verification (Self-RAG lite).** Second, direct Gemini
call checks generated claims against retrieved context after generation;
soft-fails to a footer, never blocks.

**4 — RAGAS evaluation pipeline.** LLM-judged metrics (faithfulness,
answer_relevancy, context_precision/recall) alongside the older
keyword-rubric scoring. Expanded the benchmark from 25 to 75 questions
(added adversarial and multi-hop categories). Found `eval_qa.py` had never
successfully run a single scored question before this (see Incidents).

**5 — Gene/locus validation.** Three-way labeling (verified /
classical-symbol / not-found) — see `docs/decisions.md` for why two-way
would have been misleading.

**6 — Structured trial-data layer.** LLM-extracted numeric trial data
(yield, disease scores) from paper table chunks into a queryable SQLite DB,
since no real lab spreadsheet exists (checked directly, not assumed).

**7 — Query router.** Rule-based, no LLM call, independent flags (not a
single category) for literature/structured-data/gene-validation. Fixed the
out-of-scope gate to check structured data before giving up.

**8 — Conversation memory.** Server-side, per-session, last 3 turns.
Discovered and fixed a pre-existing bug where the multi-turn history feature
would have silently dropped the current question from what Gemini actually
received (see Incidents, "ERROR 16" equivalent).

**9 — GraphRAG global query mode.** Corpus-wide synthesis over
LLM-clustered "communities" for broad questions no single chunk answers.
Caused a 30-minute startup-timeout incident (see Incidents) before
embeddings were moved to a precomputed, offline step.

**9a — Faithfulness + gene validation extended to global mode.** Closed a
gap where GraphRAG answers had zero verification. `_verify_faithfulness()`
refactored to take `context: str` instead of `chunks: list` so the same
function serves both modes.

**10 — Chat history sidebar + graph/reference UI polish.** Five commits:
real paper titles in references, a GraphRAG graph-rendering fix (see
Incidents), click-to-open-paper on graph nodes, and a localStorage-backed
chat sidebar. First UI work verified against a local headless-browser
(Playwright) repro before pushing — adopted after a blind-push regression
(Incidents).

**11 — RAGAS ground_truth answers.** Populated 11 of 75 questions with
real, corpus-verified reference answers (see `docs/decisions.md` for why not
all 75).

**12 — Unified the two mode routers.** Merged a standalone `_is_global()`
keyword check in `hf_space/app.py` into `query_router.route()` as one more
field on the shared `RoutingDecision` — one call site, one decision object,
instead of two uncoordinated ones. (Superseded by Change 13 for how mode
actually gets chosen; `decision.mode` is still computed, just not read
there anymore.)

**13 — Confidence-based escalation, replacing keyword-based mode
selection.** See `docs/decisions.md` for the full reasoning and the
measured false-positive/negative rates.

**14 — Per-paper structured metadata index.** SQLite table, one row per
paper (962), DOI/year/species/methodology as real filterable fields. Fixed
a real year-extraction bug found along the way (a 2024 paper misread as
2038 via a DOI-suffix regex false positive). Shipped as a standalone data
layer — deliberately not wired into routing yet (see Change 17).

**15 — Bounded gap-check loop (Self-RAG, pre-generation).** See
`docs/decisions.md` for how this layers under, not beside, Change 13's
escalation decision. A post-hoc review caught a real gap in the original
test suite (a "chunk present, distance=None" case was never actually
exercised) — closed by tracing the real code path and adding a test against
the actual `build_context()`, not a simplified stand-in; the other 7 test
fixtures were retroactively fixed to match rather than leaving the new test
as an inconsistent outlier.

**16 — `deploy/app.py` safety-check parity.** Ported the gene-validation
footer, faithfulness check, and gap-check loop to the secondary,
single-model deployment. See `docs/decisions.md` for the verbatim-copy and
dedicated-Gemini-key decisions. Deliberately did NOT port conversation
memory or GraphRAG (real, separate work, not equivalent-sized ports).
Corrected two pre-existing inaccuracies this investigation found in the old
documentation (a false claim that `deploy/app.py` "automatically gets" the
query router, and a false claim about its out-of-scope gate checking
structured data first). **Still uncommitted** as of this writing — the
changes exist only in the working tree.

**17 — Wired `paper_metadata.db` (Change 14) into routing.** Answered a
direct "why isn't this reachable" question by adding species/methodology/
open-ended-year detection to `query_router.route()` (reusing existing
cache-matching machinery, plus one new regex for "papers from 2015+"
phrasing) and uploading the previously-built-but-unshipped database to the
dataset repo.

**18 — Configurable generation model IDs.** Made live generation model names
overridable through environment variables (`GEMINI_MODEL`,
`GROQ_MODEL_PRIMARY`, `GROQ_MODEL_FALLBACK`, `OPENROUTER_MODEL`,
`TOGETHER_MODEL`) after provider-side model deprecations produced
all-providers-failed errors even though retrieval and artifact loading were
healthy. Updated the default Gemini model from `gemini-2.0-flash` to
`gemini-2.5-flash` and the Groq fallback from the deprecated Mixtral model
to `llama-3.1-8b-instant`. Also redacts URL query keys from provider error
logs; the UI was already sanitized, but copied server logs can otherwise
leak the same secret.

**19 — Change 18's own model choice broke, and the fix required actually
verifying with each provider (2026-09-10).** Change 18's replacement
defaults (`gemini-2.5-flash`, `llama-3.3-70b-versatile`,
`llama-3.1-8b-instant`) themselves started 404ing in production — found via
a real user query ("Which genes are associated with anthracnose
resistance...") hitting the sanitized all-providers-failed message, then a
live Space log pull (`HfApi().fetch_space_logs()`) showing the raw 404s.
`_sanitize_provider_error()` only ever logged `str(HTTPError)`
("{status} {reason} for url"), never the response body — which is the only
place a provider actually states *why* a call failed — so a first fix added
truncated (300 char), still-redacted response-body logging before touching
any model name again. That revealed the real cause, confirmed against each
provider's own authoritative source rather than guessed: Gemini's own error
body names `models/gemini-3.6-flash` as the replacement; Groq's
`console.groq.com/docs/deprecations` page confirms both Llama models were
shut down 2026-08-16, recommending `openai/gpt-oss-120b` /
`openai/gpt-oss-20b`. Updated accordingly; verified live with the exact
originally-failing query. **Lesson for next time**: a plausible-sounding
replacement model ID is still a guess — a web search alone returned an
inconsistent, likely-partly-hallucinated model lineup (Gemini "3.5" through
"3.8" flash variants) before the provider's own error response and docs
page resolved it authoritatively.

**Unnumbered — same-day security fix (2026-08-14).** A real user query
(a broad species/methodology filter matching 128 papers) inflated a prompt
past Groq's payload limit; the resulting all-providers-failed error
displayed the raw exception text to the user, which for Gemini's REST
errors embeds the API key in the request URL. Fixed same day:
`call_llm()` now logs full detail server-side only and raises a sanitized
message; the paper-metadata block is capped to the 15 most recent matches.
**The exposed Gemini key still needs rotation** — code fix ≠ credential
rotation.

## Incidents (compressed — full detail in `DOCUMENTATION_INDEX.txt`'s "Key
## Errors" section if needed)

- **SDK env-var hijacking** — HF Spaces' `OPENAI_BASE_URL` silently redirects
  the OpenAI/Groq SDKs to HF's paid router. Fixed by using raw
  `requests.post()` everywhere (see `docs/decisions.md`).
- **Trailing-newline secrets** — HF injects `\n` into secret values,
  corrupting URLs. Fixed with `.strip()` on every secret read.
- **Space stuck "Building" 20+ hours** — the vector store, committed via Git
  LFS into the Space's own repo, blocked every restart's Docker build. Fixed
  by moving it to a separate dataset repo (see `docs/decisions.md`).
- **`eval_qa.py` had never successfully scored one question** — missing
  `import re` caused a `NameError` on every run since the script was
  written. Any historical quality claim predating this fix (Change 4) was
  based on manual spot-checking only.
- **A live HF token was committed in plaintext** in a SLURM job script,
  found during an unrelated review pass. Removed from the file and flagged
  for rotation — removal alone does not invalidate an already-exposed
  credential.
- **Query-embedder mismatch, latent since the embedding-model upgrade** — 
  callers that didn't pass an explicit `embedder=` silently fell back to
  ChromaDB's own default embedding function. Coincidentally harmless while
  both happened to be the same model; broke the moment they diverged. Fixed
  with a lazy-loading default in `BeanRetriever` itself, fixing every
  caller without editing them individually.
- **GraphRAG startup timeout** — synchronous `.encode()` over ~5,130
  community-report texts at Space boot, on CPU, exceeded HF's 30-minute
  health-check window. Fixed by precomputing offline (see
  `docs/decisions.md`).
- **August 5, 2026 — stale 384-dim vector store outage.** The embedding
  model was upgraded in code months earlier, but the actual
  `data/embeddings.npy` / `chroma.sqlite3` were never regenerated to match
  — every live query crashed with a ChromaDB dimension-mismatch error.
  Confirmed and fixed same day: re-ran embedding generation and vector-store
  build, re-uploaded to the dataset repo. Concurrently discovered the
  `bean_llm` conda env's `python3` was a dangling symlink (unrelated
  corruption); SLURM jobs were switched to a different working environment
  (`ptgpu`) rather than repairing `bean_llm` in place.
- **Chat input textbox appeared broken** (no cursor, no typed text) —
  initially conflated with the outage above; actually a separate CSS rule
  hiding Gradio's structural wrapper around the real `<textarea>`, collapsing
  it to 0×0. Confirmed via headless-browser testing, not assumed. This
  incident is why all further Gradio UI work adopted local
  Playwright-verification before pushing.
- **Four undocumented Gradio 6.10 wiring quirks**, found building the chat
  sidebar — summarized in `docs/current-state.md`'s constraints list (never
  set a Chatbot's value from `js=`, `visible=False` removes from the DOM
  entirely, synthetic `input` events only reach Python via `.change()`,
  `fn=None`+`js=`-only output updates silently fail with 2+ textboxes on the
  page).
- **2026-08-14 — the same-day security fix**, see Change log above.
