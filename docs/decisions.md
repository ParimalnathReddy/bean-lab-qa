# Decisions

Important architectural/design decisions, extracted from `DOCUMENTATION_INDEX.txt`,
the code, and git history. Each: what was decided, why, what it affects, and
what shouldn't change without reconsidering the whole decision — not just
the line that looks wrong in isolation.

---

## Provider error bodies are always logged server-side (truncated, redacted) —
## never inferred from HTTP status alone

**Decision**: `_sanitize_provider_error()` appends up to 300 characters of a
`requests.HTTPError`'s response body to what gets logged server-side, on top
of the plain `str(error)`.

**Why**: `str(requests.HTTPError)` is just `"{status} {reason} for url:
{url}"` — it never includes the body, which is usually the only place a
provider states *why* a call actually failed. This was found the hard way:
a live all-providers-failed incident (2026-09-10) showed two 404s in the
logs with no way to tell whether either was "model doesn't exist" or
"endpoint path changed" until this logging was added and redeployed. The
body is still redacted with the same regexes as the rest of the message and
still never reaches the user-facing error — this only changes what
server-side logs capture.

**Affects**: `_sanitize_provider_error()` in `hf_space/app.py`. `deploy/app.py`
does not have this addition yet (see the earlier decision on verbatim,
not-shared, safety-check copies between the two apps) — port it there too
if `deploy/app.py`'s waterfall is ever debugged for the same class of
failure.

**Don't**: revert to logging only `str(error)` for an HTTP-based provider
call — that regresses to exactly the "can't tell why" state that made the
2026-09-10 incident slower to diagnose than it needed to be.

---

## A provider's replacement model ID is verified against that provider's own
## error response or docs — never picked from a general web search alone

**Decision**: when a hardcoded model ID starts failing, the replacement is
confirmed against the *provider's own* authoritative source — the actual
error response body from calling the API, or that provider's own
docs/deprecations page — before being deployed.

**Why**: Change 18 replaced a deprecated model ID with another one
(`gemini-2.5-flash`) that itself turned out to already be invalid by the
time it shipped — a plausible-sounding guess, not verified against a
first-party source. Diagnosing the follow-up incident (Change 19), a
general web search for "current Gemini model names" returned an
inconsistent, likely partly-hallucinated list (rapid-fire "3.5" through
"3.8" flash variants). The actual fix came from two first-party sources
instead: Gemini's own 404 response body, which named the exact replacement
model directly, and Groq's own `/docs/deprecations` page, which gave exact
shutdown dates and recommended replacements.

**Affects**: any future generation-model swap in `call_llm()`'s waterfall
(`hf_space/app.py`, `deploy/app.py`) or the RAGAS judge model
(`src/eval_ragas.py`).

**Don't**: deploy a replacement model ID based on a plausible-sounding
general web search result alone — confirm it against the provider's own
error response or documentation page first, the same way Change 19 did.

---

## Raw `requests.post()` instead of the OpenAI/Groq SDKs

**Decision**: every LLM call uses `requests.post()` with a hardcoded URL, in
every one of `call_llm()`'s four tiers.

**Why**: Hugging Face Spaces sets `OPENAI_BASE_URL` in its environment,
pointing at HF's own paid inference router. The OpenAI SDK and the Groq SDK
both silently honor that env var and redirect every call to HF's router
instead of the real API, producing `402 Payment Required`. Raw
`requests.post()` ignores the env var entirely.

**Affects**: `_call_gemini()`, `_call_groq()`, `_call_openrouter()`,
`_call_together()` in both `hf_space/app.py` and `deploy/app.py`.

**Don't**: reach for an official provider SDK to "simplify" any of these
calls without first checking whether `OPENAI_BASE_URL` is still set in the
deployment environment.

---

## `None`-distance handling — never `.get("distance", 1.0)`

**Decision**: every place a chunk's distance is compared numerically uses
`d = x.get("distance"); d = 1.0 if d is None else d`, never a bare
`.get("distance", 1.0)`.

**Why**: a BM25-only chunk (hybrid retrieval, Change 2) has the `distance`
key **present** with an explicit value of `None` — not missing. `.get()`'s
default only applies when the key is absent, so `.get("distance", 1.0)`
returns `None` in exactly this case, and a subsequent `> ` comparison
crashes. This was caught in code review before it ever reached production,
across five separate call sites (`assess_confidence()`,
`apply_similarity_threshold()`, `build_context()`, `format_references()`,
both apps' `_out_of_scope()` gates) — it's a pattern that recurs anywhere
new distance-comparing code is added, not a single bug that was fixed once.

**Affects**: any new code that reads a chunk's `distance` field.

**Don't**: add a new distance comparison using `.get(..., default)` without
the explicit `is None` check.

---

## Confidence-based escalation, not keyword-based (Change 13)

**Decision**: `hf_space/app.py`'s local/global mode decision is computed
**after** local retrieval runs, from the resulting confidence label — not
from a keyword scan of the question before retrieving anything.

**Why**: a fixed keyword list (`_GLOBAL_SIGNALS`) can only guess whether a
question is well-covered locally from its phrasing. Measured against 23 real
test queries with independently-verified corpus coverage: keyword-based
routing (and the initially-considered "escalate on `UNSUPPORTED` alone")
both misclassified real cases — a confirmed well-covered question ("nitrogen
fixation rates") landed `UNSUPPORTED` purely from a `None`-distance artifact
(see the next entry), and "escalate on UNSUPPORTED alone" only caught 1 of 9
confirmed-thin questions. The shipped rule (`confidence != "SUPPORTED"`,
with the `None`-distance guard) caught 6 of 9.

**Affects**: `hf_space/app.py`'s `chat()` — `mode` starts `"local"` and the
only path to `"global"` is this escalation check.

**Don't**: add `decision.mode` (the keyword signal, still computed by
`query_router.route()` for other potential consumers) back into this
decision with an `OR`. That was tried, rejected explicitly: "confidence low
OR keyword suggests breadth" means a keyword-matched-but-well-covered query
*always* escalates, defeating the entire point of measuring instead of
guessing.

---

## The `None`-distance escalation guard is deliberately asymmetric

**Decision**: `should_escalate` explicitly refuses to escalate when the
label is `UNSUPPORTED` *and* the top chunk's distance is `None` — but
Change 15's gap-check loop (which runs earlier, and can affect what
`should_escalate` sees) deliberately does **not** carry this same guard.

**Why**: a `None` distance means "unmeasured," not "bad" — the guard exists
because raw distance math can't tell the difference. The gap-check loop's
classifier reads the actual chunk text directly (a qualitative judgment, not
distance math), so it doesn't have the same blind spot the guard is
protecting against. Applying the guard there too would have turned a case
the gap-check loop can genuinely improve into a dead end instead.

**Affects**: `_gap_check_retrieve()` vs. the `should_escalate` computation,
both in `hf_space/app.py`.

**Don't**: assume every place that reads a `None`-distance chunk needs the
identical guard — check whether the guard is compensating for distance math
specifically, or whether the actual signal being used has already sidestepped
that blind spot.

---

## Gap-check loop is layered *underneath* escalation, not beside it

**Decision**: Change 15's pre-generation sufficiency check runs strictly
before Change 13's escalation decision, and may only improve the
`(chunks, confidence)` tuple that decision reads — it never makes an
escalate/don't-escalate call of its own.

**Why**: Change 12 was built specifically to eliminate the problem of two
uncoordinated decision-makers acting on the same question (the old
`_is_global()` vs. `query_router.route()` split). Giving the gap-check loop
its own competing "is this good enough" verdict would recreate exactly that
problem one layer down.

**Affects**: the ordering in `chat()` — gap-check, then escalation
computation, never the reverse, never in parallel.

**Don't**: add a second, independent "should we do something different here"
check into this pipeline without first checking whether it can instead be
expressed as a refinement layer feeding an existing decision point.

---

## Structured-data / paper-metadata detection matches against real database
## values, never a hardcoded keyword list

**Decision**: cultivar, location, trait, species, and methodology matching
in `query_router.py` all check a term against the **actual current distinct
values** in the relevant table (cached as a frozenset, refreshed per
process), not a maintained keyword list.

**Why**: a hardcoded list would false-trigger constantly ("Black"/"Red" are
both market classes and ordinary English words) and drift out of sync as
data is added. Checking against what's actually in the database only fires
when a term genuinely resolves to something real, and stays correct
automatically.

**Affects**: `query_router.py`'s `_match_against_cache()` and every caller of
it.

**Don't**: add a new entity-detection rule as a hardcoded list when a
`get_available_*()`-style query against the real data is available instead
— this is a repeated, deliberate pattern in this codebase, not a one-off.

---

## `deploy/app.py`'s safety checks are verbatim copies, not a shared module (Change 16)

**Decision**: `deploy/app.py` got its own copies of `_call_gemini()`,
`_verify_faithfulness()`, `_verify_gene_mentions()`, and the gap-check
functions — logic copied from `hf_space/app.py`, not extracted into a
module both import.

**Why**: `hf_space/app.py`'s versions are live, tested, and verified against
real production traffic. Refactoring that working code into a shared module,
purely for DRY, when the actual task was "sync `deploy/app.py`," would be
real, unrequested risk to the live Space for a maintenance-only win. This
matches an existing (if imperfect) precedent in the codebase — `NO_MATCH_THRESHOLD`
and `_out_of_scope()` are already duplicated the same way between the two
apps.

**Affects**: any future change to faithfulness checking, gene validation, or
the gap-check loop must be made in **both** `hf_space/app.py` and
`deploy/app.py` — there is no single source of truth for this logic.

**Don't**: assume fixing a bug in one app's copy fixed it in the other's.
Check both.

---

## `deploy/app.py` uses a dedicated Gemini key, not its existing Qwen model, for safety checks

**Decision**: rather than reuse `deploy/app.py`'s one existing provider
(Qwen via HF InferenceClient) for the faithfulness check and gap-check loop,
Change 16 added Gemini as a new, optional dependency — soft-failing to
no-op if unconfigured.

**Why**: the entire reason these checks are "cheap" elsewhere in the system
is that they run against Gemini directly, bypassing the primary generation
waterfall, so a secondary check never competes with primary generation for
scarce quota. `deploy/app.py` has exactly one provider total; reusing it for
checks would double load on the only quota this app has — the opposite of
"cheap."

**Affects**: `deploy/app.py`'s optional `Googlegeminiapi`/`GEMINI_API_KEY`
secret. If unset, the app behaves identically to its pre-Change-16 self.

**Don't**: "simplify" this by routing the check through Qwen because "it's
already there" — that reasoning was considered and rejected explicitly.

---

## Separate HF dataset repo for the vector store, not the Space's own repo

**Decision**: `vector_db/` (and everything that joined it later — BM25
index, gene index, trial-data DB, GraphRAG artifacts, paper-metadata DB)
lives in a separate HF **dataset** repo (`Parimalanath/bean-lab-vector-db`),
pulled at Space-boot time via `snapshot_download()` — not committed into the
Space's own git repo.

**Why**: the Space's own repo triggers a fresh Git LFS pull on every
restart, as part of the Docker build, blocking the health check. A ~1GB
vector store made builds take 20+ minutes and sometimes time out entirely,
leaving the Space stuck in "Building." `snapshot_download()` at app-startup
(not build time) downloads once, caches, and doesn't block the build.

**Affects**: every new large data artifact this project produces should go
into this same dataset repo, not into `hf_space/`'s own git history.

**Don't**: commit a new multi-MB data file into the Space repo directly.

---

## GraphRAG community embeddings are precomputed offline, never at Space boot

**Decision**: `community_embeddings.npy` is computed once, offline, on HPCC
with a GPU (`scripts/embed_community_reports.py`), and simply loaded at
Space startup — never computed by calling `.encode()` at runtime.

**Why**: the original implementation called `.encode()` over ~5,130 community
report texts synchronously at Space startup, on HF's CPU-only free tier,
using a 335M-parameter model. This hung the health check for 30+ minutes and
killed the Space with a launch timeout. Precomputing offline turns a
startup-blocking multi-minute (or longer) operation into a file load.

**Affects**: any future artifact that needs embedding at scale must be
precomputed offline and shipped through the dataset repo, never computed
inline during Space startup.

**Don't**: add a new `.encode()` call to any code path that runs at Space
import time.

---

## No LLM classifier for query routing

**Decision**: `query_router.route()` is rule-based (regex + cache lookups),
with **no LLM call**, and no LLM-based "ambiguous case" fallback either
(considered, not implemented — "ambiguous" was never given a crisp trigger
condition).

**Why**: an LLM classifier on every query costs real quota on the 80%+ of
questions that are plain literature lookups, for a problem regex/cache
matching already handles well.

**Affects**: `query_router.py`. The dataclass leaves room to add an LLM
fallback later without changing callers, but none exists today.

**Don't**: assume routing decisions involve any model call — they're 100%
deterministic given the same database contents.

---

## Gene validation is three-way, not two-way (verified/not-found)

**Decision**: gene/locus mentions are labeled one of three ways —
NCBI/UniProt-verified, "known classical genetics symbol" (a small,
hand-curated allowlist), or genuinely "not found" — rather than a simple
verified/unverified binary.

**Why**: verified directly against the live NCBI and UniProt APIs before
writing any code: classical bean genetics symbols (Co-1, Phg-1, bc-3, etc.)
are essentially **absent** from both databases — they're populated by
genome-annotation pipelines, not curated with decades-old breeding-literature
nomenclature. Flagging every mention of a real, well-established gene as
"not found — treat with caution" would be actively misleading; it implies
the system is untrustworthy when the two *databases* just don't cover this
vocabulary.

**Affects**: `gene_validator.py`'s `CLASSICAL_GENE_SYMBOLS` allowlist and its
three-way footer format.

**Don't**: collapse this back to a two-way label, or assume an NCBI/UniProt
miss on a classical symbol means the gene name is suspect.

---

## Extraction tradeoffs point in opposite directions on purpose

**Decision**: gene-mention extraction (`gene_validator.extract_gene_mentions()`)
favors **precision over recall** — a missed mention costs nothing, a false
one is misleading. Structured trial-data candidate detection
(`is_extraction_candidate()`) favors **recall over precision** — a missed
candidate silently loses real data, while an over-included one just costs
one wasted LLM call that returns an empty array.

**Why**: the cost of a false positive vs. a false negative is different in
each case, so the same "favor precision" instinct shouldn't be applied
uniformly.

**Affects**: any future extraction/detection logic added to either module —
match the existing bias for that module, don't default to "favor precision"
everywhere out of habit.

**Related edge cases worth knowing about, not obvious from the code alone**:
- Gene extraction excludes bare `"Co-N"` matches when the word `"cobalt"`
  appears within ~20 characters — `Co-1` collides with the chemistry/
  nutrient sense of "cobalt-1" in this corpus.
- `Fin`/`fin`/`Ppd` (single-word gene symbols) are only counted as gene
  mentions when a biology keyword (gene, allele, locus, QTL, resistance,
  etc.) appears within 60 characters — otherwise far too ambiguous in
  free-flowing prose. Bare `"P"` and bare `"I"` are never matched at all,
  even with context, since single-letter symbols are unworkably ambiguous.
- Cultivar/location/trait/species/methodology matching in `query_router.py`
  is a word-boundary **regex** search per cached value, not a per-token hash
  lookup — deliberately, because real cultivar names can be multi-word
  ("Black Turtle Soup") or hyphenated, and trait labels use underscores
  ("days_to_maturity") that never appear in natural phrasing ("days to
  maturity"). A hash lookup would silently miss both.

---

## Ground-truth answers only where independently verified (11 of 75)

**Decision**: RAGAS `ground_truth` fields (needed for `context_recall`) are
populated only for questions where a real supporting passage was actually
found by searching `data/processed_chunks.json` — 11 of 75 as of Change 11
— and left `None` everywhere else, rather than authored from general domain
knowledge.

**Why**: fabricating reference answers without verified corpus access would
produce evaluation numbers that look rigorous but rest on guesses. A crop-
science-journal corpus genuinely doesn't cover some plausible-sounding
agronomy-extension-style questions (soil pH ranges, herbicide product
names) — leaving those `None` is honest, not incomplete.

**Affects**: `src/benchmark_questions.py`, `src/eval_ragas.py`'s
`context_recall`/`context_precision` metrics (meaningful only for the 11).

**Don't**: fill in the remaining 64 `ground_truth` fields without first
verifying real supporting text exists in the actual corpus.
