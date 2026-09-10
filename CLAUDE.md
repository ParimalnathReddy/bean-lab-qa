# CLAUDE.md

Instructions for Claude Code working in this repository. Project knowledge
(architecture, current state, configuration, past decisions, history) lives
in `docs/` — read `docs/current-state.md` first, every session, before
touching code. This file is Claude-specific: how to work here, not what the
project is.

`AGENTS.md` is the equivalent file for Codex — it points at the same
`docs/`. Keep project knowledge in `docs/` only; never duplicate it into
this file or into `AGENTS.md`.

## Start every session

1. Read `docs/current-state.md` in full.
2. If the task touches a specific subsystem, also read the relevant section
   of `docs/architecture.md` and check `docs/decisions.md` for any decision
   that constrains it.
3. Don't trust a claim in `DOCUMENTATION_INDEX.txt` or any root-level
   `README*.md`/`*.md` file at face value — several are explicitly marked
   historical or have been found stale before (see `docs/current-state.md`'s
   "Known problems"). `docs/` is the current source of truth; if something
   there conflicts with an older doc, `docs/` wins and the older doc should
   be corrected, not trusted.

## Core working discipline (established repeatedly across this project's
## history — follow it, don't rederive it from scratch each time)

- **Verify claims against the actual code/data before writing anything
  down or acting on an assumption.** This project's own history is full of
  cases where a plausible-sounding claim ("this file is an identical copy,"
  "this feature was ported automatically," "the vector store is up to
  date") turned out to be false when actually checked — `diff`ing files,
  grepping imports, querying a live database, or reading raw file listings
  directly instead of trusting a comment or a prior doc.
- **Trace the real code path, don't reason about it in the abstract**, when
  the question is "does X actually happen" — e.g. reproduce the exact
  request/string an LLM call would receive, or run the real function against
  real data, rather than arguing from what the code "should" do.
- **State decisions in writing before implementing them**, for any genuine
  design fork — a comment directly above the relevant function, written
  before the code, not reconstructed afterward from what got built. This
  project's existing `docs/decisions.md` entries were extracted from exactly
  this kind of comment; add new ones the same way, and add the corresponding
  entry to `docs/decisions.md` when you do.
- **When a fork can't be resolved by evidence alone** (a genuine tradeoff,
  not a factual question), ask the user rather than picking silently.
- **Report gaps honestly.** If a check wasn't run, if a value is a rule-based
  best-effort rather than authoritative, if something is built but not
  wired in or not deployed — say so plainly in whatever you write, the same
  way this project's own docs mark honesty boundaries (e.g. `trial_data.db`'s
  LLM-extraction caveat, `paper_metadata.db`'s species/methodology tags).

## Testing conventions

- **Copy the real function body verbatim into a test**, not a reimplemented
  stand-in that merely looks similar — this project has been burned by test
  suites drifting from what's actually shipped. Where it matters, confirm
  the copy is verbatim (e.g. via Python's `ast` module comparison), not just
  eyeballed.
- **Test against real data where feasible** (the real vector store, a real
  local database), not only synthetic fixtures — synthetic-only tests have
  missed real edge cases here before (e.g. a `None`-distance chunk that a
  7-case synthetic suite never happened to construct).
- **When the user reviews your test coverage and finds a gap, close it with
  direct evidence** (a new test against the real function/data), not just a
  restated argument for why the original reasoning was probably fine.
- For a live-deployed change, verify against the actual deployed system when
  feasible — e.g. `gradio_client.submit()` against the real HF Space,
  polling for intermediate status output, not just a local mock of `chat()`.

## Two live git repos — know which one you're pushing to

- The main GitHub repo (`ParimalnathReddy/bean-lab-qa`) contains `src/`,
  `hf_space/`, `deploy/`, docs, etc.
- `hf_space/` is **also** its own separate git repository with its own
  remote, pointed at the live Hugging Face Space
  (`huggingface.co/spaces/Parimalanath/bean-lab-qa`). Committing/pushing in
  the main repo does **not** deploy anything — only `git -C hf_space push`
  (or working directly inside a `hf_space/` checkout of that remote) does.
- After editing `hf_space/query_router.py` or `hf_space/prompts.py`,
  `diff` against the `src/` counterpart before considering the change done
  — they're expected to stay byte-identical. `hf_space/retriever.py` is a
  known, accepted exception (already diverged from `src/retriever.py`).
- After pushing to the `hf_space/` remote, poll
  `https://huggingface.co/api/spaces/Parimalanath/bean-lab-qa/runtime` for
  `stage` reaching `RUNNING` (not `BUILD_ERROR`/`RUNTIME_ERROR`) before
  considering a deploy verified — background the poll rather than blocking
  on it if other work can proceed meanwhile.

## Security

- **Never let a caught exception's raw text reach a user-facing message.**
  `requests.HTTPError.__str__()` includes the full request URL — for
  Gemini's REST API, that URL embeds the API key as a `?key=...` query
  param. Log full detail server-side (`print(...)`, which reaches HF Space
  logs) and raise/return a sanitized, generic message instead. This was a
  real, live incident (see `docs/CHANGELOG.md`, 2026-08-14) — not a
  hypothetical to be relaxed later.
- Never commit a real credential (API key, HF token) into any file, script,
  or job. If one is ever found already committed, removing it from the file
  is necessary but not sufficient — flag it for rotation explicitly, since
  removal alone doesn't invalidate an already-exposed credential.
- Any file upload, live-Space push, or credential rotation is an
  outward-facing/hard-to-reverse action — confirm with the user before doing
  it unless they've already clearly authorized exactly that action.

## Updating `docs/`

- Update `docs/current-state.md`, `docs/architecture.md`,
  `docs/configuration.md`, or `docs/decisions.md` in the **same turn** as a
  code change that makes any of them stale — not as a separate cleanup pass
  later. A doc that's already wrong by the time it's committed is worse than
  no doc.
- Add a new entry to `docs/CHANGELOG.md` for any meaningful change (a new
  Change-numbered feature, a real incident and its fix) — not for every
  small edit. Follow the existing entry format and numbering convention.
- If you discover an existing doc (in `docs/` or elsewhere) is stale or
  contradicted by the actual code, fix it as part of the same piece of work
  that found the discrepancy — don't leave it for later, and don't silently
  work around it without correcting the record.
