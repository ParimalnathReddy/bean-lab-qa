# AGENTS.md

Instructions for Codex (and any other agent reading this file) working in
this repository. Project knowledge lives in `docs/`, not here — read
`docs/current-state.md` in full before making any change, then consult:

- `docs/architecture.md` — how the system actually works, mechanism-level
- `docs/configuration.md` — env vars, secrets, defaults, dependency pins
- `docs/decisions.md` — why things are built the way they are, and what not
  to change without reconsidering the whole decision
- `docs/CHANGELOG.md` — what changed, when, and why

`CLAUDE.md` is the equivalent file for Claude Code, covering the same
underlying rules in more detail. Both files must point here, to `docs/`, for
actual project knowledge — never duplicate project knowledge into either
instruction file itself.

## Don't trust older docs at face value

`DOCUMENTATION_INDEX.txt` (repo root) is a large, detailed historical build
log — useful for deep narrative detail, but `docs/current-state.md` is the
one to trust for what's true *right now*. Several root-level `README*.md`
files are explicitly marked as historical/stale in their own text or in
`docs/current-state.md`'s "Known problems" section. If something in `docs/`
conflicts with an older file, `docs/` wins — and the older file should be
corrected as part of your change, not silently ignored.

## Working discipline

- Verify claims against the real code, real data, or a live query before
  acting on them — this project has repeatedly found "obviously true"
  assumptions (an identical file copy, a feature ported automatically, an
  up-to-date data file) to be false once actually checked.
- For a genuine design fork (not a factual question), state the decision in
  writing before implementing it, and record it in `docs/decisions.md`.
- Copy real function bodies verbatim into tests rather than reimplementing
  similar-looking logic; prefer testing against real data over synthetic
  fixtures where feasible.
- Report gaps and unverified claims honestly rather than presenting a
  best-effort result as confirmed.

## Two live git repos

The main repo (`ParimalnathReddy/bean-lab-qa` on GitHub) holds `src/`,
`hf_space/`, `deploy/`, and docs. `hf_space/` is **also** a separate git
repository with its own remote pointed at the live Hugging Face Space —
pushing in the main repo does not deploy anything. Only pushing to the
`hf_space/` remote does. Keep `hf_space/query_router.py` and
`hf_space/prompts.py` byte-identical to their `src/` counterparts after any
edit to either (verify with `diff`); `hf_space/retriever.py` is a known,
accepted exception that has already diverged from `src/retriever.py`.

## Security

Never let a caught exception's raw text (which can embed request URLs, and
therefore API keys, for some providers) reach a user-facing message — log
full detail server-side, return/raise a sanitized generic message. Never
commit a real credential to any file. See `docs/decisions.md` and
`docs/CHANGELOG.md` (2026-08-14 entry) for the concrete incident this rule
comes from.

## Keeping `docs/` current

Update the relevant `docs/` file in the same change that makes it stale —
not as a followup. Add a `docs/CHANGELOG.md` entry for any meaningful
change (new feature, real incident + fix), following the existing
"Change N" numbering convention — not for every small edit.
