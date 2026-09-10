#!/usr/bin/env python3
"""
Bean Lab Research QA — Gradio Web App (v2)
Deployed on HuggingFace Spaces.

Improvements over v1:
  - Scientific synonym expansion before retrieval
  - Cross-encoder reranking (retrieve 20, rerank to top 10)
  - Evidence-first chain-of-thought prompting
  - Explicit SUPPORTED / PARTIALLY_SUPPORTED / INFERRED / UNSUPPORTED labels
  - Soft failure: always attempts best-effort answer

Change 16: brought this file's answer-quality safety checks up to parity
with hf_space/app.py's — gene-validation footer (Change 5), faithfulness
check (Change 3), and the bounded gap-check loop (Change 15). Deliberately
did NOT bring GraphRAG global mode (Change 9) or conversation memory
(Change 8): this file's UI is a single-shot Q&A form with no chat/session
concept to hang conversation memory on, and GraphRAG needs three data
artifacts (community_reports.json, community_embeddings.npy,
paper_graph.json) this app doesn't download and a graph-rendering panel its
UI doesn't have — both are real, separate future work, not a quick port,
unlike the three safety checks above. See DOCUMENTATION_INDEX.txt's
Change 16 section for the full investigation (what was actually inherited
via the shared src/ imports vs. what needed porting — narrower than this
file's own repo-tree doc comment used to claim) and why this ported
faithfulness/gap-check via a NEW Gemini dependency rather than reusing the
single Qwen model already configured here (see GEMINI_API_KEY below).
"""

import json
import os
import re
import sys
import time
import gradio as gr
import chromadb
import requests
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from typing import Optional

# Add src/ to path so shared modules are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from retriever import BeanRetriever
from prompts import build_context, build_messages, format_references

try:
    from gene_validator import extract_gene_mentions, validate_genes
except Exception as _gene_import_error:
    extract_gene_mentions = None
    validate_genes = None
    print(f"WARNING: gene_validator not available: {_gene_import_error}")

# ── Config ────────────────────────────────────────────────────────────────────

COLLECTION_NAME = "bean_research_docs"
EMBED_MODEL     = "BAAI/bge-large-en-v1.5"
LLM_MODEL       = "Qwen/Qwen2.5-7B-Instruct"
DB_PATH         = "vector_db"
TOP_K           = 10
N_CANDIDATES    = 20

# Change 16: a NEW dependency, deliberately, not a reuse of LLM_MODEL/Qwen
# above. hf_space/app.py's faithfulness check and gap-check loop call Gemini
# DIRECTLY rather than through its 4-tier waterfall specifically so a cheap
# secondary check never competes with the primary answer for the SAME
# scarce quota — Gemini's free tier is large and completely separate from
# whatever pool generation already uses. Reusing Qwen/InferenceClient here
# instead would mean every answer costs 2x the calls against the ONE quota
# this app has, undermining exactly the reasoning these checks were
# designed around. Soft-fails cleanly if unset (see _verify_faithfulness()
# and _check_context_sufficiency() below) — this app works exactly as it
# did before Change 16 if GEMINI_API_KEY is never configured, it just
# doesn't get the two checks that depend on it.
GEMINI_MODEL    = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()
GEMINI_API_KEY  = os.environ.get("Googlegeminiapi", os.environ.get("GEMINI_API_KEY", "")).strip()


def _sanitize_provider_error(error: Exception) -> str:
    """Return provider error text with URL query secrets redacted."""
    text = str(error)
    text = re.sub(r"([?&]key=)[^\s|)]+", r"\1<redacted>", text)
    text = re.sub(r"(Authorization: Bearer\s+)[^\s|)]+", r"\1<redacted>", text, flags=re.I)
    return text


# ── Load resources (cached at startup) ───────────────────────────────────────

_startup_error: Optional[str] = None

try:
    print("Loading embedding model...")
    embedder = SentenceTransformer(EMBED_MODEL)
    print("✓ Embedding model loaded")
except Exception as _e:
    embedder = None
    _startup_error = f"Failed to load embedding model: {_e}"
    print(f"ERROR: {_startup_error}")

try:
    print("Loading ChromaDB...")
    chroma_client = chromadb.PersistentClient(
        path=DB_PATH,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = chroma_client.get_collection(COLLECTION_NAME)
    print(f"✓ ChromaDB loaded: {collection.count()} chunks")
except Exception as _e:
    collection = None
    _startup_error = _startup_error or f"Failed to load ChromaDB: {_e}"
    print(f"ERROR: {_startup_error}")

retriever = None
if collection is not None and embedder is not None:
    retriever = BeanRetriever(collection, embedder=embedder)
    retriever._load_cross_encoder()

HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    print("WARNING: HF_TOKEN not set — LLM calls will fail.")
llm_client = InferenceClient(token=HF_TOKEN if HF_TOKEN else None)
print("✓ HF Inference client ready")


# ── LLM call ─────────────────────────────────────────────────────────────────

def call_llm(messages: list) -> str:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN not configured. Add it in Space Settings → Secrets.")
    try:
        response = llm_client.chat_completion(
            messages=messages,
            model=LLM_MODEL,
            max_tokens=1024,
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        err = str(e)
        if "401" in err or "unauthorized" in err.lower():
            raise RuntimeError("HuggingFace authentication failed. Check HF_TOKEN.") from e
        if "429" in err or "rate limit" in err.lower():
            raise RuntimeError("Rate limit reached. Please wait and try again.") from e
        if "503" in err or "unavailable" in err.lower():
            raise RuntimeError(f"LLM service temporarily unavailable. Try again shortly.") from e
        raise RuntimeError(f"LLM error: {e}") from e


# ── Gemini call (Change 16) ────────────────────────────────────────────────────
# Deliberately NOT call_llm() above — see GEMINI_API_KEY's config comment for
# why this is a separate provider, not a second call through Qwen/HF
# InferenceClient. Same REST pattern as hf_space/app.py's _call_gemini().

def _call_gemini(messages: list) -> str:
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    contents = [
        {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
        for m in messages if m["role"] != "system"
    ]
    payload = {
        "contents": contents,
        "generationConfig": {"maxOutputTokens": 1500, "temperature": 0.1},
    }
    if system_parts:
        payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}",
        json=payload,
        timeout=60)
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


# ── Faithfulness verification (Change 3, ported in Change 16) ─────────────────
# Second LLM call, Gemini specifically — a cheap secondary check, not worth
# spending this app's one HF Inference quota (Qwen) on. Soft-fails: never
# blocks the user from seeing their answer, only ever adds a footer.

_FAITHFULNESS_PROMPT = """Given these source passages:
{context}

And this generated answer:
{answer}

For each factual claim in the answer, determine if it is:
- SUPPORTED: directly stated in the sources
- UNSUPPORTED: not found in any source
- CONTRADICTED: contradicts information in the sources

Return ONLY a JSON object, no other text, no markdown code fences:
{{"claims": [{{"text": "...", "verdict": "SUPPORTED|UNSUPPORTED|CONTRADICTED"}}]}}"""


def _verify_faithfulness(context: str, answer: str) -> str:
    """
    Ask Gemini whether each claim in `answer` is actually supported by
    `context`. Returns a markdown warning footer listing any UNSUPPORTED or
    CONTRADICTED claims, or "" if everything checked out clean OR the check
    itself couldn't run/parse (soft failure — never raises). Verbatim logic
    from hf_space/app.py's _verify_faithfulness() — see that function's own
    docstring for the full reasoning (why claims are never stripped from
    the answer text, only flagged).
    """
    if not GEMINI_API_KEY or not context:
        return ""

    prompt = _FAITHFULNESS_PROMPT.format(context=context, answer=answer)

    try:
        raw = _call_gemini([{"role": "user", "content": prompt}])
    except Exception as e:
        print(f"Faithfulness check skipped (Gemini call failed): {_sanitize_provider_error(e)}")
        return ""

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return ""

    try:
        claims = json.loads(match.group()).get("claims", [])
    except Exception as e:
        print(f"Faithfulness check skipped (JSON parse failed): {e}")
        return ""

    flagged = [
        c for c in claims
        if isinstance(c, dict) and c.get("verdict") in ("UNSUPPORTED", "CONTRADICTED") and c.get("text")
    ]
    if not flagged:
        return ""

    lines = [f"- *{c['verdict'].title()}:* {c['text'].strip()}" for c in flagged]
    return (
        "\n\n---\n⚠️ **Automated faithfulness check** — the following claim(s) "
        "could not be verified against the retrieved sources:\n" + "\n".join(lines)
    )


# ── Gene/locus mention verification (Change 5, ported in Change 16) ───────────
# Free check (regex + dict lookup, no LLM call) — runs unconditionally,
# same reasoning as hf_space/app.py: a query with no genetics vocabulary can
# still produce an answer that names a gene, and catching a hallucinated one
# there matters just as much. No query-router flag to gate this on here
# (query_router.py was never wired into this file — see Change 16's doc
# section for why that stayed out of scope), so it simply always runs.

def _verify_gene_mentions(answer: str) -> str:
    if extract_gene_mentions is None:
        return ""
    try:
        mentions = extract_gene_mentions(answer)
        if not mentions:
            return ""
        checked = validate_genes(mentions)
    except Exception as e:
        print(f"Gene verification skipped ({e})")
        return ""

    lines = []
    for c in checked:
        if c["verified"] and c["source"] == "classical_literature":
            lines.append(
                f"- **{c['mention']}** — known classical genetics symbol "
                f"({c['description']}); not independently cross-referenced in NCBI/UniProt"
            )
        elif c["verified"]:
            desc = f": {c['description']}" if c.get("description") else ""
            lines.append(f"- **{c['mention']}** — verified ({c['source'].upper()} {c['gene_id']}{desc})")
        else:
            lines.append(
                f"- **{c['mention']}** — not found in NCBI Gene/UniProt for *P. vulgaris*; "
                f"treat with caution"
            )
    return "\n\n---\n🧬 **Gene/locus mentions:**\n" + "\n".join(lines)


# ── Bounded gap-check loop (Change 15, ported in Change 16) ───────────────────
# Same design as hf_space/app.py's — see that file's CHANGE 15 spec comment
# above chat() for the full rationale. One structural simplification here:
# there's no Change 13-style escalation to layer underneath, since GraphRAG
# was deliberately not ported (see this file's module docstring) — this
# loop simply tries to improve (chunks, confidence) before the existing
# _out_of_scope() gate reads them, nothing more. One real limitation worth
# stating plainly: answer_question() below is a PLAIN function, not a
# generator — unlike hf_space/app.py's chat(), there's no streaming status
# message this app can show while the loop runs (no _STEP_REFINING
# equivalent is possible here). The added latency (up to 2 extra Gemini
# round-trips and 2 extra retrieval calls) is real but invisible to the
# user until the final answer appears — an accepted limitation of this
# app's simpler, non-streaming UI, not something Change 16 attempts to fix.

_SUFFICIENCY_PROMPT = """A user asked this question:
{question}

The following passages were retrieved using the search query "{search_query}":
{context}

Determine whether these passages contain enough specific information to
answer the question. If they don't, suggest a better search query — for
example, more specific scientific terminology, different phrasing, or terms
more likely to appear directly in the source literature.

Return ONLY a JSON object, no other text, no markdown code fences:
{{"sufficient": true|false, "reformulated_query": "..." or null}}"""


def _check_context_sufficiency(question: str, search_query: str, context: str) -> tuple:
    if not GEMINI_API_KEY or not context:
        return True, None

    prompt = _SUFFICIENCY_PROMPT.format(question=question, search_query=search_query, context=context)

    try:
        raw = _call_gemini([{"role": "user", "content": prompt}])
    except Exception as e:
        print(f"Context-sufficiency check skipped (Gemini call failed): {_sanitize_provider_error(e)}")
        return True, None

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return True, None

    try:
        data = json.loads(match.group())
    except Exception as e:
        print(f"Context-sufficiency check skipped (JSON parse failed): {e}")
        return True, None

    sufficient = bool(data.get("sufficient", True))
    reformulated = data.get("reformulated_query")
    reformulated = reformulated.strip() if isinstance(reformulated, str) and reformulated.strip() else None
    return sufficient, reformulated


def _gap_check_retrieve(
    message: str, search_query: str, chunks: list, confidence: str,
    retriever_obj, year_range: Optional[str], max_iterations: int = 2,
) -> tuple:
    """
    Returns (chunks, confidence, search_query, iterations_used) — same
    contract as hf_space/app.py's _gap_check_retrieve(). One difference:
    takes `year_range` explicitly, since this app's retrieve() call (unlike
    hf_space/app.py's) is parameterized by the UI's year-filter dropdown —
    a reformulated retrieval must keep respecting that filter, not silently
    drop it.
    """
    iterations_used = 0
    for _ in range(max_iterations):
        if confidence == "SUPPORTED":
            break
        context = build_context(chunks) if chunks else ""
        sufficient, reformulated_query = _check_context_sufficiency(message, search_query, context)
        if sufficient or not reformulated_query:
            break
        search_query = reformulated_query
        chunks, confidence = retriever_obj.retrieve(
            search_query, top_k=TOP_K, year_range=year_range, n_candidates=N_CANDIDATES,
        )
        iterations_used += 1
    return chunks, confidence, search_query, iterations_used


# ── Main QA function ──────────────────────────────────────────────────────────

# Distance above this means the corpus has nothing relevant — skip LLM entirely.
NO_MATCH_THRESHOLD = 1.10

_GREETINGS = {"hello", "hi", "hey", "howdy", "hiya", "greetings", "good morning",
              "good afternoon", "good evening", "how are you", "what's up", "whats up"}

_WELCOME_MSG = (
    "Hello! I'm the Bean Lab Research Assistant. I can answer questions about "
    "bean and legume crop science based on 1,000+ research papers (1961–2026).\n\n"
    "Try asking something like:\n"
    "• What diseases affect bean crops and how can they be managed?\n"
    "• How does drought stress affect bean yield?\n"
    "• What nitrogen fixation rates have been reported for common bean?\n"
    "• How does intercropping beans with maize affect productivity?"
)

_OUT_OF_SCOPE_MSG = (
    "I couldn't find relevant research for that question in the Bean Lab database. "
    "This system is specialized for bean and legume crop science — try asking about "
    "bean diseases, drought tolerance, nitrogen fixation, breeding, or agronomic management."
)


def _out_of_scope(chunks) -> bool:
    """
    True if there's no chunk, or the top chunk's dense distance exceeds the
    out-of-scope threshold. BM25-only top chunks (hybrid retrieval,
    distance=None) never trigger this — see the identical helper in
    hf_space/app.py for the full rationale.
    """
    if not chunks:
        return True
    dist = chunks[0].get("distance")
    return dist is not None and dist > NO_MATCH_THRESHOLD


def answer_question(question: str, year_filter: str) -> tuple:
    if _startup_error:
        return f"System error: {_startup_error}", ""
    if not question.strip():
        return "", ""
    if retriever is None:
        return "System not ready — ChromaDB or embedder failed to load.", ""

    # Greetings get a friendly welcome, not a rejection
    q_lower = question.strip().lower().rstrip("!?. ")
    if q_lower in _GREETINGS:
        return _WELCOME_MSG, ""

    year_range = None if year_filter == "All years" else year_filter

    # Retrieve: expand query → get 20 candidates → rerank → top 10
    chunks, confidence = retriever.retrieve(
        question,
        top_k=TOP_K,
        year_range=year_range,
        n_candidates=N_CANDIDATES,
    )

    # Gap-check loop (Change 15/16): try to improve (chunks, confidence)
    # BEFORE the out-of-scope gate below reads them — a refinement layer
    # underneath that gate, same layering principle as hf_space/app.py's
    # relationship to its Change 13 escalation decision. No-ops entirely
    # (0 Gemini calls) if confidence is already SUPPORTED or GEMINI_API_KEY
    # is unset — see _check_context_sufficiency()'s own guard.
    if confidence != "SUPPORTED":
        chunks, confidence, _search_query, _gap_iters = _gap_check_retrieve(
            question, question, chunks, confidence, retriever, year_range,
        )

    # If the best chunk is too distant, the question is out of scope — don't waste LLM call
    if _out_of_scope(chunks):
        return _OUT_OF_SCOPE_MSG, ""

    # Build evidence-first prompt
    messages = build_messages(question, chunks, confidence)

    try:
        answer = call_llm(messages)
    except RuntimeError as e:
        return f"⚠️ {e}", ""

    # Faithfulness check + gene-mention footer (Change 3/5, ported Change 16)
    # — both soft-fail to "" and never block the answer above from being
    # shown. Appended to the answer text, not references, matching
    # hf_space/app.py's placement of both footers on the answer side.
    faithfulness_warning = _verify_faithfulness(build_context(chunks), answer)
    gene_footer = _verify_gene_mentions(answer)
    answer = f"{answer}{faithfulness_warning}{gene_footer}"

    references = format_references(chunks)
    return answer, references


# ── Gradio UI ─────────────────────────────────────────────────────────────────

APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

:root {
    --bg: #f7f8f5;
    --surface: #ffffff;
    --surface-2: #f0f4ee;
    --line: #dbe3d8;
    --text: #18211b;
    --muted: #617064;
    --accent: #2f8f5b;
    --accent-hover: #257347;
    --focus: rgba(47, 143, 91, 0.18);
}

body, .gradio-container {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.gradio-container {
    max-width: 1040px !important;
    margin: 0 auto !important;
    padding: 28px 18px !important;
}

footer, .footer, .built-with { display: none !important; }

#app-shell {
    border: 1px solid var(--line);
    border-radius: 8px;
    overflow: hidden;
    background: var(--surface);
    box-shadow: 0 18px 50px rgba(24, 33, 27, 0.08);
}

#app-header {
    padding: 18px 22px;
    border-bottom: 1px solid var(--line);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    background: var(--surface);
}

#app-header h1 {
    margin: 0;
    font-size: 20px;
    line-height: 1.2;
    letter-spacing: 0;
}

#app-header .meta {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: flex-end;
}

#app-header .meta span {
    border: 1px solid var(--line);
    border-radius: 999px;
    color: var(--muted);
    font-size: 12px;
    padding: 5px 9px;
    background: var(--surface-2);
}

#main-panel {
    padding: 22px;
}

#question-box textarea {
    border-radius: 8px !important;
    border: 1px solid var(--line) !important;
    color: var(--text) !important;
    font-size: 15px !important;
    line-height: 1.6 !important;
    box-shadow: none !important;
}

/* #answer-box is gr.Markdown as of Change 16 (was gr.Textbox) — no
   <textarea> to select, styled directly instead so the border/font
   treatment that used to come from the shared rule above doesn't just
   silently disappear. */
#answer-box {
    border-radius: 8px !important;
    border: 1px solid var(--line) !important;
    color: var(--text) !important;
    font-size: 15px !important;
    line-height: 1.6 !important;
    padding: 12px 14px !important;
    min-height: 120px;
}

#question-box textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--focus) !important;
}

#ask-btn {
    border-radius: 8px !important;
    background: var(--accent) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    height: 44px !important;
}

#ask-btn:hover {
    background: var(--accent-hover) !important;
}

#refs-box {
    border-top: 1px solid var(--line);
    padding: 16px 22px 20px;
    background: var(--surface-2);
}

#refs-box p, #refs-box li {
    color: var(--muted);
    font-size: 14px;
}

@media (max-width: 720px) {
    .gradio-container {
        padding: 0 !important;
    }
    #app-shell {
        border-radius: 0;
        border-left: none;
        border-right: none;
    }
    #app-header {
        flex-direction: column;
        align-items: flex-start;
    }
    #app-header .meta {
        justify-content: flex-start;
    }
    #main-panel {
        padding: 16px;
    }
}
"""

with gr.Blocks(
    title="Bean Lab Research QA",
    theme=gr.themes.Base(),
    css=APP_CSS,
) as demo:

    with gr.Column(elem_id="app-shell"):
        gr.HTML("""
        <div id="app-header">
          <h1>Bean Lab Research QA</h1>
          <div class="meta">
            <span>1,067 papers</span>
            <span>1961-2026</span>
            <span>DOI citations</span>
          </div>
        </div>
        """)

        with gr.Column(elem_id="main-panel"):
            with gr.Row():
                with gr.Column(scale=3):
                    question_input = gr.Textbox(
                        label="Question",
                        placeholder="What diseases affect bean crops and how can they be managed?",
                        lines=3,
                        elem_id="question-box",
                    )
                with gr.Column(scale=1):
                    year_filter = gr.Dropdown(
                        label="Time period",
                        choices=["All years", "1961-2006", "2007-2026"],
                        value="All years",
                    )

            ask_btn = gr.Button("Ask", variant="primary", size="lg", elem_id="ask-btn")

            # gr.Markdown, not gr.Textbox (Change 16) — the faithfulness and
            # gene-mention footers now appended to `answer` use markdown
            # (**bold**, bullet lists) the same way hf_space/app.py's
            # Chatbot already renders them; a plain Textbox would show the
            # literal "**" characters instead of rendering them.
            answer_output = gr.Markdown(
                label="Answer",
                elem_id="answer-box",
            )

        with gr.Column(elem_id="refs-box"):
            references_output = gr.Markdown(label="References")

    ask_btn.click(
        fn=answer_question,
        inputs=[question_input, year_filter],
        outputs=[answer_output, references_output],
    )
    question_input.submit(
        fn=answer_question,
        inputs=[question_input, year_filter],
        outputs=[answer_output, references_output],
    )

if __name__ == "__main__":
    demo.launch()
