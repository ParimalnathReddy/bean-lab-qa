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
"""

import os
import sys
import time
import gradio as gr
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from typing import Optional

# Add src/ to path so shared modules are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from retriever import BeanRetriever
from prompts import build_messages, format_references

# ── Config ────────────────────────────────────────────────────────────────────

COLLECTION_NAME = "bean_research_docs"
EMBED_MODEL     = "BAAI/bge-large-en-v1.5"
LLM_MODEL       = "Qwen/Qwen2.5-7B-Instruct"
DB_PATH         = "vector_db"
TOP_K           = 10
N_CANDIDATES    = 20


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

    # If the best chunk is too distant, the question is out of scope — don't waste LLM call
    if _out_of_scope(chunks):
        return _OUT_OF_SCOPE_MSG, ""

    # Build evidence-first prompt
    messages = build_messages(question, chunks, confidence)

    try:
        answer = call_llm(messages)
    except RuntimeError as e:
        return f"⚠️ {e}", ""

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

#question-box textarea,
#answer-box textarea {
    border-radius: 8px !important;
    border: 1px solid var(--line) !important;
    color: var(--text) !important;
    font-size: 15px !important;
    line-height: 1.6 !important;
    box-shadow: none !important;
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

            answer_output = gr.Textbox(
                label="Answer",
                lines=14,
                interactive=False,
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
