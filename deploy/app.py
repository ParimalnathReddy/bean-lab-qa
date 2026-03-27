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
EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
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
# With L2-normalized embeddings: dist=1.1 ≈ cosine similarity 0.40 (very weak).
NO_MATCH_THRESHOLD = 1.10

_OUT_OF_SCOPE_MSG = (
    "This system only answers questions about bean and legume crop science.\n\n"
    "No relevant research was found for your input. Try asking about:\n"
    "• Bean diseases and management (rust, white mold, BCMV, bacterial blight)\n"
    "• Drought or heat stress effects on bean yield\n"
    "• Nitrogen fixation rates in common bean\n"
    "• Breeding for disease resistance or yield improvement\n"
    "• Intercropping, soil nutrition, or agronomic management"
)


def answer_question(question: str, year_filter: str) -> tuple:
    if _startup_error:
        return f"System error: {_startup_error}", ""
    if not question.strip():
        return "", ""
    if retriever is None:
        return "System not ready — ChromaDB or embedder failed to load.", ""

    year_range = None if year_filter == "All years" else year_filter

    # Retrieve: expand query → get 20 candidates → rerank → top 10
    chunks, confidence = retriever.retrieve(
        question,
        top_k=TOP_K,
        year_range=year_range,
        n_candidates=N_CANDIDATES,
    )

    # If the best chunk is too distant, the question is out of scope — don't waste LLM call
    if not chunks or chunks[0].get("distance", 99) > NO_MATCH_THRESHOLD:
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

with gr.Blocks(
    title="Bean Lab Research QA",
    theme=gr.themes.Soft(),
    css=".answer-box textarea { font-size: 15px !important; line-height: 1.6 !important; }"
) as demo:

    gr.Markdown("""
    # 🌱 Bean Lab Research QA System
    **Q&A over 1,067 agricultural research papers (1961–2026)**

    Ask any question about bean and legume crop research.
    Answers are grounded in scientific papers with explicit confidence labels.
    """)

    with gr.Row():
        with gr.Column(scale=3):
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="e.g. What are the main diseases affecting bean crops and how can they be managed?",
                lines=3,
            )
        with gr.Column(scale=1):
            year_filter = gr.Dropdown(
                label="Time Period",
                choices=["All years", "1961-2006", "2007-2026"],
                value="All years",
            )

    ask_btn = gr.Button("Ask", variant="primary", size="lg")

    answer_output = gr.Textbox(
        label="Answer",
        lines=14,
        interactive=False,
        elem_classes=["answer-box"],
    )

    references_output = gr.Markdown(label="References")

    gr.Markdown("""
    ---
    **Database:** 1,067 bean/legume research papers (1961–2026) | Bean Lab, MSU |
    [GitHub](https://github.com/ParimalnathReddy/bean-lab-qa)
    """)

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
