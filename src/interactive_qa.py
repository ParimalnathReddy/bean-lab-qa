#!/usr/bin/env python3
"""
Interactive QA CLI for Bean Lab research documents.
Run this on a compute node with Ollama server running.

Improvements over v1:
  - Scientific synonym expansion before retrieval
  - Cross-encoder reranking (retrieve 20, rerank to top 10)
  - Evidence-first chain-of-thought prompting
  - Explicit SUPPORTED / PARTIALLY_SUPPORTED / INFERRED / UNSUPPORTED labels
  - Soft failure: always attempts answer, never hard-refuses
"""

import sys
import time
import requests
import chromadb
from chromadb.config import Settings
from typing import Optional

from retriever import BeanRetriever
from prompts import build_ollama_prompt, format_references

PROJECT_DIR = "/mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa"
VECTOR_DB   = f"{PROJECT_DIR}/vector_db"
MODEL       = "llama3.1:8b"
OLLAMA_HOST = "localhost:11434"
TOP_K       = 10
N_CANDIDATES = 20


# ── ChromaDB ──────────────────────────────────────────────────────────────────

def load_collection():
    client = chromadb.PersistentClient(
        path=VECTOR_DB,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_collection("bean_research_docs")


# ── Ollama ────────────────────────────────────────────────────────────────────

def check_server() -> bool:
    try:
        r = requests.get(f"http://{OLLAMA_HOST}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def ask_llm(prompt: str) -> str:
    r = requests.post(
        f"http://{OLLAMA_HOST}/api/generate",
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 1024},
        },
        timeout=180,
    )
    r.raise_for_status()
    return r.json()["response"].strip()


# ── CLI ───────────────────────────────────────────────────────────────────────

HELP_TEXT = """
Commands:
  <question>          Ask any question about bean research
  /year 1961-2006     Filter to papers from 1961-2006
  /year 2007-2026     Filter to papers from 2007-2026
  /year off           Remove year filter
  /sources <n>        Set number of sources to retrieve (default: 10)
  /help               Show this help
  /quit               Exit
"""


def print_separator():
    print("\n" + "=" * 70)


def run():
    print("\n" + "=" * 70)
    print("  Bean Lab Research QA System  (v2 — improved retrieval + reasoning)")
    print(f"  Model: {MODEL}  |  DB: {VECTOR_DB}")
    print("=" * 70)

    print("\nChecking Ollama server...", end=" ", flush=True)
    if not check_server():
        print("NOT RUNNING")
        print(f"\nStart Ollama first:\n  export OLLAMA_MODELS={PROJECT_DIR}/models/ollama")
        print("  ollama serve &\n  sleep 5")
        sys.exit(1)
    print("OK")

    print("Loading vector store...", end=" ", flush=True)
    collection = load_collection()
    print(f"OK ({collection.count()} chunks)")

    retriever = BeanRetriever(collection)
    print("Loading cross-encoder reranker...", flush=True)
    retriever._load_cross_encoder()  # pre-load so first query isn't slow

    print(HELP_TEXT)

    year_filter: Optional[str] = None
    top_k = TOP_K

    while True:
        try:
            query = input("Question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query:
            continue

        if query == "/quit":
            print("Bye!")
            break
        elif query == "/help":
            print(HELP_TEXT)
            continue
        elif query.startswith("/year "):
            arg = query.split(" ", 1)[1].strip()
            if arg == "off":
                year_filter = None
                print("Year filter removed.")
            elif arg in ("1961-2006", "2007-2026"):
                year_filter = arg
                print(f"Year filter: {year_filter}")
            else:
                print("Valid options: /year 1961-2006 | /year 2007-2026 | /year off")
            continue
        elif query.startswith("/sources "):
            try:
                top_k = int(query.split(" ", 1)[1])
                print(f"Sources set to: {top_k}")
            except ValueError:
                print("Usage: /sources <number>")
            continue

        # ── Retrieve ──────────────────────────────────────────────────────────
        filter_label = f" [filter: {year_filter}]" if year_filter else ""
        print(f"\nSearching{filter_label} (expand → retrieve {N_CANDIDATES} → rerank → top {top_k})...",
              flush=True)

        t0 = time.time()
        try:
            chunks, confidence = retriever.retrieve(
                query,
                top_k=top_k,
                year_range=year_filter,
                n_candidates=N_CANDIDATES,
            )
        except Exception as e:
            print(f"Retrieval error: {e}")
            continue
        retrieval_time = time.time() - t0

        # ── Generate ──────────────────────────────────────────────────────────
        prompt = build_ollama_prompt(query, chunks, confidence)

        t1 = time.time()
        try:
            response = ask_llm(prompt)
        except Exception as e:
            print(f"LLM error: {e}")
            continue
        gen_time = time.time() - t1

        # ── Display ───────────────────────────────────────────────────────────
        print_separator()
        print(f"CONFIDENCE: {confidence}")
        print(f"\nANSWER:\n{response}")
        print(f"\nSOURCES (ranked by relevance):")
        for i, c in enumerate(chunks, 1):
            rerank = f" rerank={c['rerank_score']:.2f}" if c.get('rerank_score') is not None else ""
            section = f" {c.get('section','')}" if c.get("section") else ""
            print(f"  [{i}] doi:{c['doi']}  p.{c['page']}{section}  dist={c['distance']}{rerank}")
        print(f"\nTiming: retrieval={retrieval_time:.2f}s | generation={gen_time:.2f}s")
        print_separator()


if __name__ == "__main__":
    run()
