#!/usr/bin/env python3
"""
Interactive QA CLI for Bean Lab research documents
Run this on a compute node with Ollama server running
"""

import os
import sys
import json
import time
import requests
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional

PROJECT_DIR = "/mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa"
VECTOR_DB   = f"{PROJECT_DIR}/vector_db"
MODEL       = "llama3.1:8b"
OLLAMA_HOST = "localhost:11434"


# ── DOI helper ────────────────────────────────────────────────────────────────

def filename_to_doi(filename: str) -> str:
    """Convert stored filename back to DOI.
    e.g. '10.2135_cropsci2004.1799.pdf' → '10.2135/cropsci2004.1799'
    """
    name = filename.replace(".pdf", "")
    # First underscore separates prefix (10.XXXX) from suffix
    parts = name.split("_", 1)
    if len(parts) == 2 and parts[0].startswith("10."):
        return f"{parts[0]}/{parts[1]}"
    return name  # fallback: return as-is


def doi_to_filename(doi_or_filename: str) -> str:
    """Convert DOI input back to stored filename.
    e.g. 'doi:10.2135/cropsci2004.1799' -> '10.2135_cropsci2004.1799.pdf'
    """
    raw = doi_or_filename.strip()
    if raw.lower().startswith("doi:"):
        raw = raw[4:].strip()

    if raw.endswith(".pdf"):
        return raw

    if raw.startswith("10.") and "/" in raw:
        prefix, suffix = raw.split("/", 1)
        return f"{prefix}_{suffix}.pdf"

    # fallback: return input as-is (lets advanced users pass raw metadata value)
    return raw


# ── ChromaDB ──────────────────────────────────────────────────────────────────

def load_collection():
    client = chromadb.PersistentClient(
        path=VECTOR_DB,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_collection("bean_research_docs")


# ── Ollama ────────────────────────────────────────────────────────────────────

def check_server():
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
            "options": {"temperature": 0.1, "num_predict": 512},
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["response"].strip()


# ── RAG ───────────────────────────────────────────────────────────────────────

def retrieve(
    collection,
    question: str,
    n: int = 5,
    year_range: Optional[str] = None,
    source_file: Optional[str] = None,
) -> List[Dict]:
    filters = []
    if year_range:
        filters.append({"year_range": year_range})
    if source_file:
        filters.append({"source_file": source_file})

    if not filters:
        where = None
    elif len(filters) == 1:
        where = filters[0]
    else:
        where = {"$and": filters}

    results = collection.query(
        query_texts=[question],
        n_results=n,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        source = meta.get("source_file", "unknown")
        chunks.append({
            "text": doc,
            "source": source,
            "doi": filename_to_doi(source),
            "year_range": meta.get("year_range", "unknown"),
            "page": meta.get("page_number", "?"),
            "distance": round(dist, 4),
        })
    return chunks


DISTANCE_THRESHOLD = 0.85


def build_prompt(question: str, chunks: List[Dict]) -> str:
    context = ""
    for i, c in enumerate(chunks, 1):
        relevance = "LOW RELEVANCE - do not cite" if c["distance"] > DISTANCE_THRESHOLD else f"relevance score={c['distance']}"
        context += f"\n[DOI: {c['doi']} | {c['year_range']} | Page {c['page']} | {relevance}]\n{c['text']}\n"

    return f"""You are a precise scientific research assistant that answers questions strictly from retrieved documents.

RETRIEVAL RULES:
- Sources marked "LOW RELEVANCE" — do not cite them
- Never use off-topic documents to answer a question

ANSWERING RULES:
- Check ALL retrieved sources before saying information is unavailable
- Never contradict yourself within the same response
- Always include specific numbers, units, and scales from tables when relevant
- Always compare values between entries rather than reporting single values in isolation
- Only cite sources that directly support your specific claim

FORMAT RULES:
- State the direct answer first
- Then provide supporting data
- Cite sources using their full DOI in parentheses, e.g. (doi:10.2135/cropsci2004.1799, p.3)
- Do NOT use generic labels like [Source 1] or [Source 2] — always use the actual DOI
- Flag any uncertainty clearly at the end

CONTEXT:{context}
QUESTION: {question}

ANSWER:"""


def answer(
    collection,
    question: str,
    n: int = 5,
    year_range: Optional[str] = None,
    source_file: Optional[str] = None,
):
    t0 = time.time()
    chunks = retrieve(collection, question, n=n, year_range=year_range, source_file=source_file)
    retrieval_time = time.time() - t0

    # If ALL sources are above threshold, topic is not in the database
    if all(c["distance"] > DISTANCE_THRESHOLD for c in chunks):
        response = "This topic is not in my database. Please upload the relevant documents and try again."
        return response, chunks, retrieval_time, 0.0

    prompt = build_prompt(question, chunks)

    t0 = time.time()
    response = ask_llm(prompt)
    gen_time = time.time() - t0

    return response, chunks, retrieval_time, gen_time


# ── CLI ───────────────────────────────────────────────────────────────────────

HELP_TEXT = """
Commands:
  <question>          Ask any question about bean research
  /year 1961-2006     Filter to papers from 1961-2006
  /year 2007-2026     Filter to papers from 2007-2026
  /year off           Remove year filter
  /paper <doi>        Filter to a single paper (e.g. /paper doi:10.1002/plr2.20334)
  /paper off          Remove paper filter
  /sources <n>        Set number of sources to retrieve (default: 5)
  /help               Show this help
  /quit               Exit
"""

def print_separator():
    print("\n" + "=" * 65)

def run():
    print("\n" + "=" * 65)
    print("  Bean Lab Research QA System")
    print(f"  Model: {MODEL}  |  DB: {VECTOR_DB}")
    print("=" * 65)

    # Check Ollama
    print("\nChecking Ollama server...", end=" ", flush=True)
    if not check_server():
        print("NOT RUNNING")
        print(f"\nStart Ollama first:\n  export OLLAMA_MODELS={PROJECT_DIR}/models/ollama")
        print("  ollama serve &")
        print("  sleep 5")
        sys.exit(1)
    print("OK")

    # Load DB
    print("Loading vector store...", end=" ", flush=True)
    collection = load_collection()
    print(f"OK ({collection.count()} chunks)")

    print(HELP_TEXT)

    year_filter = None
    source_filter = None
    n_sources = 5

    while True:
        try:
            query = input("Question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query:
            continue

        # Commands
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
                print(f"Year filter set to: {year_filter}")
            else:
                print("Valid options: /year 1961-2006  |  /year 2007-2026  |  /year off")
            continue
        elif query.startswith("/sources "):
            try:
                n_sources = int(query.split(" ", 1)[1])
                print(f"Sources per query set to: {n_sources}")
            except ValueError:
                print("Usage: /sources <number>  e.g. /sources 3")
            continue
        elif query.startswith("/paper "):
            arg = query.split(" ", 1)[1].strip()
            if arg.lower() == "off":
                source_filter = None
                print("Paper filter removed.")
            else:
                source_filter = doi_to_filename(arg)
                print(f"Paper filter set to: doi:{filename_to_doi(source_filter)}")
            continue

        # Answer
        labels = []
        if year_filter:
            labels.append(f"year={year_filter}")
        if source_filter:
            labels.append(f"paper=doi:{filename_to_doi(source_filter)}")
        filter_label = f" [filter: {', '.join(labels)}]" if labels else ""
        print(f"\nSearching{filter_label}...", flush=True)

        try:
            response, chunks, retrieval_t, gen_t = answer(
                collection, query, n=n_sources, year_range=year_filter, source_file=source_filter
            )
        except Exception as e:
            print(f"Error: {e}")
            continue

        print_separator()
        print(f"ANSWER:\n{response}")
        print(f"\nRETRIEVED SOURCES:")
        for i, c in enumerate(chunks, 1):
            flag = " [LOW RELEVANCE]" if c["distance"] > DISTANCE_THRESHOLD else ""
            print(f"  [{i}] doi:{c['doi']}  (page {c['page']}, {c['year_range']})  dist={c['distance']}{flag}")
        print(f"\nTiming: retrieval={retrieval_t:.2f}s | generation={gen_t:.2f}s")
        print_separator()


if __name__ == "__main__":
    run()
