#!/usr/bin/env python3
"""
QA System using ChromaDB vector store + Ollama LLM
Retrieval-Augmented Generation (RAG) for Bean Lab research documents
"""

import os
import json
import logging
import argparse
import time
import requests
from pathlib import Path
from typing import List, Dict, Optional

import chromadb
from chromadb.config import Settings


# ── Logging ──────────────────────────────────────────────────────────────────

def setup_logging(log_file: str) -> logging.Logger:
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


# ── DOI helper ────────────────────────────────────────────────────────────────

def filename_to_doi(filename: str) -> str:
    """Convert stored filename back to DOI.
    e.g. '10.2135_cropsci2004.1799.pdf' → '10.2135/cropsci2004.1799'
    """
    name = filename.replace(".pdf", "")
    parts = name.split("_", 1)
    if len(parts) == 2 and parts[0].startswith("10."):
        return f"{parts[0]}/{parts[1]}"
    return name


# ── Vector Store ──────────────────────────────────────────────────────────────

def load_vector_store(db_path: str, collection_name: str = "bean_research_docs"):
    """Load ChromaDB collection."""
    client = chromadb.PersistentClient(
        path=db_path,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(name=collection_name)
    return collection


def retrieve_context(collection, query: str, n_results: int = 5, year_range: Optional[str] = None) -> List[Dict]:
    """
    Retrieve top-k relevant chunks from ChromaDB using the query text.
    Optionally filter by year_range ('1961-2006' or '2007-2026').
    """
    where = {"year_range": year_range} if year_range else None

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
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
            "page": meta.get("page_number", "unknown"),
            "distance": round(dist, 4),
        })
    return chunks


# ── Ollama ────────────────────────────────────────────────────────────────────

def check_ollama_server(host: str = "localhost:11434") -> bool:
    """Check if Ollama server is running."""
    try:
        response = requests.get(f"http://{host}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def query_ollama(prompt: str, model: str = "llama3.1:8b", host: str = "localhost:11434") -> str:
    """Send prompt to Ollama and return response."""
    url = f"http://{host}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,   # low temp for factual answers
            "num_predict": 512,   # max tokens in response
        },
    }
    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["response"].strip()


# ── RAG Pipeline ──────────────────────────────────────────────────────────────

DISTANCE_THRESHOLD = 0.85


def build_prompt(question: str, context_chunks: List[Dict]) -> str:
    """Build RAG prompt from question + retrieved context."""
    context_text = ""
    for i, chunk in enumerate(context_chunks, 1):
        relevance = "LOW RELEVANCE - do not cite" if chunk["distance"] > DISTANCE_THRESHOLD else f"relevance score={chunk['distance']}"
        context_text += f"\n[DOI: {chunk['doi']} | {chunk['year_range']} | Page {chunk['page']} | {relevance}]\n"
        context_text += chunk["text"] + "\n"

    prompt = f"""You are a precise scientific research assistant that answers questions strictly from retrieved documents.

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

CONTEXT:
{context_text}

QUESTION: {question}

ANSWER:"""
    return prompt


def answer_question(
    question: str,
    collection,
    model: str,
    host: str,
    n_results: int = 5,
    year_range: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """Full RAG pipeline: retrieve → build prompt → generate answer."""
    if logger:
        logger.info(f"Question: {question}")

    # Retrieve context
    start = time.time()
    chunks = retrieve_context(collection, question, n_results=n_results, year_range=year_range)
    retrieval_time = time.time() - start

    if logger:
        logger.info(f"Retrieved {len(chunks)} chunks in {retrieval_time:.2f}s")
        for i, c in enumerate(chunks, 1):
            logger.info(f"  [{i}] {c['source']} (distance={c['distance']})")

    # If all sources are above threshold, topic is not in database
    if all(c["distance"] > DISTANCE_THRESHOLD for c in chunks):
        return {
            "question": question,
            "answer": "This topic is not in my database. Please upload the relevant documents and try again.",
            "sources": chunks,
            "retrieval_time_s": round(retrieval_time, 3),
            "generation_time_s": 0.0,
            "model": model,
            "year_filter": year_range,
        }

    # Build prompt
    prompt = build_prompt(question, chunks)

    # Generate answer
    start = time.time()
    answer = query_ollama(prompt, model=model, host=host)
    generation_time = time.time() - start

    if logger:
        logger.info(f"Generated answer in {generation_time:.2f}s")

    return {
        "question": question,
        "answer": answer,
        "sources": chunks,
        "retrieval_time_s": round(retrieval_time, 3),
        "generation_time_s": round(generation_time, 3),
        "model": model,
        "year_filter": year_range,
    }


# ── Default Questions ─────────────────────────────────────────────────────────

DEFAULT_QUESTIONS = [
    "What are the main diseases affecting bean crops and how can they be managed?",
    "What nitrogen fixation rates have been reported for common bean varieties?",
    "How does drought stress affect bean yield and what tolerance mechanisms exist?",
    "What are the most effective herbicides used in bean cultivation?",
    "How has bean breeding improved resistance to bean common mosaic virus?",
    "What soil pH and nutrient conditions are optimal for bean production?",
    "What are the yield differences between determinate and indeterminate bean varieties?",
    "How does intercropping beans with maize affect productivity?",
]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAG QA system for Bean Lab research documents")
    parser.add_argument("--vector-db", required=True, help="Path to ChromaDB directory")
    parser.add_argument("--chunks-file", help="Path to processed_chunks.json (unused, kept for compatibility)")
    parser.add_argument("--model", default="llama3.1:8b", help="Ollama model name")
    parser.add_argument("--output", required=True, help="Output JSON file for results")
    parser.add_argument("--log-file", required=True, help="Log file path")
    parser.add_argument("--ollama-host", default="localhost:11434", help="Ollama server host:port")
    parser.add_argument("--n-results", type=int, default=5, help="Number of chunks to retrieve per query")
    parser.add_argument("--questions-file", help="JSON file with list of questions (optional)")
    args = parser.parse_args()

    logger = setup_logging(args.log_file)
    logger.info("=" * 70)
    logger.info("Bean Lab RAG QA System")
    logger.info("=" * 70)
    logger.info(f"Vector DB:    {args.vector_db}")
    logger.info(f"Model:        {args.model}")
    logger.info(f"Ollama host:  {args.ollama_host}")

    # Check Ollama
    logger.info("Checking Ollama server...")
    if not check_ollama_server(args.ollama_host):
        logger.error(f"Ollama server not reachable at {args.ollama_host}")
        raise SystemExit(1)
    logger.info("✓ Ollama server is running")

    # Load vector store
    logger.info("Loading ChromaDB vector store...")
    collection = load_vector_store(args.vector_db)
    logger.info(f"✓ Collection loaded: {collection.count()} chunks")

    # Load questions
    if args.questions_file and Path(args.questions_file).exists():
        with open(args.questions_file) as f:
            questions = json.load(f)
        logger.info(f"Loaded {len(questions)} questions from {args.questions_file}")
    else:
        questions = DEFAULT_QUESTIONS
        logger.info(f"Using {len(questions)} default test questions")

    # Run QA
    results = []
    for i, question in enumerate(questions, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Question {i}/{len(questions)}")

        try:
            result = answer_question(
                question=question,
                collection=collection,
                model=args.model,
                host=args.ollama_host,
                n_results=args.n_results,
                logger=logger,
            )
            results.append(result)

            # Print to stdout
            print(f"\n{'='*60}")
            print(f"Q{i}: {question}")
            print(f"{'='*60}")
            print(f"A: {result['answer']}")
            print(f"\nSources used:")
            for j, src in enumerate(result["sources"], 1):
                print(f"  [{j}] {src['source']} ({src['year_range']}, page {src['page']})")
            print(f"\nTiming: retrieval={result['retrieval_time_s']}s | generation={result['generation_time_s']}s")

        except Exception as e:
            logger.error(f"Failed on question {i}: {e}")
            results.append({"question": question, "error": str(e)})

    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "model": args.model,
        "vector_db": args.vector_db,
        "total_questions": len(questions),
        "successful": sum(1 for r in results if "error" not in r),
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\n✓ Results saved to {args.output}")
    logger.info(f"Answered {output_data['successful']}/{len(questions)} questions successfully")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
