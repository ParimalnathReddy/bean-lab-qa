#!/usr/bin/env python3
"""
Batch QA system using ChromaDB vector store + Ollama LLM.
RAG pipeline for Bean Lab research documents.

Improvements over v1:
  - Uses BeanRetriever (query expansion + cross-encoder reranking)
  - Evidence-first prompting with confidence labels
  - Soft failure: always attempts answer, records confidence level
  - n_candidates=20, top_k=10 by default
"""

import os
import sys
import json
import logging
import argparse
import time
import requests
from pathlib import Path
from typing import List, Dict, Optional

import chromadb
from chromadb.config import Settings

# Add src/ to path so retriever and prompts can be imported
sys.path.insert(0, str(Path(__file__).parent))
from retriever import BeanRetriever
from prompts import build_ollama_prompt, format_references


# ── Logging ───────────────────────────────────────────────────────────────────

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


# ── ChromaDB ──────────────────────────────────────────────────────────────────

def load_vector_store(db_path: str, collection_name: str = "bean_research_docs"):
    client = chromadb.PersistentClient(
        path=db_path,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_collection(name=collection_name)


# ── Ollama ────────────────────────────────────────────────────────────────────

def check_ollama_server(host: str = "localhost:11434") -> bool:
    try:
        response = requests.get(f"http://{host}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def query_ollama(prompt: str, model: str = "llama3.1:8b", host: str = "localhost:11434") -> str:
    url = f"http://{host}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 1024,
        },
    }
    response = requests.post(url, json=payload, timeout=180)
    response.raise_for_status()
    return response.json()["response"].strip()


# ── QA Pipeline ───────────────────────────────────────────────────────────────

def answer_question(
    question: str,
    retriever: BeanRetriever,
    model: str,
    host: str,
    top_k: int = 10,
    n_candidates: int = 20,
    year_range: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """Full RAG pipeline: expand → retrieve → rerank → generate."""
    if logger:
        logger.info(f"Question: {question}")

    # Retrieve with expansion and reranking
    t0 = time.time()
    chunks, confidence = retriever.retrieve(
        question,
        top_k=top_k,
        year_range=year_range,
        n_candidates=n_candidates,
    )
    retrieval_time = time.time() - t0

    if logger:
        logger.info(f"Confidence: {confidence} | Retrieved {len(chunks)} chunks in {retrieval_time:.2f}s")
        for i, c in enumerate(chunks, 1):
            rerank = f" rerank={c['rerank_score']:.2f}" if c.get("rerank_score") is not None else ""
            logger.info(f"  [{i}] {c['doi']} p.{c['page']} dist={c['distance']}{rerank}")

    # Build prompt with evidence-first structure
    prompt = build_ollama_prompt(question, chunks, confidence)

    # Generate answer — always attempt, never hard-fail
    t1 = time.time()
    try:
        answer = query_ollama(prompt, model=model, host=host)
    except Exception as e:
        answer = f"[LLM ERROR] {e}. Retrieval succeeded with confidence={confidence}."
        if logger:
            logger.error(f"LLM call failed: {e}")
    generation_time = time.time() - t1

    return {
        "question": question,
        "answer": answer,
        "confidence": confidence,
        "sources": [
            {
                "doi": c["doi"],
                "page": c["page"],
                "year_range": c["year_range"],
                "section": c.get("section", ""),
                "distance": c["distance"],
                "rerank_score": c.get("rerank_score"),
            }
            for c in chunks
        ],
        "retrieval_time_s": round(retrieval_time, 3),
        "generation_time_s": round(generation_time, 3),
        "model": model,
        "year_filter": year_range,
        "n_candidates": n_candidates,
        "top_k": top_k,
    }


# ── Default Questions ─────────────────────────────────────────────────────────
# Covers 5 question types: direct lookup, synthesis, inference, critique, multi-part

DEFAULT_QUESTIONS = [
    # Direct lookup
    "What are the main diseases affecting bean crops and how can they be managed?",
    "What nitrogen fixation rates have been reported for common bean varieties?",
    "What soil pH and nutrient conditions are optimal for bean production?",

    # Cross-section synthesis
    "How does drought stress affect bean yield and what tolerance mechanisms exist?",
    "How has bean breeding improved resistance to bean common mosaic virus?",
    "How does intercropping beans with maize affect productivity?",

    # Inference / cause-effect
    "Why is pyramiding rust resistance genes from Middle American and Andean gene pools considered important?",
    "What explains the yield advantage of indeterminate over determinate bean varieties?",

    # Critique / limitations
    "What are the known limitations or challenges in breeding white mold resistance in dry beans?",
    "What gaps remain in understanding nitrogen fixation efficiency in common bean?",

    # Multi-part
    "What are the most effective herbicides used in bean cultivation, how do they work, and what are their risks?",
    "Compare the drought tolerance mechanisms of tepary bean versus common bean and explain the breeding implications.",
]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch RAG QA for Bean Lab documents")
    parser.add_argument("--vector-db", required=True)
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--output", required=True)
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--ollama-host", default="localhost:11434")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--n-candidates", type=int, default=20)
    parser.add_argument("--questions-file", help="JSON list of questions")
    parser.add_argument("--chunks-file", help="Unused; kept for compatibility")
    args = parser.parse_args()

    logger = setup_logging(args.log_file)
    logger.info("=" * 70)
    logger.info("Bean Lab Batch QA System v2")
    logger.info(f"Vector DB:    {args.vector_db}")
    logger.info(f"Model:        {args.model}")
    logger.info(f"top_k:        {args.top_k}  |  n_candidates: {args.n_candidates}")
    logger.info("=" * 70)

    if not check_ollama_server(args.ollama_host):
        logger.error(f"Ollama server not reachable at {args.ollama_host}")
        raise SystemExit(1)
    logger.info("✓ Ollama server running")

    collection = load_vector_store(args.vector_db)
    logger.info(f"✓ Collection loaded: {collection.count()} chunks")

    retriever = BeanRetriever(collection)
    retriever._load_cross_encoder()

    if args.questions_file and Path(args.questions_file).exists():
        with open(args.questions_file) as f:
            questions = json.load(f)
    else:
        questions = DEFAULT_QUESTIONS
    logger.info(f"Running {len(questions)} questions")

    results = []
    for i, question in enumerate(questions, 1):
        logger.info(f"\n{'='*60}\nQuestion {i}/{len(questions)}")
        try:
            result = answer_question(
                question=question,
                retriever=retriever,
                model=args.model,
                host=args.ollama_host,
                top_k=args.top_k,
                n_candidates=args.n_candidates,
                logger=logger,
            )
            results.append(result)

            print(f"\n{'='*60}")
            print(f"Q{i}: {question}")
            print(f"Confidence: {result['confidence']}")
            print(f"{'='*60}")
            print(result["answer"])
            print("\nSources:")
            for j, src in enumerate(result["sources"], 1):
                print(f"  [{j}] doi:{src['doi']}  p.{src['page']}  dist={src['distance']}")

        except Exception as e:
            logger.error(f"Failed on question {i}: {e}")
            results.append({"question": question, "error": str(e)})

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "model": args.model,
        "vector_db": args.vector_db,
        "top_k": args.top_k,
        "n_candidates": args.n_candidates,
        "total_questions": len(questions),
        "successful": sum(1 for r in results if "error" not in r),
        "confidence_distribution": {
            label: sum(1 for r in results if r.get("confidence") == label)
            for label in ["SUPPORTED", "PARTIALLY_SUPPORTED", "INFERRED", "UNSUPPORTED"]
        },
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\n✓ Results saved to {args.output}")
    logger.info(f"Answered {output_data['successful']}/{len(questions)} questions")
    logger.info(f"Confidence: {output_data['confidence_distribution']}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
