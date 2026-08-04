#!/usr/bin/env python3
"""
Batch QA system using ChromaDB vector store + Ollama LLM.
RAG pipeline for Bean Lab research documents.

Improvements over v1:
  - Uses BeanRetriever (query expansion + cross-encoder reranking)
  - Evidence-first prompting with confidence labels
  - Soft failure: always attempts answer, records confidence level
  - n_candidates=20, top_k=10 by default
  - Hybrid BM25+dense retrieval (Reciprocal Rank Fusion) is automatic inside
    BeanRetriever whenever vector_db/bm25_index.pkl exists — no flag needed
  - Query router (Change 7): literature retrieval and structured trial-data
    lookup run independently per question; gene mentions in the generated
    answer are checked against NCBI/UniProt/classical-symbols afterward
"""

import os
import sys
import json
import logging
import argparse
import time
import requests
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import List, Dict, Optional

import chromadb
from chromadb.config import Settings

# Add src/ to path so retriever and prompts can be imported
sys.path.insert(0, str(Path(__file__).parent))
from retriever import BeanRetriever
from prompts import build_ollama_prompt, format_references
from query_router import route
from structured_data import query_trial_data, format_trial_results
from gene_validator import extract_gene_mentions, validate_genes


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
    strict_evidence: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """Full RAG pipeline: route -> retrieve + query structured data (independently) -> generate -> gene-validate."""
    if logger:
        logger.info(f"Question: {question}")

    decision = route(question)
    if logger:
        logger.info(f"Routing: literature={decision.use_literature} "
                    f"structured_data={decision.use_structured_data} "
                    f"gene_validation={decision.run_gene_validation}")
        if decision.use_structured_data:
            logger.info(f"  Detected: cultivars={decision.detected_cultivars} "
                        f"locations={decision.detected_locations} traits={decision.detected_traits} "
                        f"year={decision.detected_year} year_range={decision.detected_year_range}")

    # Literature retrieval and structured-data lookup are independent of each
    # other, so run them concurrently rather than sequentially (Change 7).
    # Neither is slow on its own (SQLite: sub-millisecond, vector retrieval:
    # sub-second) but structuring the two as independent submissions rather
    # than a sequential dependency keeps the intent clear and leaves room for
    # real parallel gains if either grows more expensive later.
    t0 = time.time()
    chunks: List[Dict] = []
    confidence = "UNSUPPORTED"
    trial_rows: List[Dict] = []

    with ThreadPoolExecutor(max_workers=2) as executor:
        lit_future = executor.submit(
            retriever.retrieve, question, top_k=top_k, year_range=year_range, n_candidates=n_candidates,
        ) if decision.use_literature else None

        struct_future = executor.submit(
            query_trial_data,
            cultivar=decision.detected_cultivars[0] if decision.detected_cultivars else None,
            location=decision.detected_locations[0] if decision.detected_locations else None,
            year=decision.detected_year,
            year_range=decision.detected_year_range,
            trait=decision.detected_traits[0] if decision.detected_traits else None,
        ) if decision.use_structured_data else None

        if lit_future is not None:
            chunks, confidence = lit_future.result()
        if struct_future is not None:
            trial_rows = struct_future.result()

    retrieval_time = time.time() - t0

    if logger:
        logger.info(f"Confidence: {confidence} | Retrieved {len(chunks)} chunks, "
                    f"{len(trial_rows)} trial-data row(s) in {retrieval_time:.2f}s")
        for i, c in enumerate(chunks, 1):
            rerank = f" rerank={c['rerank_score']:.2f}" if c.get("rerank_score") is not None else ""
            bm25 = f" bm25={c['bm25_score']:.2f}" if c.get("bm25_score") is not None else ""
            channels = f" channels={','.join(c.get('retrieval_channels', []))}" if c.get("retrieval_channels") else ""
            logger.info(f"  [{i}] {c['doi']} p.{c['page']} dist={c.get('distance')}{bm25}{rerank}{channels}")

    routing_info = asdict(decision)

    # Structured data can give us real numbers even when literature-only
    # confidence is weak, so don't short-circuit on strict_evidence if we
    # found any trial-data rows to work with.
    if strict_evidence and confidence == "UNSUPPORTED" and not trial_rows:
        answer = "Insufficient evidence found in the indexed Bean Lab papers."
        return {
            "question": question,
            "answer": answer,
            "confidence": confidence,
            "fallback": "strict_evidence_unsupported",
            "routing": routing_info,
            "trial_data_rows": [],
            "gene_checks": [],
            "sources": [],
            "retrieval_time_s": round(retrieval_time, 3),
            "generation_time_s": 0.0,
            "model": model,
            "year_filter": year_range,
            "n_candidates": n_candidates,
            "top_k": top_k,
        }

    # Build prompt with evidence-first structure, prepending trial data (if
    # the router activated it) ahead of the literature sources
    trial_data_block = format_trial_results(trial_rows) if trial_rows else ""
    prompt = build_ollama_prompt(question, chunks, confidence, trial_data_block=trial_data_block)

    # Generate answer — always attempt, never hard-fail
    t1 = time.time()
    try:
        answer = query_ollama(prompt, model=model, host=host)
    except Exception as e:
        answer = f"[LLM ERROR] {e}. Retrieval succeeded with confidence={confidence}."
        if logger:
            logger.error(f"LLM call failed: {e}")
    generation_time = time.time() - t1

    # Gene validation is a free post-generation check (regex + dict lookup,
    # no LLM call) that already self-gates on whether the ANSWER mentions any
    # genes at all — so it runs unconditionally rather than being hard-gated
    # by decision.run_gene_validation. That flag reflects the QUERY, but a
    # query with no genetics vocabulary can still produce an answer that
    # names a gene, and catching a hallucinated one there matters just as
    # much. The flag is logged/returned as a routing signal, not used to
    # skip a check that costs nothing extra to run.
    gene_checks: List[Dict] = []
    try:
        mentions = extract_gene_mentions(answer)
        if mentions:
            gene_checks = validate_genes(mentions)
    except Exception as e:
        if logger:
            logger.warning(f"Gene validation failed: {e}")

    return {
        "question": question,
        "answer": answer,
        "confidence": confidence,
        "routing": routing_info,
        "trial_data_rows": trial_rows,
        "gene_checks": gene_checks,
        "sources": [
            {
                "doi": c["doi"],
                "page": c["page"],
                "year_range": c["year_range"],
                "section": c.get("section", ""),
                "distance": c.get("distance"),
                "bm25_score": c.get("bm25_score"),
                "rerank_score": c.get("rerank_score"),
                "retrieval_channels": c.get("retrieval_channels", []),
                "text": c.get("text", ""),  # needed as RAGAS retrieved_contexts (eval_ragas.py)
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
    parser.add_argument("--strict-evidence", action="store_true", help="Return insufficient evidence instead of generating on unsupported retrieval")
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
    retriever._load_bm25()  # reports whether hybrid BM25 retrieval is available

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
                strict_evidence=args.strict_evidence,
                logger=logger,
            )
            results.append(result)

            print(f"\n{'='*60}")
            print(f"Q{i}: {question}")
            print(f"Confidence: {result['confidence']}")
            routing = result.get("routing", {})
            print(f"Routing: literature={routing.get('use_literature')} "
                  f"structured_data={routing.get('use_structured_data')} "
                  f"gene_validation={routing.get('run_gene_validation')}")
            if result.get("trial_data_rows"):
                print(f"Trial data rows used: {len(result['trial_data_rows'])}")
            if result.get("gene_checks"):
                flagged = [g for g in result["gene_checks"] if not g.get("verified")]
                print(f"Gene mentions checked: {len(result['gene_checks'])} "
                      f"({len(flagged)} unverified)")
            print(f"{'='*60}")
            print(result["answer"])
            print("\nSources:")
            for j, src in enumerate(result["sources"], 1):
                bm25 = f"  bm25={src['bm25_score']:.2f}" if src.get("bm25_score") is not None else ""
                channels = f"  channels={','.join(src.get('retrieval_channels', []))}" if src.get("retrieval_channels") else ""
                print(f"  [{j}] doi:{src['doi']}  p.{src['page']}  dist={src.get('distance')}{bm25}{channels}")

        except Exception as e:
            logger.error(f"Failed on question {i}: {e}")
            results.append({"question": question, "error": str(e)})

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "model": args.model,
        "vector_db": args.vector_db,
        "top_k": args.top_k,
        "n_candidates": args.n_candidates,
        "hybrid": retriever._bm25 is not None,
        "strict_evidence": args.strict_evidence,
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
