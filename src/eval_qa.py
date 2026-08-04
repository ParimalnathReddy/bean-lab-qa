#!/usr/bin/env python3
"""
Evaluation benchmark for the Bean Lab RAG QA system.

Coarse keyword-rubric scoring — see eval_ragas.py for LLM-judged RAGAS
metrics (faithfulness, answer relevancy, context precision, context recall),
which are far less foolable than keyword matching. Both scripts import the
same 75-question set from benchmark_questions.py, across 7 categories:
  1. Direct lookup      — fact is stated verbatim in a single paper
  2. Cross-section      — fact must be assembled from multiple papers/sections
  3. Inference          — answer follows logically but is not stated directly
  4. Critique           — asks for limitations, gaps, or criticisms
  5. Multi-part         — compound questions requiring multiple sub-answers
  6. Adversarial        — fake genes/cultivars/out-of-scope topics; probes
                          hallucination rather than fact recall
  7. Multi-hop          — requires synthesizing across 3+ papers/sub-topics

Usage (on HPCC with Ollama running):
    python eval_qa.py \\
        --vector-db /mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa/vector_db \\
        --output    /mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa/data/eval_results.json \\
        --log-file  /mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa/logs/eval.log

Optional flags:
    --model        Ollama model name (default: llama3.1:8b)
    --ollama-host  Ollama server address (default: localhost:11434)
    --top-k        Sources to use per answer (default: 10)
    --n-candidates Candidates to retrieve before reranking (default: 20)
"""

import os
import re
import sys
import json
import logging
import argparse
import time
from pathlib import Path
from typing import List, Dict, Optional

import chromadb
from chromadb.config import Settings

sys.path.insert(0, str(Path(__file__).parent))
from retriever import BeanRetriever
from qa_with_ollama import (
    setup_logging,
    load_vector_store,
    check_ollama_server,
    answer_question,
)
from benchmark_questions import BENCHMARK


# ── Scoring helpers ────────────────────────────────────────────────────────────

def score_answer(answer: str, rubric: List[str], confidence: str,
                  expect_low_confidence: bool = False) -> Dict:
    """
    Lightweight automated scoring:
      - rubric_coverage: fraction of rubric items found as keywords in answer
      - has_doi_citation: answer contains a doi: reference
      - has_confidence_label: answer contains a bracketed confidence label
      - confidence: retrieval confidence from BeanRetriever
      - confidence_appropriate: for adversarial questions (expect_low_confidence=True),
        whether the system actually hedged (INFERRED/UNSUPPORTED) rather than
        confidently answering about a fabricated gene/cultivar/out-of-scope topic.
        None for non-adversarial questions, where this check doesn't apply.
    """
    answer_lower = answer.lower()

    # Check rubric keyword coverage (coarse heuristic)
    rubric_hits = 0
    for item in rubric:
        # If any 3+ char word from the rubric appears in the answer, count it
        words = [w for w in re.findall(r'\b\w{3,}\b', item.lower()) if w not in
                 {"the", "are", "for", "and", "that", "have", "been", "with", "from", "this"}]
        if words and any(w in answer_lower for w in words):
            rubric_hits += 1
    coverage = round(rubric_hits / len(rubric), 2) if rubric else 0.0

    has_doi  = bool(re.search(r'doi:\s*10\.\d{4,}', answer_lower))
    has_conf = bool(re.search(r'\[(supported|partially_supported|inferred|unsupported)\]',
                               answer_lower))

    confidence_appropriate = None
    if expect_low_confidence:
        confidence_appropriate = confidence in ("INFERRED", "UNSUPPORTED")

    return {
        "rubric_coverage": coverage,
        "has_doi_citation": has_doi,
        "has_confidence_label": has_conf,
        "retrieval_confidence": confidence,
        "confidence_appropriate": confidence_appropriate,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Bean Lab RAG Evaluation Benchmark")
    parser.add_argument("--vector-db",    required=True)
    parser.add_argument("--output",       required=True)
    parser.add_argument("--log-file",     required=True)
    parser.add_argument("--model",        default="llama3.1:8b")
    parser.add_argument("--ollama-host",  default="localhost:11434")
    parser.add_argument("--top-k",        type=int, default=10)
    parser.add_argument("--n-candidates", type=int, default=20)
    parser.add_argument("--categories",   nargs="*",
                        help="Run only these categories (e.g. direct_lookup inference)")
    args = parser.parse_args()

    logger = setup_logging(args.log_file)
    logger.info("=" * 70)
    logger.info("Bean Lab RAG Evaluation Benchmark")
    logger.info(f"Model:        {args.model}")
    logger.info(f"Vector DB:    {args.vector_db}")
    logger.info(f"top_k={args.top_k}  n_candidates={args.n_candidates}")
    logger.info("=" * 70)

    if not check_ollama_server(args.ollama_host):
        logger.error(f"Ollama not reachable at {args.ollama_host}")
        raise SystemExit(1)

    collection = load_vector_store(args.vector_db)
    logger.info(f"✓ Collection loaded: {collection.count()} chunks")

    retriever = BeanRetriever(collection)
    retriever._load_cross_encoder()

    # Filter categories if requested
    questions = BENCHMARK
    if args.categories:
        questions = [q for q in BENCHMARK if q["category"] in args.categories]
        logger.info(f"Running {len(questions)} questions in categories: {args.categories}")
    else:
        logger.info(f"Running all {len(questions)} benchmark questions")

    results = []
    category_scores: Dict[str, List[float]] = {}

    for i, item in enumerate(questions, 1):
        logger.info(f"\n{'='*60}\n[{item['id']}] {item['category']} ({i}/{len(questions)})")
        logger.info(f"Q: {item['question']}")

        t0 = time.time()
        try:
            result = answer_question(
                question=item["question"],
                retriever=retriever,
                model=args.model,
                host=args.ollama_host,
                top_k=args.top_k,
                n_candidates=args.n_candidates,
                logger=logger,
            )
            elapsed = time.time() - t0

            scores = score_answer(
                result["answer"],
                item["rubric"],
                result["confidence"],
                expect_low_confidence=item.get("expect_low_confidence", False),
            )

            entry = {
                **item,
                "answer": result["answer"],
                "confidence": result["confidence"],
                "scores": scores,
                "sources": result["sources"],
                "retrieval_time_s": result["retrieval_time_s"],
                "generation_time_s": result["generation_time_s"],
                "total_time_s": round(elapsed, 2),
            }
            results.append(entry)

            cat = item["category"]
            category_scores.setdefault(cat, []).append(scores["rubric_coverage"])

            logger.info(f"  Confidence: {result['confidence']}")
            logger.info(f"  Rubric coverage: {scores['rubric_coverage']:.0%}")
            logger.info(f"  DOI cited: {scores['has_doi_citation']} | "
                        f"Confidence label: {scores['has_confidence_label']}")

            print(f"\n[{item['id']}] {item['question'][:80]}...")
            print(f"  Confidence: {result['confidence']} | "
                  f"Coverage: {scores['rubric_coverage']:.0%} | "
                  f"DOI: {scores['has_doi_citation']}")

        except Exception as e:
            logger.error(f"  Failed: {e}")
            results.append({**item, "error": str(e)})

    # ── Aggregate statistics ───────────────────────────────────────────────────
    successful  = [r for r in results if "error" not in r]
    n_ok        = len(successful)
    avg_coverage = round(sum(r["scores"]["rubric_coverage"] for r in successful) / n_ok, 3) if n_ok else 0
    doi_rate     = round(sum(r["scores"]["has_doi_citation"] for r in successful) / n_ok, 3) if n_ok else 0
    conf_rate    = round(sum(r["scores"]["has_confidence_label"] for r in successful) / n_ok, 3) if n_ok else 0

    conf_dist = {
        label: sum(1 for r in successful if r.get("confidence") == label)
        for label in ["SUPPORTED", "PARTIALLY_SUPPORTED", "INFERRED", "UNSUPPORTED"]
    }

    cat_avg = {cat: round(sum(scores) / len(scores), 3)
               for cat, scores in category_scores.items()}

    adversarial_checked = [
        r["scores"]["confidence_appropriate"] for r in successful
        if r["scores"].get("confidence_appropriate") is not None
    ]
    adversarial_hedge_rate = (
        round(sum(adversarial_checked) / len(adversarial_checked), 3)
        if adversarial_checked else None
    )

    output = {
        "model": args.model,
        "vector_db": args.vector_db,
        "top_k": args.top_k,
        "n_candidates": args.n_candidates,
        "total_questions": len(questions),
        "successful": n_ok,
        "aggregate": {
            "avg_rubric_coverage": avg_coverage,
            "doi_citation_rate": doi_rate,
            "confidence_label_rate": conf_rate,
            "adversarial_appropriate_hedge_rate": adversarial_hedge_rate,
        },
        "by_category": cat_avg,
        "confidence_distribution": conf_dist,
        "results": results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\n{'='*70}")
    logger.info(f"Evaluation complete — {n_ok}/{len(questions)} answered")
    logger.info(f"Avg rubric coverage:    {avg_coverage:.1%}")
    logger.info(f"DOI citation rate:      {doi_rate:.1%}")
    logger.info(f"Confidence label rate:  {conf_rate:.1%}")
    if adversarial_hedge_rate is not None:
        logger.info(f"Adversarial hedge rate: {adversarial_hedge_rate:.1%} "
                     f"(of {len(adversarial_checked)} adversarial questions)")
    logger.info(f"By category: {cat_avg}")
    logger.info(f"Confidence distribution: {conf_dist}")
    logger.info(f"Results → {args.output}")
    logger.info("=" * 70)

    print(f"\n{'='*70}")
    print(f"EVALUATION SUMMARY")
    print(f"  Successful:           {n_ok}/{len(questions)}")
    print(f"  Avg rubric coverage:  {avg_coverage:.1%}")
    print(f"  DOI citation rate:    {doi_rate:.1%}")
    print(f"  Confidence labels:    {conf_rate:.1%}")
    if adversarial_hedge_rate is not None:
        print(f"  Adversarial hedge rate: {adversarial_hedge_rate:.1%} "
              f"(of {len(adversarial_checked)} adversarial questions)")
    print(f"\n  By category:")
    for cat, score in cat_avg.items():
        print(f"    {cat:<25} {score:.1%}")
    print(f"\n  Confidence distribution: {conf_dist}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
