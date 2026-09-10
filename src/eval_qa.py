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
#
# Change 20 — retrieval and generation are scored as two INDEPENDENT layers,
# never blended into one number. The original score_answer() checked rubric
# keywords against the final ANSWER only, which conflates two different
# things a rubric miss could mean: retrieval never found the fact, or
# retrieval found it and generation just phrased it differently. It also
# can't catch the opposite failure — a fluent, plausible-sounding answer
# that happens to use the expected keywords despite retrieval never having
# surfaced that evidence at all (an unsupported/hallucinated claim scoring
# as if it were correct). Neither failure is visible in a single blended
# score; both are real risks in a RAG pipeline, and optimizing against one
# number risks tuning generation to sound right while retrieval quietly
# stays broken, or vice versa. See docs/decisions.md.

_RUBRIC_STOPWORDS = {"the", "are", "for", "and", "that", "have", "been",
                      "with", "from", "this"}


def _rubric_hits(text: str, rubric: List[str]) -> List[bool]:
    """Per rubric item: does any 3+ char content word from it appear in `text`?"""
    text_lower = text.lower()
    hits = []
    for item in rubric:
        words = [w for w in re.findall(r'\b\w{3,}\b', item.lower())
                  if w not in _RUBRIC_STOPWORDS]
        hits.append(bool(words) and any(w in text_lower for w in words))
    return hits


def score_answer(answer: str, contexts: List[str], rubric: List[str],
                  confidence: str, expect_low_confidence: bool = False) -> Dict:
    """
    Lightweight automated scoring, retrieval and generation reported
    separately:
      - retrieval_rubric_coverage: fraction of rubric items found as keywords
        ANYWHERE IN THE RETRIEVED CONTEXT — did the correct evidence appear
        at all? Independent of what the LLM did with it.
      - answer_rubric_coverage: fraction of rubric items found as keywords in
        the final ANSWER — the old "rubric_coverage," kept but relabeled so
        it's never mistaken for a retrieval-quality signal.
      - unsupported_rubric_items: count of rubric items present in the
        answer but NOT in retrieved context — claims with no visible
        supporting evidence, the "good wording hides bad retrieval" case.
      - dropped_rubric_items: count of rubric items present in retrieved
        context but NOT in the answer — evidence was available and the
        generation step didn't use it.
      - has_doi_citation: answer contains a doi: reference (generation layer)
      - has_confidence_label: answer contains a bracketed confidence label
      - confidence: retrieval confidence from BeanRetriever (retrieval layer)
      - confidence_appropriate: for adversarial questions (expect_low_confidence=True),
        whether the system actually hedged (INFERRED/UNSUPPORTED) rather than
        confidently answering about a fabricated gene/cultivar/out-of-scope topic.
        None for non-adversarial questions, where this check doesn't apply.
    """
    answer_lower = answer.lower()
    context_text = " ".join(contexts)

    retrieval_hits = _rubric_hits(context_text, rubric)
    answer_hits = _rubric_hits(answer_lower, rubric)

    retrieval_coverage = round(sum(retrieval_hits) / len(rubric), 2) if rubric else 0.0
    answer_coverage = round(sum(answer_hits) / len(rubric), 2) if rubric else 0.0

    unsupported = sum(1 for r, a in zip(retrieval_hits, answer_hits) if a and not r)
    dropped = sum(1 for r, a in zip(retrieval_hits, answer_hits) if r and not a)

    has_doi  = bool(re.search(r'doi:\s*10\.\d{4,}', answer_lower))
    has_conf = bool(re.search(r'\[(supported|partially_supported|inferred|unsupported)\]',
                               answer_lower))

    confidence_appropriate = None
    if expect_low_confidence:
        confidence_appropriate = confidence in ("INFERRED", "UNSUPPORTED")

    return {
        "retrieval_rubric_coverage": retrieval_coverage,
        "answer_rubric_coverage": answer_coverage,
        "unsupported_rubric_items": unsupported,
        "dropped_rubric_items": dropped,
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

            contexts = [s["text"] for s in result["sources"] if s.get("text")]
            scores = score_answer(
                result["answer"],
                contexts,
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
            category_scores.setdefault(cat, {"retrieval": [], "answer": []})
            category_scores[cat]["retrieval"].append(scores["retrieval_rubric_coverage"])
            category_scores[cat]["answer"].append(scores["answer_rubric_coverage"])

            logger.info(f"  Confidence: {result['confidence']}")
            logger.info(f"  Retrieval coverage: {scores['retrieval_rubric_coverage']:.0%}  "
                        f"(evidence present in retrieved context)")
            logger.info(f"  Answer coverage:    {scores['answer_rubric_coverage']:.0%}  "
                        f"(evidence present in final answer)")
            if scores["unsupported_rubric_items"]:
                logger.info(f"  ⚠ {scores['unsupported_rubric_items']} rubric item(s) in the "
                            f"answer with no matching retrieved evidence")
            if scores["dropped_rubric_items"]:
                logger.info(f"  ⚠ {scores['dropped_rubric_items']} rubric item(s) retrieved "
                            f"but not used in the answer")
            logger.info(f"  DOI cited: {scores['has_doi_citation']} | "
                        f"Confidence label: {scores['has_confidence_label']}")

            print(f"\n[{item['id']}] {item['question'][:80]}...")
            print(f"  Confidence: {result['confidence']} | "
                  f"Retrieval: {scores['retrieval_rubric_coverage']:.0%} | "
                  f"Answer: {scores['answer_rubric_coverage']:.0%} | "
                  f"DOI: {scores['has_doi_citation']}")

        except Exception as e:
            logger.error(f"  Failed: {e}")
            results.append({**item, "error": str(e)})

    # ── Aggregate statistics — retrieval and generation kept as separate ────────
    # layers throughout (Change 20). Never averaged together into one number:
    # a low retrieval score and a low answer score point at different fixes
    # (better retrieval/reranking vs. better prompting/faithfulness), and a
    # blended number would hide which one actually needs attention.
    successful  = [r for r in results if "error" not in r]
    n_ok        = len(successful)

    avg_retrieval_coverage = (
        round(sum(r["scores"]["retrieval_rubric_coverage"] for r in successful) / n_ok, 3)
        if n_ok else 0
    )
    avg_answer_coverage = (
        round(sum(r["scores"]["answer_rubric_coverage"] for r in successful) / n_ok, 3)
        if n_ok else 0
    )
    unsupported_rate = (
        round(sum(1 for r in successful if r["scores"]["unsupported_rubric_items"] > 0) / n_ok, 3)
        if n_ok else 0
    )
    dropped_rate = (
        round(sum(1 for r in successful if r["scores"]["dropped_rubric_items"] > 0) / n_ok, 3)
        if n_ok else 0
    )
    doi_rate     = round(sum(r["scores"]["has_doi_citation"] for r in successful) / n_ok, 3) if n_ok else 0
    conf_rate    = round(sum(r["scores"]["has_confidence_label"] for r in successful) / n_ok, 3) if n_ok else 0

    conf_dist = {
        label: sum(1 for r in successful if r.get("confidence") == label)
        for label in ["SUPPORTED", "PARTIALLY_SUPPORTED", "INFERRED", "UNSUPPORTED"]
    }

    cat_avg = {
        cat: {
            "retrieval_rubric_coverage": round(sum(s["retrieval"]) / len(s["retrieval"]), 3),
            "answer_rubric_coverage": round(sum(s["answer"]) / len(s["answer"]), 3),
        }
        for cat, s in category_scores.items()
    }

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
            "retrieval": {
                "avg_rubric_coverage": avg_retrieval_coverage,
                "note": "did the correct evidence appear in retrieved context?",
            },
            "generation": {
                "avg_rubric_coverage": avg_answer_coverage,
                "unsupported_claim_rate": unsupported_rate,
                "evidence_dropped_rate": dropped_rate,
                "doi_citation_rate": doi_rate,
                "confidence_label_rate": conf_rate,
                "note": "did the answer faithfully use retrieved evidence, "
                        "without inventing content retrieval never surfaced?",
            },
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
    logger.info(f"RETRIEVAL  avg coverage:        {avg_retrieval_coverage:.1%}")
    logger.info(f"GENERATION avg coverage:        {avg_answer_coverage:.1%}")
    logger.info(f"GENERATION unsupported-claim rate: {unsupported_rate:.1%}  "
                f"(answer states something retrieval never surfaced)")
    logger.info(f"GENERATION evidence-dropped rate:  {dropped_rate:.1%}  "
                f"(retrieval found it, answer didn't use it)")
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
    print(f"  Successful:                {n_ok}/{len(questions)}")
    print(f"\n  RETRIEVAL LAYER — did the correct evidence appear?")
    print(f"    Avg coverage:            {avg_retrieval_coverage:.1%}")
    print(f"\n  GENERATION LAYER — did the answer faithfully use that evidence?")
    print(f"    Avg coverage:            {avg_answer_coverage:.1%}")
    print(f"    Unsupported-claim rate:  {unsupported_rate:.1%}  (stated, not retrieved)")
    print(f"    Evidence-dropped rate:   {dropped_rate:.1%}  (retrieved, not stated)")
    print(f"    DOI citation rate:       {doi_rate:.1%}")
    print(f"    Confidence labels:       {conf_rate:.1%}")
    if adversarial_hedge_rate is not None:
        print(f"    Adversarial hedge rate:  {adversarial_hedge_rate:.1%} "
              f"(of {len(adversarial_checked)} adversarial questions)")
    print(f"\n  By category (retrieval | generation):")
    for cat, scores in cat_avg.items():
        print(f"    {cat:<25} {scores['retrieval_rubric_coverage']:.0%}  |  "
              f"{scores['answer_rubric_coverage']:.0%}")
    print(f"\n  Confidence distribution: {conf_dist}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
