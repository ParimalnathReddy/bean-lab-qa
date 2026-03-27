#!/usr/bin/env python3
"""
Evaluation benchmark for the Bean Lab RAG QA system.

Covers 5 question categories, 5 questions each (25 total):
  1. Direct lookup      — fact is stated verbatim in a single paper
  2. Cross-section      — fact must be assembled from multiple papers/sections
  3. Inference          — answer follows logically but is not stated directly
  4. Critique           — asks for limitations, gaps, or criticisms
  5. Multi-part         — compound questions requiring multiple sub-answers

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


# ── Benchmark questions ────────────────────────────────────────────────────────
# Each entry: {id, category, question, rubric}
# rubric: list of expected answer elements used for manual or automated scoring.

BENCHMARK = [

    # ── 1. Direct lookup ──────────────────────────────────────────────────────
    {
        "id": "DL-01",
        "category": "direct_lookup",
        "question": "What nitrogen fixation rates have been reported for common bean (Phaseolus vulgaris) varieties?",
        "rubric": [
            "specific numerical fixation rates (kg N/ha or % N derived from atmosphere)",
            "mention of Rhizobium or nodulation",
            "at least one cultivar or experimental condition",
        ],
    },
    {
        "id": "DL-02",
        "category": "direct_lookup",
        "question": "What soil pH range is considered optimal for common bean production?",
        "rubric": [
            "pH range (typically 6.0–7.0)",
            "reference to nutrient availability or root health",
        ],
    },
    {
        "id": "DL-03",
        "category": "direct_lookup",
        "question": "What are the symptoms of bean rust caused by Uromyces appendiculatus?",
        "rubric": [
            "description of pustules or uredinia",
            "leaf or stem symptoms",
            "mention of sporulation or color",
        ],
    },
    {
        "id": "DL-04",
        "category": "direct_lookup",
        "question": "What seed yield advantage has been reported for indeterminate over determinate bean varieties under field conditions?",
        "rubric": [
            "numerical yield comparison (% or kg/ha)",
            "mention of growth habit type (Type I, II, III, IV)",
            "at least one study or environment cited",
        ],
    },
    {
        "id": "DL-05",
        "category": "direct_lookup",
        "question": "What herbicides are commonly used for weed control in dry bean production?",
        "rubric": [
            "at least two herbicide names (e.g., S-metolachlor, imazethapyr, fomesafen)",
            "mention of application timing or target weed species",
        ],
    },

    # ── 2. Cross-section synthesis ────────────────────────────────────────────
    {
        "id": "CS-01",
        "category": "cross_section",
        "question": "How has marker-assisted selection improved bean resistance to bean common mosaic virus (BCMV)?",
        "rubric": [
            "identifies specific resistance genes or QTLs (e.g., I gene, bc-1, bc-3)",
            "describes how MAS is applied in breeding",
            "mentions genetic diversity or gene pools",
        ],
    },
    {
        "id": "CS-02",
        "category": "cross_section",
        "question": "How does intercropping common bean with maize affect bean yield and nitrogen fixation?",
        "rubric": [
            "yield comparison (monoculture vs. intercrop)",
            "light or nutrient competition effects",
            "nitrogen fixation change (increase or decrease with rationale)",
        ],
    },
    {
        "id": "CS-03",
        "category": "cross_section",
        "question": "What is the role of phosphorus availability in nitrogen fixation efficiency in common bean?",
        "rubric": [
            "mechanism linking P to nodulation or nitrogenase activity",
            "threshold P level or fertilization trial results",
            "interaction with soil pH or organic matter",
        ],
    },
    {
        "id": "CS-04",
        "category": "cross_section",
        "question": "How do drought stress and heat stress interact to affect bean pod set and seed filling?",
        "rubric": [
            "separate effects of drought and heat on reproductive stage",
            "combined/interaction effects if reported",
            "mention of canopy temperature or water deficit measurement",
        ],
    },
    {
        "id": "CS-05",
        "category": "cross_section",
        "question": "How have bean yield improvements been achieved through genetic improvement vs. agronomic management since 1990?",
        "rubric": [
            "distinguishes genetic gain from management-based yield increases",
            "references breeding programs or variety releases",
            "references agronomic practices (fertilization, planting density, irrigation)",
        ],
    },

    # ── 3. Inference / cause-effect ───────────────────────────────────────────
    {
        "id": "IN-01",
        "category": "inference",
        "question": "Why is pyramiding rust resistance genes from both Middle American and Andean gene pools considered important for bean breeding programs?",
        "rubric": [
            "explanation of gene pool diversity (Middle American vs. Andean)",
            "reason for combining genes (broader spectrum resistance or durability)",
            "mention of Uromyces races or pathogen variation",
        ],
    },
    {
        "id": "IN-02",
        "category": "inference",
        "question": "Why might tepary bean (Phaseolus acutifolius) be more drought-tolerant than common bean at the physiological level?",
        "rubric": [
            "physiological mechanisms (deeper roots, reduced transpiration, osmotic adjustment)",
            "comparison with common bean",
            "at least one study or trait measurement cited",
        ],
    },
    {
        "id": "IN-03",
        "category": "inference",
        "question": "What explains why biological nitrogen fixation in common bean is often insufficient to meet crop nitrogen demand?",
        "rubric": [
            "factors limiting BNF (carbon cost, soil mineral N, P deficiency)",
            "quantitative shortfall mentioned (kg N/ha fixed vs. required)",
            "interaction with Rhizobium strain or soil conditions",
        ],
    },
    {
        "id": "IN-04",
        "category": "inference",
        "question": "Why does early-season drought have a more severe effect on bean yield than late-season drought?",
        "rubric": [
            "impact on flowering or pod set (vs. seed filling)",
            "critical period concept",
            "supporting data or study result",
        ],
    },
    {
        "id": "IN-05",
        "category": "inference",
        "question": "Why might low-input farming systems in sub-Saharan Africa benefit more from improved bean varieties than high-input systems?",
        "rubric": [
            "yield gap argument (genetic potential vs. realized yield)",
            "role of disease resistance or adaptation in low-input contexts",
            "comparison of input-response between systems",
        ],
    },

    # ── 4. Critique / limitations ─────────────────────────────────────────────
    {
        "id": "CR-01",
        "category": "critique",
        "question": "What are the known challenges in breeding white mold (Sclerotinia sclerotiorum) resistance in dry beans?",
        "rubric": [
            "complexity of host-pathogen interaction",
            "environmental variation in disease expression",
            "lack of complete resistance in germplasm",
            "difficulty of field screening",
        ],
    },
    {
        "id": "CR-02",
        "category": "critique",
        "question": "What methodological limitations affect the measurement of nitrogen fixation in field bean experiments?",
        "rubric": [
            "15N isotope dilution or acetylene reduction assay limitations",
            "spatial or temporal variability in field trials",
            "reference plant selection issues",
        ],
    },
    {
        "id": "CR-03",
        "category": "critique",
        "question": "What gaps remain in understanding the genetic basis of drought tolerance in common bean?",
        "rubric": [
            "complexity of quantitative inheritance",
            "environment-by-genotype interaction",
            "limited marker-trait associations or QTL stability",
        ],
    },
    {
        "id": "CR-04",
        "category": "critique",
        "question": "What are the limitations of using yield trials as the primary method for evaluating drought tolerance in bean breeding programs?",
        "rubric": [
            "confounding factors in field yield trials",
            "inconsistency of drought timing across environments",
            "need for physiological or secondary traits",
        ],
    },
    {
        "id": "CR-05",
        "category": "critique",
        "question": "What are the barriers to adopting improved bean varieties among smallholder farmers in developing countries?",
        "rubric": [
            "seed system or seed access issues",
            "market or cultural preference factors",
            "input cost or risk aversion",
        ],
    },

    # ── 5. Multi-part questions ───────────────────────────────────────────────
    {
        "id": "MP-01",
        "category": "multi_part",
        "question": (
            "What are the most effective fungicides for bean rust management, "
            "how do they work mechanistically, and what resistance risks do they pose?"
        ),
        "rubric": [
            "names at least two effective fungicide classes or active ingredients",
            "explains mode of action (sterol inhibition, respiration, etc.)",
            "addresses fungicide resistance risk or resistance management",
        ],
    },
    {
        "id": "MP-02",
        "category": "multi_part",
        "question": (
            "Compare the drought tolerance mechanisms of tepary bean and common bean, "
            "and explain the implications for introgression breeding programs."
        ),
        "rubric": [
            "mechanism comparison (tepary vs. common bean physiology)",
            "crossability barriers or reproductive isolation issues",
            "practical implications for breeding (backcrossing, marker selection)",
        ],
    },
    {
        "id": "MP-03",
        "category": "multi_part",
        "question": (
            "What soil amendments and inoculants have been used to enhance nitrogen fixation "
            "in common bean, what are their effects on yield, and what factors limit their adoption?"
        ),
        "rubric": [
            "specific amendments or inoculants (Rhizobium, P fertilizer, organic matter)",
            "yield or fixation response data",
            "adoption barriers (cost, availability, farmer knowledge)",
        ],
    },
    {
        "id": "MP-04",
        "category": "multi_part",
        "question": (
            "How does common bacterial blight (CBB) spread in bean fields, "
            "what resistance mechanisms have been identified, "
            "and which breeding strategies are most effective for durable resistance?"
        ),
        "rubric": [
            "describes disease spread pathway (seed, rain splash, insects)",
            "identifies resistance QTLs or genes in Middle American or Andean germplasm",
            "discusses durability strategy (pyramiding, multilines, rotation)",
        ],
    },
    {
        "id": "MP-05",
        "category": "multi_part",
        "question": (
            "What are the nutritional benefits of common beans for human health, "
            "what processing methods improve their bioavailability, "
            "and what genetic variation exists for key nutritional traits?"
        ),
        "rubric": [
            "nutritional content (protein, iron, zinc, fiber, polyphenols)",
            "processing effects (soaking, cooking, fermentation) on anti-nutrients",
            "genetic variability in bean accessions for nutritional traits",
        ],
    },
]


# ── Scoring helpers ────────────────────────────────────────────────────────────

def score_answer(answer: str, rubric: List[str], confidence: str) -> Dict:
    """
    Lightweight automated scoring:
      - rubric_coverage: fraction of rubric items found as keywords in answer
      - has_doi_citation: answer contains a doi: reference
      - has_confidence_label: answer contains a bracketed confidence label
      - confidence: retrieval confidence from BeanRetriever
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

    return {
        "rubric_coverage": coverage,
        "has_doi_citation": has_doi,
        "has_confidence_label": has_conf,
        "retrieval_confidence": confidence,
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
    print(f"\n  By category:")
    for cat, score in cat_avg.items():
        print(f"    {cat:<25} {score:.1%}")
    print(f"\n  Confidence distribution: {conf_dist}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
