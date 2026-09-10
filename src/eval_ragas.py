#!/usr/bin/env python3
"""
RAGAS evaluation pipeline for the Bean Lab RAG QA system (Change 4).

Runs the SAME BeanRetriever + Ollama generation pipeline as qa_with_ollama.py
(hybrid BM25+dense retrieval, cross-encoder reranking, evidence-first
prompting — nothing about the production path is reimplemented here) against
the shared 75-question benchmark in benchmark_questions.py, then scores each
(question, answer, retrieved_contexts) triple with RAGAS metrics judged by an
LLM — Gemini 2.5 Flash by default — instead of eval_qa.py's coarse
keyword-rubric matching.

Metrics computed:
  faithfulness              Does the answer only contain claims supported by
                             the retrieved context? No ground truth needed.
  answer_relevancy           Does the answer actually address the question?
                             No ground truth needed (uses embeddings only).
  context_precision_no_ref   Are the retrieved chunks actually relevant?
                             Computed via RAGAS's LLMContextPrecisionWithoutReference,
                             which judges relevance using the generated answer
                             as the yardstick instead of a ground-truth
                             reference. Runs on every question.
  context_precision          The reference-based variant of the above. Only
                             computed for questions with ground_truth filled
                             in benchmark_questions.py — 11 of 75 as of
                             Change 11 (see that module's docstring, item 1,
                             for exactly which and why not more).
  context_recall             Did retrieval capture everything needed to
                             answer? REQUIRES a ground_truth reference answer.
                             Same 11/75 coverage as context_precision above.

Why context_precision_no_ref instead of forcing a ground truth everywhere:
authoring reference answers without verified access to the actual paper text
would mean grading retrieval against invented facts — worse than no
context_recall at all. Change 11 populated ground_truth for 11 questions by
searching data/processed_chunks.json directly and using only what real chunk
text actually supported (see benchmark_questions.py's docstring); the other
64 remain None precisely because that same search came up empty or unclear
for them, not because no one got around to it. Whenever more get filled in
the same verified way, context_precision/context_recall activate
automatically for those questions, no code changes needed here.

Dependency note: ragas + langchain-google-genai + langchain-community is a
version-sensitive combination — pin to the set in requirements.txt
(ragas==0.2.15, langchain-community==0.3.7, langchain-google-genai==2.0.11),
verified to actually resolve and import together. Newer langchain-google-genai
releases require langchain-core>=1.0, which conflicts with ragas 0.2.x's
langchain-core<0.4 requirement.

Usage (on HPCC with Ollama running):
    export Googlegeminiapi=<your-gemini-api-key>   # or GEMINI_API_KEY
    python eval_ragas.py \\
        --vector-db /mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa/vector_db \\
        --output    /mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa/data/ragas_results.json \\
        --log-file  /mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa/logs/ragas_eval.log

Optional flags:
    --model            Ollama model name (default: llama3.1:8b)
    --ollama-host      Ollama server address (default: localhost:11434)
    --top-k            Sources to use per answer (default: 10)
    --n-candidates     Candidates to retrieve before reranking (default: 20)
    --categories       Run only these categories (e.g. adversarial multi_hop)
    --limit            Cap number of questions (smoke-testing)
    --gemini-model     Judge model (default: gemini-2.5-flash)
    --embedding-model  Embedding model for answer_relevancy (default: matches
                       the production system, BAAI/bge-large-en-v1.5)
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))
from retriever import BeanRetriever
from qa_with_ollama import setup_logging, load_vector_store, check_ollama_server, answer_question
from benchmark_questions import BENCHMARK


def _import_ragas_stack() -> Dict:
    """
    Import RAGAS + langchain deps lazily, with an actionable error message if
    the (fragile, version-sensitive) dependency chain isn't installed —
    rather than a bare ImportError traceback.
    """
    try:
        from ragas import evaluate, SingleTurnSample, EvaluationDataset
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            LLMContextPrecisionWithoutReference,
            context_precision,
            context_recall,
        )
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError as e:
        raise SystemExit(
            f"Missing dependency for eval_ragas.py: {e}\n\n"
            "Install the verified-compatible set with:\n"
            "  pip install ragas==0.2.15 langchain-community==0.3.7 "
            "langchain-google-genai==2.0.11\n\n"
            "These exact versions matter: ragas 0.2.x requires langchain-core<0.4, "
            "which conflicts with langchain-google-genai>=3.0 (needs langchain-core>=1.0). "
            "See requirements.txt for the full pinned set."
        ) from e

    return {
        "evaluate": evaluate,
        "SingleTurnSample": SingleTurnSample,
        "EvaluationDataset": EvaluationDataset,
        "LangchainLLMWrapper": LangchainLLMWrapper,
        "LangchainEmbeddingsWrapper": LangchainEmbeddingsWrapper,
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "LLMContextPrecisionWithoutReference": LLMContextPrecisionWithoutReference,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "ChatGoogleGenerativeAI": ChatGoogleGenerativeAI,
        "HuggingFaceEmbeddings": HuggingFaceEmbeddings,
    }


def _nanmean(values: List[Optional[float]]) -> Optional[float]:
    """Mean over non-None, non-NaN values; None if nothing valid to average."""
    clean = [v for v in values if v is not None and v == v]  # v == v is False for NaN
    return round(sum(clean) / len(clean), 4) if clean else None


def main():
    parser = argparse.ArgumentParser(description="RAGAS evaluation for the Bean Lab RAG QA system")
    parser.add_argument("--vector-db", required=True)
    parser.add_argument("--output", default="data/ragas_results.json")
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--ollama-host", default="localhost:11434")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--n-candidates", type=int, default=20)
    parser.add_argument("--categories", nargs="*",
                        help="Run only these categories (e.g. adversarial multi_hop)")
    parser.add_argument("--limit", type=int, help="Cap number of questions (smoke test)")
    parser.add_argument("--gemini-model", default=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip())
    parser.add_argument("--embedding-model", default="BAAI/bge-large-en-v1.5")
    args = parser.parse_args()

    logger = setup_logging(args.log_file)
    logger.info("=" * 70)
    logger.info("Bean Lab RAGAS Evaluation Pipeline")
    logger.info(f"Model:        {args.model}  |  Judge: {args.gemini_model}")
    logger.info(f"Vector DB:    {args.vector_db}")
    logger.info("=" * 70)

    gemini_key = (
        os.environ.get("Googlegeminiapi", "").strip()
        or os.environ.get("GEMINI_API_KEY", "").strip()
    )
    if not gemini_key:
        raise SystemExit(
            "No Gemini API key found. Set Googlegeminiapi or GEMINI_API_KEY in the environment."
        )

    ragas_mods = _import_ragas_stack()

    if not check_ollama_server(args.ollama_host):
        logger.error(f"Ollama not reachable at {args.ollama_host}")
        raise SystemExit(1)

    collection = load_vector_store(args.vector_db)
    logger.info(f"✓ Collection loaded: {collection.count()} chunks")

    retriever = BeanRetriever(collection)
    retriever._load_cross_encoder()
    retriever._load_bm25()  # reports whether hybrid retrieval is available

    questions = BENCHMARK
    if args.categories:
        questions = [q for q in BENCHMARK if q["category"] in args.categories]
    if args.limit:
        questions = questions[: args.limit]
    logger.info(f"Running {len(questions)} questions")

    # ── Step 1: generate answers with the real production pipeline ──────────
    generated = []
    for i, item in enumerate(questions, 1):
        logger.info(f"[{i}/{len(questions)}] {item['id']}: {item['question'][:70]}...")
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
            contexts = [s["text"] for s in result["sources"] if s.get("text")]
            generated.append({
                **item,
                "answer": result["answer"],
                "confidence": result["confidence"],
                "contexts": contexts,
                "retrieval_channels_seen": sorted({
                    ch for s in result["sources"] for ch in s.get("retrieval_channels", [])
                }),
            })
        except Exception as e:
            logger.error(f"  Failed to generate answer: {e}")
            generated.append({**item, "answer": None, "contexts": [], "error": str(e)})

    answerable = [g for g in generated if g.get("answer") is not None and g.get("contexts")]
    logger.info(f"✓ Generated {len(answerable)}/{len(questions)} answers with non-empty context")

    if not answerable:
        raise SystemExit("No questions produced an answer with retrieved context — nothing to score.")

    # ── Step 2: set up the Gemini judge + embeddings ─────────────────────────
    judge_chat = ragas_mods["ChatGoogleGenerativeAI"](
        model=args.gemini_model, google_api_key=gemini_key, temperature=0.0,
    )
    judge_llm = ragas_mods["LangchainLLMWrapper"](judge_chat)
    judge_embeddings = ragas_mods["LangchainEmbeddingsWrapper"](
        ragas_mods["HuggingFaceEmbeddings"](model_name=args.embedding_model)
    )

    SingleTurnSample = ragas_mods["SingleTurnSample"]
    EvaluationDataset = ragas_mods["EvaluationDataset"]
    evaluate = ragas_mods["evaluate"]

    # ── Step 3: reference-free metrics — run on every answered question ─────
    free_samples = [
        SingleTurnSample(
            user_input=g["question"],
            retrieved_contexts=g["contexts"],
            response=g["answer"],
        )
        for g in answerable
    ]
    logger.info("Running reference-free RAGAS metrics (faithfulness, answer_relevancy, "
                "context_precision_no_ref)...")
    t0 = time.time()
    free_result = evaluate(
        EvaluationDataset(samples=free_samples),
        metrics=[
            ragas_mods["faithfulness"],
            ragas_mods["answer_relevancy"],
            ragas_mods["LLMContextPrecisionWithoutReference"](),
        ],
        llm=judge_llm,
        embeddings=judge_embeddings,
        raise_exceptions=False,
        show_progress=False,
    )
    logger.info(f"✓ Reference-free metrics done in {time.time() - t0:.1f}s")

    # ── Step 4: reference-needing metrics — only where ground_truth exists ──
    grounded = [g for g in answerable if g.get("ground_truth")]
    ref_scores_by_id: Dict[str, Dict] = {}
    if grounded:
        ref_samples = [
            SingleTurnSample(
                user_input=g["question"],
                retrieved_contexts=g["contexts"],
                response=g["answer"],
                reference=g["ground_truth"],
            )
            for g in grounded
        ]
        logger.info(f"Running reference-based metrics (context_precision, context_recall) "
                    f"on {len(grounded)} question(s) with ground_truth populated...")
        ref_result = evaluate(
            EvaluationDataset(samples=ref_samples),
            metrics=[ragas_mods["context_precision"], ragas_mods["context_recall"]],
            llm=judge_llm,
            embeddings=judge_embeddings,
            raise_exceptions=False,
            show_progress=False,
        )
        ref_scores_by_id = {g["id"]: s for g, s in zip(grounded, ref_result.scores)}
    else:
        logger.warning(
            "No questions have ground_truth populated in benchmark_questions.py — "
            "context_recall (and reference-based context_precision) are not available "
            "this run. See that module's docstring for why these can't be auto-filled."
        )

    # ── Step 5: merge everything per-question, save ─────────────────────────
    results = []
    for g, scores in zip(answerable, free_result.scores):
        ref_scores = ref_scores_by_id.get(g["id"], {})
        results.append({
            "id": g["id"],
            "category": g["category"],
            "question": g["question"],
            "answer": g["answer"],
            "confidence": g["confidence"],
            "expect_low_confidence": g.get("expect_low_confidence", False),
            "retrieval_channels_seen": g["retrieval_channels_seen"],
            "num_contexts": len(g["contexts"]),
            "faithfulness": scores.get("faithfulness"),
            "answer_relevancy": scores.get("answer_relevancy"),
            "context_precision_no_ref": scores.get("llm_context_precision_without_reference"),
            "context_precision": ref_scores.get("context_precision"),
            "context_recall": ref_scores.get("context_recall"),
            "had_ground_truth": g["id"] in ref_scores_by_id,
        })

    # Unanswerable / generation-failed questions still get a visible row
    answerable_ids = {g["id"] for g in answerable}
    for g in generated:
        if g["id"] not in answerable_ids:
            results.append({
                "id": g["id"], "category": g["category"], "question": g["question"],
                "answer": g.get("answer"), "error": g.get("error", "no retrieved context"),
                "confidence": g.get("confidence"),
                "expect_low_confidence": g.get("expect_low_confidence", False),
                "faithfulness": None, "answer_relevancy": None,
                "context_precision_no_ref": None, "context_precision": None,
                "context_recall": None, "had_ground_truth": False,
            })

    # Change 20: reported as two independent layers, never blended into one
    # score. RAGAS's own metric taxonomy already separates these cleanly --
    # context_precision*/context_recall judge the RETRIEVED CONTEXT alone
    # (would this evidence support a good answer, regardless of what the LLM
    # actually wrote), faithfulness/answer_relevancy judge the ANSWER against
    # that context. What was missing wasn't the metrics, it was reporting
    # them as one flat list of five numbers instead of two clearly-labeled
    # layers -- easy to eyeball as a single blob and average together in your
    # head, which is exactly the mixing this change is meant to prevent. A
    # low retrieval score and a low generation score point at different
    # fixes (reranking/retrieval tuning vs. prompting/faithfulness), so they
    # stay visually and structurally separate all the way through this
    # report. See docs/decisions.md.
    aggregate = {
        "retrieval": {
            "context_precision_no_ref": _nanmean([r["context_precision_no_ref"] for r in results]),
            "context_precision": _nanmean([r["context_precision"] for r in results]),
            "context_recall": _nanmean([r["context_recall"] for r in results]),
        },
        "generation": {
            "faithfulness": _nanmean([r["faithfulness"] for r in results]),
            "answer_relevancy": _nanmean([r["answer_relevancy"] for r in results]),
        },
    }

    by_category = {}
    for cat in sorted({r["category"] for r in results}):
        cat_rows = [r for r in results if r["category"] == cat]
        by_category[cat] = {
            "retrieval": {
                "context_precision_no_ref": _nanmean([r["context_precision_no_ref"] for r in cat_rows]),
            },
            "generation": {
                "faithfulness": _nanmean([r["faithfulness"] for r in cat_rows]),
                "answer_relevancy": _nanmean([r["answer_relevancy"] for r in cat_rows]),
            },
        }

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "judge_model": args.gemini_model,
        "embedding_model": args.embedding_model,
        "vector_db": args.vector_db,
        "top_k": args.top_k,
        "n_candidates": args.n_candidates,
        "total_questions": len(questions),
        "answered_with_context": len(answerable),
        "questions_with_ground_truth": len(grounded),
        "aggregate": aggregate,
        "by_category": by_category,
        "results": results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("=" * 70)
    logger.info(f"RAGAS evaluation complete → {args.output}")
    logger.info(f"Aggregate: {aggregate}")
    if not grounded:
        logger.info("context_recall / reference-based context_precision: N/A "
                    "(no ground_truth populated in benchmark_questions.py)")
    logger.info("=" * 70)

    print(f"\n{'=' * 70}")
    print("RAGAS EVALUATION SUMMARY")
    print(f"  Questions answered:       {len(answerable)}/{len(questions)}")
    print(f"\n  RETRIEVAL LAYER — did the correct evidence appear?")
    print(f"    Context precision*:     {aggregate['retrieval']['context_precision_no_ref']}  (*reference-free variant)")
    if grounded:
        print(f"    Context precision (ref): {aggregate['retrieval']['context_precision']}")
        print(f"    Context recall:          {aggregate['retrieval']['context_recall']}")
    else:
        print("    Context recall:          N/A — no questions have ground_truth populated yet")
    print(f"\n  GENERATION LAYER — did the answer faithfully use that evidence?")
    print(f"    Faithfulness:           {aggregate['generation']['faithfulness']}")
    print(f"    Answer relevancy:       {aggregate['generation']['answer_relevancy']}")
    print(f"\n  A low retrieval score points at retrieval/reranking; a low")
    print(f"  generation score points at prompting/faithfulness — read them")
    print(f"  separately, don't average them together.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
