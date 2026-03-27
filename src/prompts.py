#!/usr/bin/env python3
"""
Shared prompt templates for the Bean Lab RAG system.

Design principles:
  1. Evidence-first chain of thought: extract → synthesize → conclude
  2. Explicit confidence labels: SUPPORTED / PARTIALLY_SUPPORTED / INFERRED / UNSUPPORTED
  3. Soft failure: always attempt a best-effort answer; label uncertainty explicitly
  4. Section-aware: chunks tagged with [METHODS] / [RESULTS] / [DISCUSSION] etc.
  5. Multi-part questions: decompose into sub-questions before answering
  6. Citation format: doi:10.XXXX/suffix (no generic [Source N] labels)
"""

from typing import List, Dict

DISTANCE_THRESHOLD = 0.85   # chunks above this are included but flagged as weak


# ── Section tag mapping ───────────────────────────────────────────────────────

SECTION_TAG_MAP = {
    "abstract":     "[ABSTRACT]",
    "introduction": "[INTRO]",
    "background":   "[INTRO]",
    "methods":      "[METHODS]",
    "materials":    "[METHODS]",
    "results":      "[RESULTS]",
    "discussion":   "[DISCUSSION]",
    "conclusion":   "[CONCLUSION]",
    "references":   "[REFERENCES]",
    "table":        "[TABLE]",
    "figure":       "[FIGURE]",
}


def get_section_tag(section: str) -> str:
    if not section:
        return ""
    s = section.lower()
    for key, tag in SECTION_TAG_MAP.items():
        if key in s:
            return tag
    return ""


# ── Context builder ───────────────────────────────────────────────────────────

def build_context(chunks: List[Dict]) -> str:
    """
    Format retrieved chunks into a structured context block.

    Each chunk is labeled with:
    - DOI and page
    - Section tag if available
    - Weak-evidence warning if distance > threshold
    """
    lines = []
    for i, c in enumerate(chunks, 1):
        dist = c.get("distance", 1.0)
        section_tag = get_section_tag(c.get("section", ""))
        quality = "⚠ WEAK EVIDENCE" if dist > DISTANCE_THRESHOLD else ""

        header_parts = [f"doi:{c['doi']}", f"p.{c['page']}"]
        if section_tag:
            header_parts.append(section_tag)
        if quality:
            header_parts.append(quality)

        lines.append(f"[{i}] {' | '.join(header_parts)}")
        lines.append(c["text"])
        lines.append("")
    return "\n".join(lines)


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a rigorous scientific research assistant specializing in bean and legume crop science.

You answer questions by:
1. EXTRACTING evidence from each provided source
2. SYNTHESIZING across sources to identify agreements, contradictions, and gaps
3. STATING your answer with an explicit confidence label

CONFIDENCE LABELS — you MUST include exactly one:
• [SUPPORTED]           — answer is directly stated in the sources
• [PARTIALLY_SUPPORTED] — some aspects supported, others inferred or missing
• [INFERRED]            — answer follows logically from sources but is not stated directly
• [UNSUPPORTED]         — sources do not contain relevant information; answer draws on general knowledge

CITATION RULES:
• Cite using the format: (doi:10.XXXX/suffix, p.N)
• Only cite sources that directly support the specific claim you are making
• Do NOT use generic labels like [Source 1] or [Document 2]
• Sources marked ⚠ WEAK EVIDENCE may be cited only if no stronger source exists, and must be flagged with (weak evidence)

ANSWER FORMAT:
**Direct Answer:** [one-sentence answer with confidence label]

**Evidence:**
[specific facts, numbers, units, and percentages extracted from sources, each cited by DOI]

**Reasoning:**
[how you connected the evidence to reach the answer; note any assumptions or inferences]

**Uncertainty:**
[gaps in evidence, contradictions between sources, or aspects that cannot be answered from the provided context]

IMPORTANT:
• Always include specific numbers and units when the papers provide them
• Compare values across sources when multiple sources report the same metric
• Never refuse to answer entirely — always provide the best-effort answer and label its confidence
• If a question has multiple parts, answer each part separately under its own sub-heading"""


# ── Prompt builders ───────────────────────────────────────────────────────────

def build_messages(question: str, chunks: List[Dict], confidence: str = "") -> List[Dict]:
    """
    Build OpenAI-style messages list for chat completion API (HF Spaces / any chat LLM).

    Args:
        question:   User's question
        chunks:     Retrieved and reranked chunks
        confidence: Retrieval confidence label from BeanRetriever
    """
    context = build_context(chunks)

    confidence_note = ""
    if confidence in ("INFERRED", "UNSUPPORTED"):
        confidence_note = (
            f"\n\nNOTE: Retrieval confidence is {confidence}. "
            "The sources may not directly address this question. "
            "Provide your best-effort answer and clearly label it."
        )

    user_content = (
        f"RETRIEVED SOURCES:\n\n{context}\n"
        f"{confidence_note}\n"
        f"QUESTION: {question}"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_ollama_prompt(question: str, chunks: List[Dict], confidence: str = "") -> str:
    """
    Build a single prompt string for Ollama (non-chat models or llama3 instruct format).

    Args:
        question:   User's question
        chunks:     Retrieved and reranked chunks
        confidence: Retrieval confidence label from BeanRetriever
    """
    context = build_context(chunks)

    confidence_note = ""
    if confidence in ("INFERRED", "UNSUPPORTED"):
        confidence_note = (
            f"\nNOTE: Retrieval confidence is {confidence}. "
            "Provide best-effort answer with clear confidence labeling.\n"
        )

    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"RETRIEVED SOURCES:\n\n{context}\n"
        f"{confidence_note}"
        f"QUESTION: {question}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )


def format_references(chunks: List[Dict]) -> str:
    """
    Build a clean deduplicated reference list from retrieved chunks.
    Only includes chunks below the distance threshold.
    """
    seen = set()
    refs = []
    for c in chunks:
        doi = c.get("doi", "")
        dist = c.get("distance", 1.0)
        if doi and doi not in seen:
            seen.add(doi)
            flag = " *(weak evidence)*" if dist > DISTANCE_THRESHOLD else ""
            refs.append(f"• doi:{doi}{flag}")
    if not refs:
        return ""
    return "**References:**\n" + "\n".join(refs)
