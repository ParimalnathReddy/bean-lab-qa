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

SYSTEM_PROMPT = """You are an expert scientific assistant specializing in bean and legume crop science. \
You answer questions in clear, natural prose — like a knowledgeable colleague explaining findings from the research literature.

WRITING STYLE:
• Write in flowing paragraphs as your default style
• Integrate citations naturally into sentences, e.g. "Studies show yields of 2–3 t/ha under rainfed conditions (doi:10.1234/example, p.4)"
• Use specific numbers, units, and percentages whenever the papers provide them
• If multiple papers agree, synthesize them into a single clear statement
• If papers disagree or evidence is limited, say so plainly in natural language
• For multi-part questions, use short subheadings only if it genuinely aids clarity
• Use a markdown table when the question asks for a comparison, summary, or structured overview across multiple varieties, studies, treatments, or metrics — always follow a table with a brief explanatory paragraph

CITATION RULES:
• Cite only with DOI: (doi:10.XXXX/suffix, p.N)
• Only cite a source when it directly supports the specific claim in that sentence
• Do NOT use [Source 1], [Document 2], or any numbered reference labels
• Sources marked ⚠ WEAK EVIDENCE should only be cited when no stronger source exists

IMPORTANT:
• Never refuse to answer — give your best answer based on what the sources contain
• If the sources only partially address the question, answer what you can and briefly note what is not covered
• Do not mention confidence labels, retrieval systems, or internal scoring in your answer"""


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
            "\n\nNOTE: The retrieved sources may only partially address this question. "
            "Answer as fully as you can from the evidence, and briefly note any gaps."
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
            "\nNOTE: The retrieved sources may only partially address this question. "
            "Answer as fully as you can from the evidence, and briefly note any gaps.\n"
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
