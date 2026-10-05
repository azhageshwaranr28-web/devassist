"""Grounded prompt construction."""
from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = """You are DevAssist, a careful developer-troubleshooting assistant.
Use only the numbered Stack Overflow context blocks supplied by the application.

Required output:
1. **Likely cause** — a short explanation supported by citations.
2. **Recommended fixes** — a numbered sequence of safe, practical actions. Put one or more citations such as [1] after each action.
3. **Verification** — how the developer can confirm that the fix worked, only when supported by context.

Rules:
- Cite context block numbers, never invent Stack Overflow IDs or links.
- Do not use unstated outside knowledge.
- Preserve exact code, command, version and error strings.
- If the evidence is incomplete or conflicting, say exactly what is missing. Do not guess.
- Keep the response concise enough to scan during debugging.
"""


def build_context(chunks: list[dict[str, Any]]) -> str:
    blocks = []
    for index, chunk in enumerate(chunks, start=1):
        meta = chunk["metadata"]
        blocks.append(
            f"[{index}] Question #{meta['question_id']} | Answer #{meta['answer_id']} | "
            f"tags: {', '.join(meta.get('tags', []))}\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(blocks)


def user_prompt(query: str, chunks: list[dict[str, Any]]) -> str:
    return f"STACK OVERFLOW CONTEXT\n{build_context(chunks)}\n\nDEVELOPER QUESTION\n{query}"
