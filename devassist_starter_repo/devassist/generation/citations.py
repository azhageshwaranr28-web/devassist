"""Build verifiable source links from retrieved metadata, never from model text."""
from __future__ import annotations

import re
from typing import Any

CITATION_RE = re.compile(r"\[(\d+)\]")


def cited_context_numbers(answer: str, context_count: int) -> list[int]:
    values = []
    for match in CITATION_RE.findall(answer):
        number = int(match)
        if 1 <= number <= context_count and number not in values:
            values.append(number)
    return values


def sources_from_answer(answer: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    numbers = cited_context_numbers(answer, len(chunks))
    # A model that omits citations does not get to invent IDs. Use the actual supplied context.
    selected = [chunks[i - 1] for i in numbers] if numbers else chunks
    sources: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for chunk in selected:
        meta = chunk["metadata"]
        key = (int(meta["question_id"]), int(meta["answer_id"]))
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "question_id": key[0],
                "answer_id": key[1],
                "title": meta.get("title", "Stack Overflow question"),
                "question_url": meta.get("question_url", f"https://stackoverflow.com/questions/{key[0]}"),
                "answer_url": meta.get("answer_url", f"https://stackoverflow.com/a/{key[1]}"),
                "tags": meta.get("tags", []),
            }
        )
    return sources
