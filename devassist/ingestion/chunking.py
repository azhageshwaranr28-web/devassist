"""Token-aware document chunking with stable metadata."""
from __future__ import annotations

from typing import Any, Iterable


def _token_len(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _split_long_text(tokenizer: Any, text: str, limit: int) -> list[str]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return [tokenizer.decode(ids[start : start + limit]).strip() for start in range(0, len(ids), limit)]


def _units(text: str) -> Iterable[str]:
    """Yield paragraphs/code blocks. Code blocks stay intact unless over model limit."""
    current: list[str] = []
    in_code = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            current.append(line)
            in_code = not in_code
            if not in_code:
                yield "\n".join(current).strip()
                current = []
            continue
        if not in_code and not line.strip():
            if current:
                yield "\n".join(current).strip()
                current = []
        else:
            current.append(line)
    if current:
        yield "\n".join(current).strip()


def chunk_document(
    document: dict[str, Any], tokenizer: Any, chunk_size: int = 220, overlap: int = 30
) -> list[dict[str, Any]]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size")
    title = str(document["title"])
    prefix = f"Title: {title}\n"
    available = chunk_size - _token_len(tokenizer, prefix)
    if available < 32:
        raise ValueError("Title leaves too little room for content; raise CHUNK_SIZE")

    parts: list[str] = []
    for unit in _units(str(document["content"])):
        if _token_len(tokenizer, unit) <= available:
            parts.append(unit)
        else:
            parts.extend(_split_long_text(tokenizer, unit, available))

    chunks: list[str] = []
    current = ""
    for part in parts:
        candidate = f"{current}\n\n{part}".strip() if current else part
        if _token_len(tokenizer, candidate) <= available:
            current = candidate
            continue
        if current:
            chunks.append((prefix + current).strip())
            overlap_ids = tokenizer.encode(current, add_special_tokens=False)[-overlap:] if overlap else []
            overlap_text = tokenizer.decode(overlap_ids).strip() if overlap_ids else ""
            current = f"{overlap_text}\n\n{part}".strip() if overlap_text else part
        else:
            current = part
        if _token_len(tokenizer, current) > available:
            forced = _split_long_text(tokenizer, current, available)
            chunks.extend((prefix + x).strip() for x in forced[:-1])
            current = forced[-1]
    if current:
        chunks.append((prefix + current).strip())

    answer_id = int(document["metadata"]["answer_id"])
    output = []
    for index, text in enumerate(chunks):
        metadata = dict(document["metadata"])
        metadata.update({"chunk_index": index, "title": title})
        output.append(
            {
                "chunk_id": answer_id * 1000 + index,
                "text": text,
                "metadata": metadata,
                "token_count": _token_len(tokenizer, text),
            }
        )
    return output
