"""Create token-aware chunks, embeddings and a persistent local Qdrant index."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from config import settings
from ingestion.chunking import chunk_document


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    if not settings.documents_file.exists():
        raise SystemExit("Processed documents not found. Run: python -m ingestion.preprocess")

    model = SentenceTransformer(settings.embedding_model)
    max_input = int(model.max_seq_length)
    if settings.chunk_size > max_input:
        raise SystemExit(
            f"CHUNK_SIZE={settings.chunk_size} exceeds {settings.embedding_model} limit {max_input}."
        )

    documents = read_jsonl(settings.documents_file)
    chunks = [
        chunk
        for document in documents
        for chunk in chunk_document(document, model.tokenizer, settings.chunk_size, settings.chunk_overlap)
    ]
    write_jsonl(settings.chunks_file, chunks)

    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    qdrant_path = str((settings.root / settings.qdrant_path).resolve()) if not Path(settings.qdrant_path).is_absolute() else settings.qdrant_path
    client = QdrantClient(path=qdrant_path)
    if client.collection_exists(settings.qdrant_collection):
        client.delete_collection(settings.qdrant_collection)
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=models.VectorParams(size=int(embeddings.shape[1]), distance=models.Distance.COSINE),
    )

    batch_size = 128
    for start in range(0, len(chunks), batch_size):
        points = []
        for offset, chunk in enumerate(chunks[start : start + batch_size]):
            payload = {"text": chunk["text"], **chunk["metadata"]}
            points.append(
                models.PointStruct(
                    id=int(chunk["chunk_id"]),
                    vector=embeddings[start + offset].tolist(),
                    payload=payload,
                )
            )
        client.upsert(settings.qdrant_collection, points=points, wait=True)

    stats = {
        "documents": len(documents),
        "chunks": len(chunks),
        "embedding_model": settings.embedding_model,
        "vector_size": int(embeddings.shape[1]),
        "model_max_tokens": max_input,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "average_chunk_tokens": round(mean(c["token_count"] for c in chunks), 2),
        "maximum_chunk_tokens": max(c["token_count"] for c in chunks),
    }
    stats_path = settings.root / "data/processed/index_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
