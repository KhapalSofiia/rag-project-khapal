
import json
from pathlib import Path

import chromadb
import numpy as np

CHUNKS_FILE = Path("data/chunks/chunks.jsonl")
VECTORS_FILE = Path("data/embeddings/vectors.npy")
IDS_FILE = Path("data/embeddings/chunk_ids.json")
CHROMA_DIR = Path("data/chroma")

COLLECTION_NAME = "movie_scripts"
BATCH_SIZE = 100


def build_index():
    print("Loading embeddings...")

    vectors = np.load(VECTORS_FILE)

    with IDS_FILE.open("r", encoding="utf-8") as f:
        chunk_ids = json.load(f)

    chunks = {}

    with CHUNKS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            chunks[record["chunk_id"]] = record

    assert len(vectors) == len(chunk_ids), "Vectors/IDs count mismatch"
    assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunk IDs"
    assert all(cid in chunks for cid in chunk_ids), "Missing chunks"

    print(f"Total vectors: {len(vectors)}")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}}
    )

    total = len(chunk_ids)

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)

        batch_ids = chunk_ids[start:end]
        batch_vectors = vectors[start:end].tolist()

        batch_texts = []
        batch_metadata = []

        for chunk_id in batch_ids:
            chunk = chunks[chunk_id]

            batch_texts.append(chunk["text"])

            batch_metadata.append({
                "document_id": str(chunk["document_id"]),
                "chunk_index": int(chunk["chunk_index"])
            })

        collection.upsert(
            ids=batch_ids,
            embeddings=batch_vectors,
            documents=batch_texts,
            metadatas=batch_metadata
        )

        print(f"Індексовано: {end}/{total}", flush=True)

    print("\nІндекс створено.")
    print(f"Усього поундексовано чанків: {collection.count()}")

if __name__ == "__main__":
    build_index()
