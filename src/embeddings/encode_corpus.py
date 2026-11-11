
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

INPUT_FILE = Path("data/chunks/chunks.jsonl")
OUTPUT_DIR = Path("data/embeddings")

MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 16


def encode_corpus():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME, device="cuda")
    model.max_seq_length = 512

    print(f"Model device: {model.device}")

    chunk_ids = []
    embeddings = []

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        batch_texts = []
        batch_ids = []

        def process_batch():
            vectors = model.encode(
                batch_texts,
                batch_size=BATCH_SIZE,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False
            )

            embeddings.append(vectors)
            chunk_ids.extend(batch_ids)
            print(f"Оброблено чанків: {len(chunk_ids)} / 92175", flush=True)

        for line in file:
            record = json.loads(line)

            batch_texts.append(record["text"])
            batch_ids.append(record["chunk_id"])

            if len(batch_texts) == BATCH_SIZE:
                process_batch()
                batch_texts.clear()
                batch_ids.clear()

        if batch_texts:
            process_batch()

    vectors = np.concatenate(embeddings, axis=0)

    np.save(OUTPUT_DIR / "vectors.npy", vectors)

    with (OUTPUT_DIR / "chunk_ids.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(chunk_ids, file)

    print("Encoding finished.")
    print(f"Total chunks: {len(chunk_ids)}")
    print(f"Embedding dimension: {vectors.shape[1]}")
    print(f"Vectors shape: {vectors.shape}")


if __name__ == "__main__":
    encode_corpus()
