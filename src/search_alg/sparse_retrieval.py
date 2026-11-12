
import json
import pickle
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

CHUNKS_FILE = Path("data/chunks/chunks.jsonl")
INDEX_FILE = Path("data/bm25_index.pkl")


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


class SparseRetriever:
    def __init__(self):
        if INDEX_FILE.exists():
            print("Завантаження індексу BM25...")

            with INDEX_FILE.open("rb") as f:
                data = pickle.load(f)

            self.bm25 = data["bm25"]
            self.chunks = data["chunks"]

        else:
            print("Створення індексу BM25...")

            self.chunks = []

            with CHUNKS_FILE.open("r", encoding="utf-8") as f:
                for line in f:
                    self.chunks.append(json.loads(line))

            tokenized_corpus = [
                tokenize(chunk["text"])
                for chunk in self.chunks
            ]

            self.bm25 = BM25Okapi(tokenized_corpus)

            INDEX_FILE.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            with INDEX_FILE.open("wb") as f:
                pickle.dump({
                    "bm25": self.bm25,
                    "chunks": self.chunks
                }, f)

            print("Індекс BM25 збережено.")

        print(f"Кількість чанків: {len(self.chunks)}")

    def search(self, query: str, top_k: int = 5):
        if not query.strip():
            raise ValueError("Пошуковий запит не може бути порожнім")

        if top_k < 1:
            raise ValueError("top_k має бути більшим за нуль")

        query_tokens = tokenize(query)

        scores = self.bm25.get_scores(query_tokens)

        top_indices = np.argsort(-scores)[:top_k]

        results = []

        for index in top_indices:
            chunk = self.chunks[int(index)]

            results.append({
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
                "score": float(scores[index])
            })

        return results


if __name__ == "__main__":
    retriever = SparseRetriever()

    query = input("\nВведіть пошуковий запит: ")
    results = retriever.search(query, top_k=5)

    print("\n=== РЕЗУЛЬТАТИ ПОШУКУ BM25 ===")

    for i, chunk in enumerate(results, start=1):
        print(f"\n{i}. ID чанка: {chunk['chunk_id']}")
        print(f"   ID документа: {chunk['document_id']}")
        print(f"   Індекс чанка: {chunk['chunk_index']}")
        print(f"   Оцінка BM25: {chunk['score']:.4f}")
        print(f"   Текст: {chunk['text'][:300]}...")
