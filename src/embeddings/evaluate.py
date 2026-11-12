
import json
from pathlib import Path

import chromadb
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-m3"
EVAL_FILE = Path("tests/eval-set.jsonl")
CHROMA_PATH = "data/chroma"
COLLECTION_NAME = "movie_scripts"

CANDIDATE_COUNTS = [10, 50, 100, 200]


def load_eval_set():
    with EVAL_FILE.open("r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    if not data:
        raise ValueError("Оцінювальний набір порожній")

    return data


def calculate_recall(results, relevant_ids, k):
    hits = sum(
        relevant_id in retrieved[:k]
        for retrieved, relevant_id in zip(results, relevant_ids)
    )
    return hits / len(relevant_ids)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Пристрій для обчислень: {device}")

    print("Завантаження моделі...")
    model = SentenceTransformer(MODEL_NAME, device=device)

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    eval_set = load_eval_set()
    queries = [item["query"] for item in eval_set]
    relevant_ids = [item["relevant_chunk_id"] for item in eval_set]

    print(f"Кількість тестових запитів: {len(queries)}")
    print(f"Кількість проіндексованих чанків: {collection.count()}")

    # Перевірка наявності еталонних чанків
    existing = set(collection.get(ids=list(set(relevant_ids)))["ids"])
    missing = set(relevant_ids) - existing

    if missing:
        raise ValueError(f"Відсутні еталонні чанки: {missing}")

    print("\nВекторизація тестових запитів...")

    query_vectors = model.encode(
        queries,
        batch_size=8,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    print("\n=== БАЗОВИЙ ПОШУК: ChromaDB ===")

    baseline = collection.query(
        query_embeddings=query_vectors.tolist(),
        n_results=10,
        include=["distances"]
    )

    baseline_results = baseline["ids"]

    recall5 = calculate_recall(
        baseline_results, relevant_ids, 5
    )
    recall10 = calculate_recall(
        baseline_results, relevant_ids, 10
    )

    print(f"Recall@5:  {recall5:.4f} ({recall5:.2%})")
    print(f"Recall@10: {recall10:.4f} ({recall10:.2%})")

    best_config = {
        "name": "Базовий пошук",
        "recall5": recall5,
        "recall10": recall10
    }

    for candidate_count in CANDIDATE_COUNTS:
        print(
            f"\n=== ПОВТОРНЕ РАНЖУВАННЯ: "
            f"{candidate_count} кандидатів ==="
        )

        search_results = collection.query(
            query_embeddings=query_vectors.tolist(),
            n_results=candidate_count,
            include=["embeddings"]
        )

        reranked_results = []

        for query_vector, ids, embeddings in zip(
            query_vectors,
            search_results["ids"],
            search_results["embeddings"]
        ):
            candidate_vectors = np.asarray(
                embeddings, dtype=np.float32
            )

            norms = np.linalg.norm(
                candidate_vectors, axis=1, keepdims=True
            )
            candidate_vectors = candidate_vectors / np.maximum(
                norms, 1e-12
            )

            similarities = candidate_vectors @ query_vector
            sorted_indices = np.argsort(-similarities)

            reranked_results.append([
                ids[i] for i in sorted_indices[:10]
            ])

        r5 = calculate_recall(
            reranked_results, relevant_ids, 5
        )
        r10 = calculate_recall(
            reranked_results, relevant_ids, 10
        )

        print(f"Recall@5:  {r5:.4f} ({r5:.2%})")
        print(f"Recall@10: {r10:.4f} ({r10:.2%})")

        if (r10, r5) > (
            best_config["recall10"],
            best_config["recall5"]
        ):
            best_config = {
                "name": f"Повторне ранжування ({candidate_count} кандидатів)",
                "recall5": r5,
                "recall10": r10
            }

    print("\n" + "=" * 50)
    print("НАЙКРАЩА КОНФІГУРАЦІЯ ПОШУКУ")
    print("=" * 50)

    print(f"Метод: {best_config['name']}")
    print(
        f"Recall@5:  {best_config['recall5']:.4f} "
        f"({best_config['recall5']:.2%})"
    )
    print(
        f"Recall@10: {best_config['recall10']:.4f} "
        f"({best_config['recall10']:.2%})"
    )

    print("\nОцінювання завершено.")


if __name__ == "__main__":
    main()
