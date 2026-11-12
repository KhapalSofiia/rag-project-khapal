
from search_alg.dense_retrieval import DenseRetriever
from search_alg.sparse_retrieval import SparseRetriever

TEST_QUERIES = [
    "A boy named Georgie meets a mysterious clown in a storm drain",
    "In which film does the imprisoned wizard Grindelwald sit in a magically protected cell while officials discuss his transfer to Europe?",
    "Find a scene where Pennywise talks to Georgie in the sewer",
    "A boy loses his paper boat in the rain and meets a clown",
    "A powerful dark wizard is held prisoner and guarded by magic"
]


def print_results(results, method):
    print(f"\n--- {method} ---")

    for i, result in enumerate(results, start=1):
        score = result.get(
            "similarity",
            result.get("score", 0)
        )

        print(
            f"{i}. Документ: {result['document_id']} | "
            f"Чанк: {result['chunk_id']} | "
            f"Оцінка: {score:.4f}"
        )


def main():
    print("Ініціалізація пошукових алгоритмів...")

    dense = DenseRetriever()
    sparse = SparseRetriever()

    for i, query in enumerate(TEST_QUERIES, start=1):
        print("\n" + "=" * 70)
        print(f"ЗАПИТ {i}: {query}")
        print("=" * 70)

        dense_results = dense.search(query, top_k=5)
        sparse_results = sparse.search(query, top_k=5)

        print_results(dense_results, "ЩІЛЬНИЙ ПОШУК")
        print_results(sparse_results, "BM25")


if __name__ == "__main__":
    main()
