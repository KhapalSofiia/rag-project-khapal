
from search_alg.dense_retrieval import DenseRetriever
from search_alg.sparse_retrieval import SparseRetriever


class HybridRetriever:
    def __init__(self, rrf_k=60, candidate_count=100):
        self.dense = DenseRetriever()
        self.sparse = SparseRetriever()

        self.rrf_k = rrf_k
        self.candidate_count = candidate_count

    def search(self, query: str, top_k: int = 5):
        if not query.strip():
            raise ValueError("Пошуковий запит не може бути порожнім")

        if top_k < 1:
            raise ValueError("top_k має бути більшим за нуль")

        candidate_count = max(self.candidate_count, top_k)

        dense_results = self.dense.search(
            query, top_k=candidate_count
        )

        sparse_results = self.sparse.search(
            query, top_k=candidate_count
        )

        combined = {}

        for method, results in [
            ("dense", dense_results),
            ("bm25", sparse_results)
        ]:
            for rank, chunk in enumerate(results, start=1):
                chunk_id = chunk["chunk_id"]

                if chunk_id not in combined:
                    combined[chunk_id] = {
                        "chunk_id": chunk_id,
                        "document_id": chunk["document_id"],
                        "chunk_index": chunk["chunk_index"],
                        "text": chunk["text"],
                        "rrf_score": 0.0,
                        "dense_rank": None,
                        "bm25_rank": None
                    }

                combined[chunk_id]["rrf_score"] += (
                    1.0 / (self.rrf_k + rank)
                )

                combined[chunk_id][f"{method}_rank"] = rank

        ranked = sorted(
            combined.values(),
            key=lambda item: (
                -item["rrf_score"],
                item["chunk_id"]
            )
        )

        return ranked[:top_k]


if __name__ == "__main__":
    retriever = HybridRetriever(
        rrf_k=60,
        candidate_count=100
    )

    query = input("\nВведіть пошуковий запит: ")

    results = retriever.search(query, top_k=5)

    print("\n=== РЕЗУЛЬТАТИ ГІБРИДНОГО ПОШУКУ ===")

    for i, chunk in enumerate(results, start=1):
        print(f"\n{i}. ID чанка: {chunk['chunk_id']}")
        print(f"   ID документа: {chunk['document_id']}")
        print(f"   Індекс чанка: {chunk['chunk_index']}")
        print(f"   Оцінка RRF: {chunk['rrf_score']:.6f}")
        print(f"   Позиція Dense: {chunk['dense_rank']}")
        print(f"   Позиція BM25: {chunk['bm25_rank']}")
        print(f"   Текст: {chunk['text'][:300]}...")
