
import chromadb
import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-m3"
CHROMA_PATH = "data/chroma"
COLLECTION_NAME = "movie_scripts"


class DenseRetriever:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Пристрій для обчислень: {device}")
        print("Завантаження моделі BGE-M3...")

        self.model = SentenceTransformer(
            MODEL_NAME,
            device=device
        )

        self.client = chromadb.PersistentClient(
            path=CHROMA_PATH
        )

        self.collection = self.client.get_collection(
            name=COLLECTION_NAME
        )

        print(
            f"Кількість чанків в індексі: "
            f"{self.collection.count()}"
        )

    def search(self, query: str, top_k: int = 5):
        if not query.strip():
            raise ValueError("Пошуковий запит не може бути порожнім")

        if top_k < 1:
            raise ValueError("top_k має бути більшим за нуль")

        query_vector = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        results = self.collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        chunks = []

        for chunk_id, document, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            chunks.append({
                "chunk_id": chunk_id,
                "document_id": metadata["document_id"],
                "chunk_index": metadata["chunk_index"],
                "text": document,
                "similarity": float(1 - distance)
            })

        return chunks


if __name__ == "__main__":
    retriever = DenseRetriever()

    query = input("\nВведіть пошуковий запит: ")
    results = retriever.search(query, top_k=5)

    print("\n=== РЕЗУЛЬТАТИ ЩІЛЬНОГО ПОШУКУ ===")

    for i, chunk in enumerate(results, start=1):
        print(f"\n{i}. ID чанка: {chunk['chunk_id']}")
        print(f"   ID документа: {chunk['document_id']}")
        print(f"   Індекс чанка: {chunk['chunk_index']}")
        print(f"   Подібність: {chunk['similarity']:.4f}")
        print(f"   Текст: {chunk['text'][:300]}...")
