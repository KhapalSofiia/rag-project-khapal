
import argparse
import chromadb
import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-m3"
CHROMA_PATH = "data/chroma"
COLLECTION_NAME = "movie_scripts"


def search(query, top_k=5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = SentenceTransformer(MODEL_NAME, device=device)

    query_vector = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    results = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    print(f"\nQuery: {query}")
    print(f"Top-{top_k} results:\n")

    for i, (chunk_id, distance, metadata, document) in enumerate(
        zip(
            results["ids"][0],
            results["distances"][0],
            results["metadatas"][0],
            results["documents"][0]
        ),
        start=1
    ):
        similarity = 1 - distance

        print(f"{i}. ID чанку: {chunk_id}")
        print(f"   ID документу: {metadata['document_id']}")
        print(f"   Індекс чанку: {metadata['chunk_index']}")
        print(f"   Подібність: {similarity:.4f}")
        print(f"   Текст: {document[:300]}...")
        print("-" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()
    search(args.query, args.top_k)
