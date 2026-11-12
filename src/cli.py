
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

from generation import RAGGenerator
from citation_validator import CitationValidator


def run_rag(query: str, top_k: int = 5) -> dict:
    rag = RAGGenerator()

    result = rag.generate(query, top_k=top_k)

    answer = result.get("answer", "")
    chunks = result.get("sources", [])

    validation = CitationValidator.validate(answer, chunks)

    sources = []
    for index, chunk in enumerate(chunks, start=1):
        sources.append({
            "number": index,
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk["document_id"],
            "chunk_index": chunk.get("chunk_index"),
            "cited": index in validation["cited_numbers"]
        })

    citation_valid = (
        result.get("citation_valid", validation["is_valid"])
    )

    return {
        "query": query,
        "answer": answer,
        "citation_valid": citation_valid,
        "cited_chunk_ids": result.get(
            "cited_chunk_ids",
            validation["cited_chunk_ids"]
        ),
        "sources": sources,
        "model": "llama3.2:3b",
        "retrieval_method": "hybrid_rrf",
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat()
    }


def main():
    parser = argparse.ArgumentParser(
        description="CLI для RAG-пошуку по сценаріях фільмів"
    )

    parser.add_argument(
        "--query", "-q",
        type=str,
        required=True,
        help="Запит користувача"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Кількість знайдених чанків"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Шлях для збереження JSON-файлу"
    )

    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k має бути більше нуля")

    try:
        result = run_rag(args.query, args.top_k)

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            with open(
                output_path, "w", encoding="utf-8"
            ) as file:
                json.dump(
                    result,
                    file,
                    ensure_ascii=False,
                    indent=4
                )

            print(
                f"\nРезультат збережено: {output_path}"
            )

        else:
            print("\n" + "=" * 60)
            print("ВІДПОВІДЬ RAG-СИСТЕМИ")
            print("=" * 60)

            print(f"\nЗапит: {result['query']}")
            print(f"\nВідповідь: {result['answer']}")

            print(
                f"\nКоректність цитувань: "
                f"{result['citation_valid']}"
            )

            print("\nЗНАЙДЕНІ ДЖЕРЕЛА:")

            for source in result["sources"]:
                print(
                    f"[{source['number']}] "
                    f"{source['document_id']} | "
                    f"{source['chunk_id']}"
                )

    except Exception as error:
        print(f"Помилка: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
