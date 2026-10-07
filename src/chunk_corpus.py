
from pathlib import Path
import hashlib
import json


INPUT_DIR = Path("data/cleaned_scripts")
OUTPUT_FILE = Path("data/chunks/chunks.jsonl")

CHUNK_SIZE = 300
OVERLAP = 45


def generate_chunk_id(document_id: str, chunk_index: int) -> str:
    """Generate stable global chunk ID."""

    value = f"{document_id}:{chunk_index}"

    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()[:16]


def chunk_text(text: str):
    """Split text into fixed-size chunks with overlap."""

    words = text.split()

    chunks = []

    start = 0
    chunk_index = 0

    step = CHUNK_SIZE - OVERLAP

    while start < len(words):

        end = min(
            start + CHUNK_SIZE,
            len(words)
        )

        chunk_words = words[start:end]

        chunks.append(
            (chunk_index, chunk_words)
        )

        if end == len(words):
            break

        start += step
        chunk_index += 1

    return chunks


def process_corpus():

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    total_chunks = 0

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as output:

        for input_file in sorted(
            INPUT_DIR.glob("*.txt")
        ):

            document_id = input_file.stem

            text = input_file.read_text(
                encoding="utf-8"
            )

            chunks = chunk_text(text)

            for chunk_index, words in chunks:

                chunk_id = generate_chunk_id(
                    document_id,
                    chunk_index
                )

                record = {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "chunk_index": chunk_index,
                    "text": " ".join(words)
                }

                output.write(
                    json.dumps(
                        record,
                        ensure_ascii=False
                    ) + "\n"
                )

                total_chunks += 1

    print("Chunking finished.")
    print(f"Total chunks: {total_chunks}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    process_corpus()

