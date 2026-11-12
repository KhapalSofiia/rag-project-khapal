
from citation_validator import CitationValidator

chunks = [
    {"chunk_id": "chunk_001"},
    {"chunk_id": "chunk_002"},
    {"chunk_id": "chunk_003"},
    {"chunk_id": "chunk_004"},
    {"chunk_id": "chunk_005"}
]

tests = [
    ("Це фільм «It» [2].", True),
    ("Відповідь підтверджується [1] та [3].", True),
    ("Це фільм «It» [7].", False),
    ("Це фільм «It».", False),
    ("Недостатньо інформації у знайдених фрагментах.", True)
]

for i, (answer, expected) in enumerate(tests, start=1):
    result = CitationValidator.validate(answer, chunks)

    passed = result["is_valid"] == expected

    print(f"\nТест {i}: {'ПРОЙДЕНО' if passed else 'ПОМИЛКА'}")
    print(f"Відповідь: {answer}")
    print(f"Цитування: {result['cited_numbers']}")
    print(f"ID чанків: {result['cited_chunk_ids']}")
    print(f"Коректність: {result['is_valid']}")
