"""Оцінювання RAG-системи на золотому наборі лабораторної №5."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generation import RAGGenerator
from citation_validator import CitationValidator

GOLD_SET = ROOT / "tests" / "gold-set.jsonl"
OUTPUT = ROOT / "reports" / "lab5_evaluation.json"
REFUSAL = "Недостатньо інформації у знайдених фрагментах."


def load_gold_set(path):
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def evaluate():
    tests = load_gold_set(GOLD_SET)
    rag = RAGGenerator() 
    records = []

    for index, item in enumerate(tests, start=1):
        print(f"[{index}/{len(tests)}] {item['id']}: {item['query']}", flush=True)
        try:
            result = rag.generate(item["query"], top_k=5)
            answer = result.get("answer", "").strip()
            chunks = result.get("sources", [])

            validation = CitationValidator.validate(answer, chunks)
            refusal = answer.casefold() == REFUSAL.casefold()
            citation_valid = (
                not validation["invalid_numbers"]
                and (validation["has_citations"] or refusal)
            )

            expected_doc = item.get("document_id")
            retrieval_hit = (
                any(chunk["document_id"] == expected_doc for chunk in chunks[:5])
                if expected_doc is not None else None
            )
            correct_refusal = refusal if item["question_type"] == "negative" else None
            records.append({
                "id": item["id"], "type": item["question_type"],
                "query": item["query"], "expected_answer": item["expected_answer"],
                "answer": answer, "retrieval_hit_at_5": retrieval_hit,
                "citation_valid": citation_valid, "correct_refusal": correct_refusal,
                "retrieved_document_ids": [c["document_id"] for c in chunks[:5]],
                "cited_chunk_ids": validation["cited_chunk_ids"],
            })
        except Exception as exc:
            records.append({"id": item["id"], "type": item["question_type"],
                            "query": item["query"], "error": str(exc),
                            "retrieval_hit_at_5": False if item.get("document_id") else None,
                            "citation_valid": False,
                            "correct_refusal": False if item["question_type"] == "negative" else None})

    positive = [r for r in records if r["retrieval_hit_at_5"] is not None]
    negative = [r for r in records if r["correct_refusal"] is not None]
    rate = lambda successes, total: round(100 * successes / total, 2) if total else 0.0

    metrics = {
        "total_queries": len(records),
        "document_level_recall_at_5_percent": rate(sum(r["retrieval_hit_at_5"] for r in positive), len(positive)),
        "citation_validity_percent": rate(sum(r["citation_valid"] for r in records), len(records)),
        "correct_refusal_percent": rate(sum(r["correct_refusal"] for r in negative), len(negative)),
        "positive_queries": len(positive), "negative_queries": len(negative),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as file:
        json.dump({"metrics": metrics, "results": records}, file, ensure_ascii=False, indent=2)

    print("\nРЕЗУЛЬТАТИ ОЦІНЮВАННЯ")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"Деталі збережено: {OUTPUT}")


if __name__ == "__main__":
    evaluate()
