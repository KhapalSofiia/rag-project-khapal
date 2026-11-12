
import re


class CitationValidator:
    @staticmethod
    def validate(answer: str, chunks: list) -> dict:
        citations = re.findall(r"\[(\d+)\]", answer)
        cited_numbers = sorted(set(int(n) for n in citations))

        valid_numbers = set(range(1, len(chunks) + 1))

        invalid_numbers = [
            n for n in cited_numbers
            if n not in valid_numbers
        ]

        has_citations = len(cited_numbers) > 0

        refusal = (
            "недостатньо інформації" in answer.lower()
            or "недостатньо контексту" in answer.lower()
        )

        is_valid = not invalid_numbers

        cited_chunk_ids = [
            chunks[n - 1]["chunk_id"]
            for n in cited_numbers
            if n in valid_numbers
        ]

        return {
            "is_valid": is_valid,
            "cited_numbers": cited_numbers,
            "cited_chunk_ids": cited_chunk_ids,
            "invalid_numbers": invalid_numbers,
            "has_citations": has_citations,
            "is_refusal": refusal
        }

