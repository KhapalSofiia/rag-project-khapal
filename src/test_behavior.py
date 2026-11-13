import json
import re
from pathlib import Path

from generation import RAGGenerator

REPORT_PATH = Path('reports/lab5_behavior_results.json')


def normalize(text: str) -> str:
    """Нормалізація тексту для пошуку ключових слів."""
    return re.sub(r'\s+', ' ', text.casefold().replace('ё', 'е')).strip()


def contains_any(text: str, variants: list[str]) -> bool:
    normalized = normalize(text)
    return any(normalize(word) in normalized for word in variants)

TESTS = [
    {
        'id': 'MF1', 'type': 'Minimum Functionality',
        'query': 'In the movie It, what is the name of the clown who meets Georgie?',
        'expected_terms': ['пеннівайз', 'pennywise'],
    },
    {
        'id': 'MF2', 'type': 'Minimum Functionality',
        'query': 'In Shrek, what is the name of the princess Shrek rescues?',
        'expected_terms': ['фіон', 'fiona'],
    },
    {
        'id': 'INV1', 'type': 'Invariance',
        'query': 'In It, which clown speaks to Georgie near the storm drain?',
        'paraphrase': 'What is the name of the clown Georgie encounters at the drain in It?',
        'expected_terms': ['пеннівайз', 'pennywise'],
    },
    {
        'id': 'INV2', 'type': 'Invariance',
        'query': 'Who is the princess Shrek is sent to rescue?',
        'paraphrase': 'What is the name of the princess rescued by Shrek?',
        'expected_terms': ['фіон', 'fiona'],
    },
    {
        'id': 'DE1', 'type': 'Directional Expectation',
        'query': 'In Shrek, who is the princess rescued by Shrek?',
        'changed_query': 'In Shrek, who is the lord who sends Shrek to rescue the princess?',
        'expected_terms': ['фіон', 'fiona'],
        'changed_expected_terms': ['фаркуад', 'фарквад', 'farquaad'],
    },
    {
        'id': 'DE2', 'type': 'Directional Expectation',
        'query': 'In It, who is the clown that meets Georgie?',
        'changed_query': 'In It, what is the name of the boy who meets Pennywise near the drain?',
        'expected_terms': ['пеннівайз', 'pennywise'],
        'changed_expected_terms': ['джордж', 'georgie'],
    },
]


def main():
    rag = RAGGenerator()
    cache = {}
    results = []

    def ask(query: str) -> str:
        if query not in cache:
            cache[query] = rag.generate(query, top_k=5).get('answer', '')
        return cache[query]

    for test in TESTS:
        first_answer = ask(test['query'])
        first_ok = contains_any(first_answer, test['expected_terms'])
        second_answer = None
        second_ok = None

        if test['type'] == 'Minimum Functionality':
            passed = first_ok
        elif test['type'] == 'Invariance':
            second_answer = ask(test['paraphrase'])
            second_ok = contains_any(second_answer, test['expected_terms'])
            passed = first_ok and second_ok
        else:
            second_answer = ask(test['changed_query'])
            second_ok = contains_any(second_answer, test['changed_expected_terms'])
            passed = first_ok and second_ok

        record = {
            'id': test['id'], 'type': test['type'], 'passed': passed,
            'query': test['query'], 'answer': first_answer,
            'second_query': test.get('paraphrase', test.get('changed_query')),
            'second_answer': second_answer,
            'first_fact_found': first_ok, 'second_fact_found': second_ok,
        }
        results.append(record)
        print(f"{test['id']} ({test['type']}): {'ПРОЙДЕНО' if passed else 'ПОМИЛКА'}")
        print(f"  Відповідь 1: {first_answer}")
        if second_answer is not None:
            print(f"  Відповідь 2: {second_answer}")

    passed_count = sum(item['passed'] for item in results)
    print(f'\nРезультат: {passed_count}/{len(results)} тестів пройдено')

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps({'passed': passed_count, 'total': len(results), 'tests': results},
                   ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(f'Результати збережено: {REPORT_PATH}')


if __name__ == '__main__':
    main()
