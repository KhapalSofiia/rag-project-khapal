import json
import re
from pathlib import Path

from generation import RAGGenerator

REPORT_PATH = Path('reports/lab5_metamorphic_results.json')
K = 5

PAIRS = [
    {
        'id': 'MT1', 'document_id': '615_It',
        'query': 'In It, which clown speaks to Georgie near the storm drain?',
        'paraphrase': 'What is the name of the clown Georgie encounters at the drain in It?',
    },
    {
        'id': 'MT2', 'document_id': '1000_Shrek',
        'query': 'Who is the princess Shrek is sent to rescue?',
        'paraphrase': 'What is the name of the princess rescued by Shrek?',
    },
    {
        'id': 'MT3', 'document_id': '414_Fantastic_Beasts_and_Where_to_Find_Them',
        'query': 'Who is the No-Maj who befriends Newt in Fantastic Beasts?',
        'paraphrase': 'What is the name of the non-magical man who becomes Newt Scamander’s friend?',
    },
]


def get_field(record, *names):
    if isinstance(record, dict):
        for name in names:
            if record.get(name) is not None:
                return record[name]
    else:
        for name in names:
            value = getattr(record, name, None)
            if value is not None:
                return value
    return None


def extract_sources(result):
    """Adapt here if generate() names its retrieval metadata differently."""
    sources = get_field(result, 'sources', 'retrieved_chunks', 'chunks', 'retrieved_documents')
    if not isinstance(sources, list):
        raise ValueError(
            'generate() did not return retrieved source chunks. '
            'Add `sources` to its return dict (top-5 chunks with chunk_id and document_id). '
            f'Available keys: {list(result.keys()) if isinstance(result, dict) else type(result).__name__}'
        )
    records = []
    for source in sources[:K]:
        meta = get_field(source, 'metadata') or {}
        chunk_id = get_field(source, 'chunk_id', 'id') or get_field(meta, 'chunk_id', 'id')
        doc_id = get_field(source, 'document_id', 'doc_id', 'source') or get_field(meta, 'document_id', 'doc_id', 'source')
        if chunk_id is None or doc_id is None:
            raise ValueError(f'Chunk/source IDs missing in retrieved source: {source!r}')
        records.append({'chunk_id': str(chunk_id), 'document_id': str(doc_id)})
    return records


def extract_cited_ids(result, sources):
    """Use structured citation IDs if available, else map answer [1] to source #1."""
    direct = get_field(result, 'cited_chunk_ids')
    if direct is not None:
        if not isinstance(direct, (list, tuple, set)):
            raise ValueError('cited_chunk_ids must be a list of chunk IDs')
        return set(map(str, direct))

    answer = get_field(result, 'answer') or ''
    indices = {int(x) for x in re.findall(r'\[(\d+)\]', answer)}
    return {sources[i - 1]['chunk_id'] for i in indices if 1 <= i <= len(sources)}


def ask(rag, query):
    result = rag.generate(query, top_k=K)
    sources = extract_sources(result)
    cited = extract_cited_ids(result, sources)
    return {'sources': sources, 'cited_ids': cited, 'answer': get_field(result, 'answer') or ''}


def main():
    rag = RAGGenerator()
    records = []
    for pair in PAIRS:
        a = ask(rag, pair['query'])
        b = ask(rag, pair['paraphrase'])
        gold = pair['document_id']
        hit_a = any(s['document_id'] == gold for s in a['sources'])
        hit_b = any(s['document_id'] == gold for s in b['sources'])
        intersection = a['cited_ids'] & b['cited_ids']
        union = a['cited_ids'] | b['cited_ids']
        jaccard = len(intersection) / len(union) if union else None
        row = {
            'id': pair['id'], 'query': pair['query'], 'paraphrase': pair['paraphrase'],
            'gold_document_id': gold, 'recall_at_5_original': int(hit_a),
            'recall_at_5_paraphrase': int(hit_b), 'recall_invariant': hit_a == hit_b,
            'both_relevant': hit_a and hit_b,
            'cited_chunk_ids_original': sorted(a['cited_ids']),
            'cited_chunk_ids_paraphrase': sorted(b['cited_ids']),
            'shared_cited_chunk_ids': sorted(intersection),
            'citation_jaccard': jaccard,
            'answer_original': a['answer'], 'answer_paraphrase': b['answer'],
        }
        records.append(row)
        value = 'N/A (no citations)' if jaccard is None else f'{jaccard:.2f}'
        print(f"{pair['id']}: Recall@5 {int(hit_a)} -> {int(hit_b)}; "
              f"both relevant: {hit_a and hit_b}; cited Jaccard: {value}")

    measurable = [r['citation_jaccard'] for r in records if r['citation_jaccard'] is not None]
    summary = {
        'pairs': len(records),
        'recall_at_5_original': sum(r['recall_at_5_original'] for r in records) / len(records),
        'recall_at_5_paraphrase': sum(r['recall_at_5_paraphrase'] for r in records) / len(records),
        'recall_invariance_rate': sum(r['recall_invariant'] for r in records) / len(records),
        'both_relevant_rate': sum(r['both_relevant'] for r in records) / len(records),
        'mean_citation_jaccard': (sum(measurable) / len(measurable)) if measurable else None,
        'pairs_without_citations': len(records) - len(measurable),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps({'summary': summary, 'tests': records}, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\nSummary:', json.dumps(summary, ensure_ascii=False, indent=2))
    print('Saved:', REPORT_PATH)


if __name__ == '__main__':
    main()
