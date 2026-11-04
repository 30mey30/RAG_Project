import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

from rank_bm25 import BM25Okapi
from rag_chroma import CSV_PATH, generate_answer, get_embedding_model, get_reranker, hybrid_search, load_documents_from_csv, rerank_faq, setup_vector_database, tokenize


def main():
    parser = argparse.ArgumentParser(description='آزمایش پاسخ مدل روی سؤال‌های چالشی')
    parser.add_argument('--limit', type=int, default=24, help='تعداد سؤال‌ها')
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('تعداد سؤال‌ها باید مثبت باشد.')
    data = Path('data/answer_eval_challenging.jsonl').read_bytes()
    cases = [json.loads(line) for line in data.decode('utf-8').splitlines() if line.strip()][:args.limit]
    source_hash = hashlib.sha256(CSV_PATH.read_bytes()).hexdigest()
    if not cases or any(c['source_sha256'] != source_hash for c in cases):
        raise ValueError('دیتاست خالی است یا با فایل CSV فعلی مطابقت ندارد.')

    docs = load_documents_from_csv(CSV_PATH)
    model = get_embedding_model()
    collection = setup_vector_database(docs, model)
    bm25 = BM25Okapi([tokenize(d['text']) for d in docs])
    reranker = get_reranker()
    output = Path(f'answer_challenge_results_{datetime.now():%Y%m%d_%H%M%S_%f}.jsonl')
    with output.open('x', encoding='utf-8', newline='\n') as file:
        for n, case in enumerate(cases, 1):
            print(f"\nسؤال {n}/{len(cases)}: {case['question']}", flush=True)
            candidates = hybrid_search(case['question'], collection, model, docs, bm25, top_k=10)
            result = rerank_faq(case['question'], candidates, reranker, top_k=3)
            record = {'case_id': case['case_id'], 'question': case['question'], 'question_types': case['question_types'],
                      'expected_answer': case['expected_answer'], 'expected_chunk_ids': case['expected_chunk_ids'],
                      'relevance_mode': case['relevance_mode'], 'reference_review_status': case['review_status'],
                      'retrieved_ids': result['ids'][0], 'retrieved_texts': result['documents'][0],
                      'dataset_sha256': hashlib.sha256(data).hexdigest(), 'source_sha256': source_hash,
                      'answer_model': 'qwen3-vl:4b-instruct-q4_K_M', 'evaluation_status': 'not_reviewed'}
            try:
                record['generated_answer'] = generate_answer(case['question'], result)
            except RuntimeError as error:
                record['answer_error'] = str(error)
                file.write(json.dumps(record, ensure_ascii=False) + '\n')
                file.flush()
                print(f'خطا: {error}\nفایل تا اینجا: {output.resolve()}', flush=True)
                raise SystemExit(1) from error
            file.write(json.dumps(record, ensure_ascii=False) + '\n')
            file.flush()
            print(f"سندهای بازیابی‌شده: {record['retrieved_ids']}\nپاسخ مرجع: {case['expected_answer']}\nپاسخ مدل: {record['generated_answer']}", flush=True)
    print(f'\nفایل نتیجه: {output.resolve()}')


if __name__ == '__main__':
    main()
