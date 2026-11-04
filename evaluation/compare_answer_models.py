import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

from rag_chroma import CSV_PATH, generate_answer


def main(argv=None):
    parser = argparse.ArgumentParser(description="آزمایش یک مدل زبانی با سندهای ثابت")
    parser.add_argument('--model', required=True, help='نام دقیق مدل در اولاما')
    parser.add_argument('--limit', type=int, default=24, help='تعداد سؤال‌ها')
    args = parser.parse_args(argv)
    if args.limit < 1:
        parser.error('تعداد سؤال‌ها باید مثبت باشد.')
    input_path = Path('results/answer_challenge_results_20261102_233056_823318.jsonl')
    input_bytes = input_path.read_bytes()
    rows = [json.loads(line) for line in input_bytes.decode('utf-8').splitlines() if line.strip()][:args.limit]
    source_hash = hashlib.sha256(CSV_PATH.read_bytes()).hexdigest()
    if not rows or any(row['source_sha256'] != source_hash or not row.get('retrieved_texts') for row in rows):
        raise ValueError('منبع تغییر کرده یا متن سندهای ورودی موجود نیست.')
    output = Path(f'llm_comparison_{datetime.now():%Y%m%d_%H%M%S_%f}.jsonl')
    with output.open('x', encoding='utf-8', newline='\n') as file:
        for n, row in enumerate(rows, 1):
            record = {k: row[k] for k in ('case_id', 'question', 'question_types', 'expected_answer', 'expected_chunk_ids', 'retrieved_ids', 'retrieved_texts', 'source_sha256', 'dataset_sha256')}
            record.update(answer_model=args.model, input_results_sha256=hashlib.sha256(input_bytes).hexdigest(), evaluation_status='not_reviewed')
            print(f"\nسؤال {n}/{len(rows)}: {row['question']}", flush=True)
            try:
                record['generated_answer'] = generate_answer(row['question'], {'documents': [row['retrieved_texts']]}, model_name=args.model)
            except RuntimeError as error:
                record['answer_error'] = str(error)
                file.write(json.dumps(record, ensure_ascii=False) + '\n')
                file.flush()
                print(f'خطا: {error}\nفایل تا اینجا: {output.resolve()}')
                raise SystemExit(1) from error
            file.write(json.dumps(record, ensure_ascii=False) + '\n')
            file.flush()
            print(f"پاسخ مرجع: {row['expected_answer']}\nپاسخ مدل: {record['generated_answer']}", flush=True)
    print(f'\nفایل نتیجه: {output.resolve()}')


if __name__ == '__main__':
    main()
