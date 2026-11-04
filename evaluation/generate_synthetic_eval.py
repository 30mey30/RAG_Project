import argparse
import csv
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "faq_qa_pairs_excel.csv"
OUTPUT_PATH = BASE_DIR / "data" / "synthetic_eval.jsonl"
MODEL = "qwen3-vl:4b-instruct-q4_K_M"


def generate_cases(question, answer):
    prompt = (
        "برای ارزیابی بازیابی، بر اساس پرسش و پاسخ زیر دقیقاً سه سؤال فارسی متفاوت "
        "و مستقل از هم بساز که پاسخ هرکدام در متن موجود باشد. برای هر سؤال پاسخ "
        "مرجع در حد یک یا دو جمله بنویس. اعداد، استثناها و شرط‌ها را حفظ کن؛ اطلاعات نساز. "
        "سؤالات را طوری بنویس که بدون دیدن متن قابل فهم باشند.\n\n"
        f"پرسش منبع: {question}\nپاسخ منبع: {answer}"
    )
    item = {
        "type": "object",
        "properties": {"question": {"type": "string"}, "answer": {"type": "string"}},
        "required": ["question", "answer"],
    }
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "format": {
            "type": "object",
            "properties": {"items": {"type": "array", "minItems": 3, "maxItems": 3, "items": item}},
            "required": ["items"],
        },
        "stream": False,
        "options": {"temperature": 0.3, "num_ctx": 8192, "num_predict": 2048},
    }
    request = Request(
        "http://127.0.0.1:11434/api/generate",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=180) as response:
        items = json.loads(json.load(response)["response"])["items"]
    if (not isinstance(items, list) or len(items) != 3
            or any(not isinstance(item, dict)
                   or not isinstance(item.get("question"), str) or not item["question"].strip()
                   or not isinstance(item.get("answer"), str) or not item["answer"].strip()
                   for item in items)):
        raise ValueError("خروجی مدل شامل سه سؤال و پاسخ معتبر نیست.")
    if len({item["question"].strip() for item in items}) != 3:
        raise ValueError("مدل سؤال‌های تکراری تولید کرده است.")
    return items


def main():
    parser = argparse.ArgumentParser(description="ساخت مجموعهٔ مصنوعی ارزیابی از داده‌های FAQ")
    parser.add_argument("--limit", type=int, help="حداکثر تعداد سندهای جدید در این اجرا")
    args = parser.parse_args()

    source_hash = hashlib.sha256(CSV_PATH.read_bytes()).hexdigest()
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    counts = {}
    if OUTPUT_PATH.exists():
        with OUTPUT_PATH.open(encoding="utf-8") as file:
            for line in file:
                case = json.loads(line)
                if case["source_sha256"] != source_hash or case["generator_model"] != MODEL:
                    raise ValueError("منبع یا مدل سازنده تغییر کرده است؛ فایل قبلی را جداگانه نگه دارید.")
                chunk_id = case["expected_chunk_ids"][0]
                counts[chunk_id] = counts.get(chunk_id, 0) + 1
    if any(count != 3 for count in counts.values()):
        raise ValueError("تعداد سؤال‌های ذخیره‌شده برای یکی از سندها سه عدد نیست؛ فایل را بررسی کنید.")
    completed = set(counts)

    made = 0
    for index, row in enumerate(rows, start=1):
        chunk_id = f"faq_{index:03d}"
        if chunk_id in completed:
            continue
        question, answer = row["question"].strip(), row["answer"].strip()
        if not question or not answer:
            raise ValueError(f"سؤال یا پاسخ {chunk_id} خالی است.")

        for attempt in range(3):
            try:
                items = generate_cases(question, answer)
                break
            except (ValueError, KeyError) as error:
                if attempt == 2:
                    raise ValueError(f"ساخت سؤال‌های {chunk_id} پس از سه تلاش ناموفق بود.") from error
                print(f"خروجی {chunk_id} ناقص بود؛ تلاش دوباره...", flush=True)
        cases = [
            {
                "case_id": f"{chunk_id}_q{number}",
                "question": item["question"].strip(),
                "expected_answer": item["answer"].strip(),
                "expected_chunk_ids": [chunk_id],
                "source_question": question,
                "source_answer": answer,
                "source_sha256": source_hash,
                "generator_model": MODEL,
                "review_status": "unreviewed",
            }
            for number, item in enumerate(items, start=1)
        ]
        with OUTPUT_PATH.open("a", encoding="utf-8", newline="\n") as file:
            for case in cases:
                file.write(json.dumps(case, ensure_ascii=False) + "\n")
        made += 1
        print(f"{chunk_id}: سه سؤال ساخته شد ({len(completed) + made}/{len(rows)} سند)", flush=True)
        if args.limit is not None and made >= args.limit:
            break


if __name__ == "__main__":
    main()
