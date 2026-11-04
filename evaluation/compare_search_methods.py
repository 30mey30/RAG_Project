import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi

from evaluation.evaluate_retrieval import score_retrieval
from rag_chroma import CHROMA_DIR, CSV_PATH, EMBEDDING_MODEL_NAME, RERANKER_MODEL_NAME, get_embedding_model, get_reranker, hybrid_search, load_documents_from_csv, rerank_faq, search_faq, tokenize


def main():
    parser = argparse.ArgumentParser(description="مقایسهٔ جست‌وجوی معنایی و ترکیبی")
    parser.add_argument("--dataset", type=Path, default=Path("data/eval_development_large.jsonl"))
    parser.add_argument("--limit", type=int, help="تعداد سؤال‌های نخست")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("تعداد سؤال‌ها باید مثبت باشد.")
    data = args.dataset.read_bytes()
    cases = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    if args.limit is not None:
        cases = cases[:args.limit]
    source_hash = hashlib.sha256(CSV_PATH.read_bytes()).hexdigest()
    if not cases or any(case["source_sha256"] != source_hash for case in cases):
        raise ValueError("دیتاست خالی است یا با فایل منبع فعلی مطابقت ندارد.")

    docs = load_documents_from_csv(CSV_PATH)
    collection = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection("faq_collection")
    if (collection.metadata or {}).get("embedding_model") != EMBEDDING_MODEL_NAME:
        raise ValueError("مدل سازندهٔ بردارهای پایگاه با مدل انتخاب‌شده یکسان نیست.")
    stored = collection.get(include=["documents", "metadatas"])
    actual = {i: (t, m) for i, t, m in zip(stored["ids"], stored["documents"], stored["metadatas"])}
    expected = {d["id"]: (d["text"], d["metadata"]) for d in docs}
    if actual != expected:
        raise ValueError("اسناد پایگاه با فایل CSV مطابقت ندارند.")
    model = get_embedding_model()
    bm25 = BM25Okapi([tokenize(d["text"]) for d in docs])
    reranker = get_reranker()
    output = Path(f"search_comparison_{datetime.now():%Y%m%d_%H%M%S_%f}.jsonl")
    scores = {"semantic": [0, 0], "hybrid": [0, 0]}
    rerank_scores = {"semantic": 0, "hybrid": 0}
    total = 0
    with output.open("x", encoding="utf-8", newline="\n") as file:
        for number, case in enumerate(cases, 1):
            question = case["question"]
            semantic_result = search_faq(question, collection, model, top_k=10)
            hybrid_result = hybrid_search(question, collection, model, docs, bm25, top_k=10)
            semantic = semantic_result["ids"][0]
            hybrid = hybrid_result["ids"][0]
            gold = case["expected_chunk_ids"]
            mode = case.get("relevance_mode", "any")
            record = {"case_id": case["case_id"], "question": question, "question_types": case.get("question_types", []),
                      "expected_ids": gold, "relevance_mode": mode, "semantic_ids": semantic, "hybrid_ids": hybrid,
                      "dataset_sha256": hashlib.sha256(data).hexdigest(), "source_sha256": source_hash,
                      "embedding_model": EMBEDDING_MODEL_NAME, "reranker_enabled": True, "reranker_model": RERANKER_MODEL_NAME}
            if mode != "unanswerable":
                total += 1
            for name, ids, candidates in (("semantic", semantic, semantic_result), ("hybrid", hybrid, hybrid_result)):
                for index, k in enumerate((3, 10)):
                    hit, rank, coverage = score_retrieval(gold, ids[:k], mode)
                    record[f"{name}_top{k}"] = {"hit": hit, "first_rank": rank, "coverage": coverage}
                    if hit is not None:
                        scores[name][index] += hit
                ranked = rerank_faq(question, candidates, reranker, top_k=3)
                final_ids = ranked["ids"][0]
                hit, rank, coverage = score_retrieval(gold, final_ids, mode)
                record[f"{name}_reranked_ids"] = final_ids
                record[f"{name}_reranked_top3"] = {"hit": hit, "first_rank": rank, "coverage": coverage}
                record[f"{name}_rerank_scores"] = ranked["rerank_scores"][0]
                if hit is not None:
                    rerank_scores[name] += hit
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.flush()
            print(f"\nسؤال {number}: {question}\n  مرجع: {gold} | نوع: {mode}\n  معنایی قبل: {semantic[:3]} | بعد: {record['semantic_reranked_ids']}\n  ترکیبی قبل: {hybrid[:3]} | بعد: {record['hybrid_reranked_ids']}", flush=True)
    print(f"\nتعداد سؤال‌های پاسخ‌دار: {total} | بدون پاسخ: {len(cases) - total}")
    for name, label in (("semantic", "معنایی"), ("hybrid", "ترکیبی")):
        print(f"روش {label}: سه نتیجهٔ اول {scores[name][0]}/{total} | ده نتیجهٔ اول {scores[name][1]}/{total}")
        print(f"روش {label} بعد از رنکر: سه نتیجهٔ اول {rerank_scores[name]}/{total}")
    print(f"فایل خروجی: {output.resolve()}")


if __name__ == "__main__":
    main()
