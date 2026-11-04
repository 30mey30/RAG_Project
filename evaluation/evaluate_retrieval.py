import argparse
import gc
import hashlib
import json
from datetime import datetime
from pathlib import Path

import chromadb
import torch

from rag_chroma import (
    CHROMA_DIR, CSV_PATH, EMBEDDING_MODEL_NAME, RERANKER_MODEL_NAME, generate_answer,
    get_embedding_model, get_reranker, load_documents_from_csv, rerank_faq,
)


BASE_DIR = Path(__file__).resolve().parent.parent
CASES_PATH = BASE_DIR / "data" / "synthetic_eval.jsonl"


def score_retrieval(expected_ids, retrieved_ids, mode):
    if mode == "unanswerable":
        if expected_ids:
            raise ValueError("سؤال بدون پاسخ نباید شناسهٔ مرجع داشته باشد.")
        return None, None, None
    if mode not in {"single", "any", "all"} or not expected_ids:
        raise ValueError("نوع ارتباط یا شناسه‌های مرجع معتبر نیستند.")
    found = [doc_id for doc_id in expected_ids if doc_id in retrieved_ids]
    hit = len(found) == len(expected_ids) if mode == "all" else bool(found)
    ranks = [retrieved_ids.index(doc_id) + 1 for doc_id in found]
    return hit, min(ranks) if ranks else None, len(found) / len(expected_ids)


def main():
    parser = argparse.ArgumentParser(description="سنجش بازیابی روی سؤال‌های مصنوعی")
    parser.add_argument("--dataset", type=Path, default=CASES_PATH, help="مسیر فایل سؤال‌های ارزیابی")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--limit", type=int, help="تعداد نمونه‌های نخست برای آزمایش")
    parser.add_argument("--answers", action="store_true", help="تولید و ذخیرهٔ پاسخ مدل زبانی")
    parser.add_argument("--rerank", action="store_true", help="بازمرتب‌سازی سندهای اولیه")
    args = parser.parse_args()
    if args.top_k < 1 or (args.limit is not None and args.limit < 1):
        parser.error("تعداد نتایج و تعداد نمونه‌ها باید بیشتر از صفر باشند.")

    source_hash = hashlib.sha256(CSV_PATH.read_bytes()).hexdigest()
    dataset_bytes = args.dataset.read_bytes()
    dataset_hash = hashlib.sha256(dataset_bytes).hexdigest()
    cases = [json.loads(line) for line in dataset_bytes.decode("utf-8").splitlines() if line.strip()]
    if args.limit is not None:
        cases = cases[:args.limit]
    if not cases:
        raise ValueError("مجموعهٔ سؤال‌های آزمایشی خالی است.")
    if any(case["source_sha256"] != source_hash for case in cases):
        raise ValueError("فایل منبع نسبت به زمان ساخت مجموعه تغییر کرده است.")

    collection = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection("faq_collection")
    if (collection.metadata or {}).get("embedding_model") != EMBEDDING_MODEL_NAME:
        raise ValueError("مدل سازندهٔ بردارهای پایگاه با مدل انتخاب‌شده یکسان نیست.")
    stored = collection.get(include=["documents", "metadatas"])
    stored_docs = {doc_id: (text, meta) for doc_id, text, meta in zip(
        stored["ids"], stored["documents"], stored["metadatas"]
    )}
    expected_docs = {doc["id"]: (doc["text"], doc["metadata"]) for doc in load_documents_from_csv(CSV_PATH)}
    if stored_docs != expected_docs:
        raise ValueError("متن یا متادیتای پایگاه داده با CSV یکسان نیست.")

    model = get_embedding_model()
    query_vectors = model.encode([case["question"] for case in cases], show_progress_bar=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    reranker = get_reranker() if args.rerank else None
    candidate_count = max(10, args.top_k) if args.rerank else args.top_k
    results = collection.query(query_embeddings=query_vectors.tolist(), n_results=candidate_count)
    candidates_by_case = [
        {key: [results[key][i]] for key in ("ids", "documents", "metadatas", "distances")}
        for i in range(len(cases))
    ]
    selected_by_case = []
    for i, case in enumerate(cases):
        candidates = candidates_by_case[i]
        selected_by_case.append(
            rerank_faq(case["question"], candidates, reranker, top_k=args.top_k)
            if args.rerank else candidates
        )
        if args.rerank:
            print(f"بازمرتب‌سازی: {i + 1}/{len(cases)}", flush=True)
    if reranker is not None:
        del reranker
        gc.collect()

    output = BASE_DIR / f"retrieval_results_{datetime.now():%Y%m%d_%H%M%S_%f}.jsonl"
    hit_count = 0
    initial_hit_count = 0
    reciprocal_rank_sum = 0.0
    answerable_count = 0
    with output.open("x", encoding="utf-8", newline="\n") as file:
        for i, case in enumerate(cases):
            candidates = candidates_by_case[i]
            selected = selected_by_case[i]
            retrieved_ids = selected["ids"][0]
            relevant_ids = case["expected_chunk_ids"]
            mode = case.get("relevance_mode", "any")
            hit, first_relevant_rank, coverage = score_retrieval(relevant_ids, retrieved_ids, mode)
            initial_hit, _, _ = score_retrieval(relevant_ids, candidates["ids"][0][:args.top_k], mode)
            if hit is not None:
                answerable_count += 1
                hit_count += hit
                initial_hit_count += initial_hit
            reciprocal_rank_sum += 1 / first_relevant_rank if first_relevant_rank else 0
            record = {
                "case_id": case["case_id"],
                "question": case["question"],
                "expected_answer": case["expected_answer"],
                "source_answer": case["source_answer"],
                "expected_chunk_ids": relevant_ids,
                "retrieved_ids": retrieved_ids,
                "retrieved_texts": selected["documents"][0],
                "distances": selected["distances"][0],
                "candidate_ids": candidates["ids"][0],
                "reranker_model": RERANKER_MODEL_NAME if args.rerank else None,
                "rerank_scores": selected["rerank_scores"][0] if args.rerank else None,
                "retrieval_ranks": selected["retrieval_ranks"][0] if args.rerank else None,
                "hit": hit,
                "first_relevant_rank": first_relevant_rank,
                "expected_id_coverage": coverage,
                "relevance_mode": mode,
                "question_types": case.get("question_types", []),
                "top_k": args.top_k,
                "query_embedding_model": EMBEDDING_MODEL_NAME,
                "collection_embedding_model": collection.metadata["embedding_model"],
                "collection": "faq_collection",
                "source_sha256": source_hash,
                "dataset_sha256": dataset_hash,
                "generator_model": case["generator_model"],
                "review_status": case["review_status"],
            }
            if args.answers:
                print(f"ساخت پاسخ: {i + 1}/{len(cases)}", flush=True)
                try:
                    record["generated_answer"] = generate_answer(
                        case["question"], {"documents": [record["retrieved_texts"]]}
                    )
                except RuntimeError as error:
                    record["answer_error"] = str(error)
                    file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    file.flush()
                    raise
                record["answer_model"] = "qwen3-vl:4b-instruct-q4_K_M"
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.flush()
    print(f"تعداد سؤال‌ها: {len(cases)}")
    print(f"سندهای مرجع لازم در نتایج پیدا شدند: {hit_count}/{answerable_count}")
    if args.rerank:
        print(f"سندهای مرجع لازم در {args.top_k} نتیجهٔ اولیه، قبل از بازمرتب‌سازی: {initial_hit_count}/{answerable_count}")
    if answerable_count:
        print(f"میانگین معکوس رتبهٔ اولین سند مرتبط تا {args.top_k} نتیجه: {reciprocal_rank_sum / answerable_count:.3f}")
    print(f"سؤال‌های بدون پاسخ، خارج از معیار بازیابی: {len(cases) - answerable_count}")
    print(f"فایل نتیجه: {output}")
    print("کیفیت نتیجه به بازبینی سؤال‌ها و سندهای مرجع بستگی دارد.")


if __name__ == "__main__":
    main()
