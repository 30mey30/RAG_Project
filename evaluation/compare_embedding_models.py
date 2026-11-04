import argparse
import gc
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from evaluation.evaluate_retrieval import score_retrieval
from rag_chroma import CSV_PATH, MODELS_DIR, get_reranker, load_documents_from_csv, rerank_faq


MODELS = {
    "e5": "intfloat/multilingual-e5-small",
    "bge": "BAAI/bge-m3",
    "qwen": "Qwen/Qwen3-Embedding-0.6B",
    "e5large": "intfloat/multilingual-e5-large-instruct",
}


def main(argv=None):
    parser = argparse.ArgumentParser(description="مقایسهٔ بازیابی و رنکر با مدل‌های embedding محلی")
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=True,
                        help="بازمرتب‌سازی نتایج؛ برای غیرفعال‌کردن از --no-rerank استفاده کنید")
    parser.add_argument("--dataset", type=Path, default=Path("data/eval_development_large.jsonl"))
    parser.add_argument("--output", type=Path, help="مسیر خروجی؛ در صورت حذف، نام تازه ساخته می‌شود")
    args = parser.parse_args(argv)
    if args.output is None:
        args.output = Path(f"embedding_compare_{args.model}_{datetime.now():%Y%m%d_%H%M%S_%f}.jsonl")

    dataset = args.dataset.read_bytes()
    cases = [json.loads(line) for line in dataset.decode("utf-8").splitlines() if line.strip()]
    source_hash = hashlib.sha256(CSV_PATH.read_bytes()).hexdigest()
    if not cases or any(case["source_sha256"] != source_hash for case in cases):
        raise ValueError("دیتاست خالی است یا با نسخهٔ فایل CSV مطابقت ندارد.")
    documents = load_documents_from_csv(CSV_PATH)
    texts = [doc["text"] for doc in documents]
    questions = [case["question"] for case in cases]
    start = time.perf_counter()
    model = SentenceTransformer(MODELS[args.model], cache_folder=str(MODELS_DIR) if args.model in {"e5", "qwen", "e5large"} else None,
                                device="cuda", local_files_only=True)
    prefix_doc, prefix_query = ("passage: ", "query: ") if args.model == "e5" else ("", "")
    if args.model == "e5large":
        prefix_query = "Instruct: Given a question, retrieve relevant passages that answer the question\nQuery: "
    too_long = sum(len(model.tokenizer(prefix_doc + text, truncation=False)["input_ids"]) > model.max_seq_length for text in texts)
    doc_vectors = model.encode([prefix_doc + text for text in texts], batch_size=4 if args.model == "qwen" else 32, normalize_embeddings=True, show_progress_bar=True)
    query_vectors = model.encode([prefix_query + text for text in questions], batch_size=4 if args.model == "qwen" else 32, prompt_name="query" if args.model == "qwen" else None, normalize_embeddings=True, show_progress_bar=True)
    embedding_seconds = time.perf_counter() - start
    scores = np.asarray(query_vectors) @ np.asarray(doc_vectors).T
    del model, query_vectors, doc_vectors
    gc.collect()
    torch.cuda.empty_cache()
    reranker = get_reranker() if args.rerank else None
    output = []
    for i, case in enumerate(cases):
        indices = np.argsort(-scores[i], kind="stable")[:10]
        candidates = {
            "ids": [[documents[j]["id"] for j in indices]],
            "documents": [[texts[j] for j in indices]],
            "metadatas": [[documents[j]["metadata"] for j in indices]],
            "distances": [[float(1 - scores[i, j]) for j in indices]],
        }
        initial = candidates["ids"][0]
        final = rerank_faq(case["question"], candidates, reranker, top_k=3)["ids"][0] if reranker else None
        mode = case.get("relevance_mode", "any")
        gold = case["expected_chunk_ids"]
        output.append({"case_id": case["case_id"], "question": case["question"],
                       "question_types": case.get("question_types", []), "relevance_mode": mode,
                       "expected_ids": gold, "initial_ids": initial, "reranked_ids": final,
                       "initial_top10": score_retrieval(gold, initial, mode),
                       "initial_top3": score_retrieval(gold, initial[:3], mode),
                       "reranked_top3": score_retrieval(gold, final, mode) if final is not None else None,
                       "reranker_enabled": args.rerank,
                       "model": MODELS[args.model], "dataset_sha256": hashlib.sha256(dataset).hexdigest(),
                       "source_sha256": source_hash, "embedding_seconds_including_load": embedding_seconds,
                       "truncated_documents": too_long})
        print(f"ارزیابی: {i + 1}/{len(cases)}", flush=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as file:
        for item in output:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
    answerable = [item for item in output if item["relevance_mode"] != "unanswerable"]
    print(f"\nمدل: {MODELS[args.model]} | سؤال‌های پاسخ‌دار: {len(answerable)} | بدون پاسخ: {len(output) - len(answerable)}")
    print(f"رنکر: {'روشن' if args.rerank else 'خاموش'}")
    fields = [("initial_top10", "ده نامزد اولیه"), ("initial_top3", "سه نتیجهٔ اولیه")]
    if args.rerank:
        fields.append(("reranked_top3", "سه نتیجه پس از رنکر"))
    for field, label in fields:
        print(f"{label}: {sum(item[field][0] for item in answerable)}/{len(answerable)}")
    for item in output:
        print(f"\n{item['case_id']} | {item['question']}")
        print(f"  مرجع: {item['expected_ids']} | نوع: {item['relevance_mode']}")
        print(f"  نتایج اولیه: {item['initial_ids'][:3]} | پیدا شدن: {item['initial_top3'][0]}")
        if args.rerank:
            print(f"  پس از رنکر: {item['reranked_ids']} | پیدا شدن: {item['reranked_top3'][0]}")
    print(f"\nنتیجه: {args.output.resolve()}؛ سندهای طولانی‌تر از محدودیت مدل: {too_long}")


if __name__ == "__main__":
    main()
