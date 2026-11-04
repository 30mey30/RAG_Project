from pathlib import Path
import json
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import chromadb
import pandas as pd
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from sentence_transformers import CrossEncoder, SentenceTransformer
from rank_bm25 import BM25Okapi

# مسیرهای پروژه
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "data" / "faq_qa_pairs_excel.csv"
MODELS_DIR = BASE_DIR / "models"
CHROMA_DIR = BASE_DIR / "chroma_db_bge_m3"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

def load_documents_from_csv(csv_path: Path):
    """خواندن فایل CSV و تبدیل هر ردیف به ساختار سند استاندارد"""
    df = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    required_columns = {"category", "question", "answer"}
    if not required_columns.issubset(df.columns):
        raise ValueError("فایل CSV باید ستون‌های category، question و answer را داشته باشد.")

    documents = []
    
    for index, row in df.iterrows():
        question = row["question"].strip()
        answer = row["answer"].strip()
        category = row["category"].strip()

        if not question or not answer:
            raise ValueError(f"سؤال یا پاسخِ رکورد {index + 1} خالی است.")

        doc = {
            "id": f"faq_{index + 1:03d}",
            "text": f"سؤال: {question}\nپاسخ: {answer}",
            "metadata": {
                "category": category,
                "question": question,
                "source": csv_path.name,
            },
        }
        documents.append(doc)
    if not documents:
        raise ValueError("هیچ سؤال‌وجوابی در فایل پیدا نشد.")
    return documents

def get_embedding_model():
    """بارگذاری مدل امبدینگ از کش محلی"""
    return SentenceTransformer(
        EMBEDDING_MODEL_NAME,
        device="cuda",
        local_files_only=True,
    )

def setup_vector_database(documents, model):
    """ایجاد یا باز کردن دیتابیس کرومادی‌بی و ایندکس اسناد"""
    # ذخیره پایدار روی دیسک در پوشه chroma_db
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    
    # ساخت یا دریافت کالکشن با معیار شباهت کسینوسی (cosine)
    collection = client.get_or_create_collection(
        name="faq_collection",
        metadata={"hnsw:space": "cosine", "embedding_model": EMBEDDING_MODEL_NAME}
    )
    if (collection.metadata or {}).get("embedding_model") != EMBEDDING_MODEL_NAME:
        raise ValueError("مدل سازندهٔ بردارهای پایگاه با مدل انتخاب‌شده یکسان نیست.")

    texts_to_embed = [doc["text"] for doc in documents]
    for doc, text in zip(documents, texts_to_embed):
        token_count = len(model.tokenizer(text, truncation=False)["input_ids"])
        if token_count > model.max_seq_length:
            print(f"هشدار: سند {doc['id']} دارای {token_count} توکن است؛ "
                  f"مدل فقط {model.max_seq_length} توکن اول را می‌بیند.")

    current_count = collection.count()
    print(f"تعداد رکوردهای موجود در ChromaDB: {current_count}")

    # اگر دیتابیس خالی است، اسناد را امبد کرده و ذخیره می‌کنیم
    if current_count == 0:
        print("در حال استخراج بردارها و ذخیره در ChromaDB...")
        embeddings = model.encode(texts_to_embed, show_progress_bar=True)

        collection.add(
            ids=[doc["id"] for doc in documents],
            embeddings=embeddings.tolist(),
            documents=[doc["text"] for doc in documents],
            metadatas=[doc["metadata"] for doc in documents],
        )
        print(f"با موفقیت {collection.count()} سند روی دیسک ذخیره شد.")
    else:
        stored = collection.get(include=["documents", "metadatas"])
        stored_documents = {
            doc_id: (text, metadata)
            for doc_id, text, metadata in zip(
                stored["ids"], stored["documents"], stored["metadatas"]
            )
        }
        expected_documents = {
            doc["id"]: (doc["text"], doc["metadata"]) for doc in documents
        }
        if stored_documents != expected_documents:
            raise ValueError("داده‌های CSV با پایگاه داده یکسان نیستند؛ پایگاه داده باید بازسازی شود.")
        print("متن‌ها و اطلاعات سندها با پایگاه داده یکسان هستند.")

    return collection

def search_faq(query: str, collection, model, top_k: int = 3, category_filter: str = None):
    """جستجوی معنایی برای یافتن مرتبط‌ترین پاسخ‌ها به پرسش کاربر"""
    query_vector = model.encode([query])
    
    where_clause = {"category": category_filter} if category_filter else None

    results = collection.query(
        query_embeddings=query_vector.tolist(),
        n_results=top_k,
        where=where_clause
    )
    return results


def tokenize(text):
    return re.findall(r"[^\W_]+", text.lower())


def hybrid_search(query, collection, model, documents, bm25, top_k=10, rrf_k=60):
    """ادغام رتبه‌های جست‌وجوی معنایی و BM25 با RRF."""
    if top_k < 1 or rrf_k < 1:
        raise ValueError("تعداد نتایج و ثابت RRF باید مثبت باشند.")
    semantic = search_faq(query, collection, model, top_k=min(top_k, len(documents)))
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)
    matches = [i for i, doc in enumerate(documents) if set(query_tokens) & set(tokenize(doc["text"]))]
    lexical = sorted(matches, key=lambda i: (-float(scores[i]), i))[:top_k]
    fused = {}
    for rank, doc_id in enumerate(semantic["ids"][0], 1):
        fused[doc_id] = fused.get(doc_id, 0.0) + 1 / (rrf_k + rank)
    for rank, index in enumerate(lexical, 1):
        doc_id = documents[index]["id"]
        fused[doc_id] = fused.get(doc_id, 0.0) + 1 / (rrf_k + rank)
    ids = sorted(fused, key=lambda doc_id: (-fused[doc_id], doc_id))[:top_k]
    by_id = {doc["id"]: doc for doc in documents}
    distances = dict(zip(semantic["ids"][0], semantic["distances"][0]))
    return {"ids": [ids], "documents": [[by_id[i]["text"] for i in ids]],
            "metadatas": [[by_id[i]["metadata"] for i in ids]],
            "distances": [[distances.get(i) for i in ids]],
            "rrf_scores": [[fused[i] for i in ids]]}


def get_reranker():
    """بارگذاری بازمرتب‌ساز چندزبانه از کش محلی روی GPU."""
    return CrossEncoder(RERANKER_MODEL_NAME, device="cuda", max_length=512, local_files_only=True)


def rerank_faq(question, search_results, reranker, top_k=3):
    """بازمرتب‌سازی سندهای بازیابی‌شده با امتیاز جفت سؤال و سند."""
    if top_k < 1:
        raise ValueError("تعداد نتیجه‌های نهایی باید بیشتر از صفر باشد.")
    documents = search_results["documents"][0]
    if not documents:
        return {**search_results, "rerank_scores": [[]], "retrieval_ranks": [[]]}

    scores = reranker.predict([(question, document) for document in documents], show_progress_bar=False)
    indices = sorted(range(len(documents)), key=lambda i: float(scores[i]), reverse=True)[:top_k]
    reranked = dict(search_results)
    for key in ("ids", "documents", "metadatas", "distances", "rrf_scores"):
        if search_results.get(key) is not None:
            reranked[key] = [[search_results[key][0][i] for i in indices]]
    reranked["rerank_scores"] = [[float(scores[i]) for i in indices]]
    reranked["retrieval_ranks"] = [[i + 1 for i in indices]]
    return reranked


def generate_answer(question, search_results, model_name="qwen3-vl:4b-instruct-q4_K_M"):
    """ساخت پاسخ با مدل محلی اولاما و سندهای بازیابی‌شده."""
    documents = search_results["documents"][0]
    if not documents:
        return "سند مرتبطی پیدا نشد."

    context = "\n\n".join(documents)
    prompt = (
        "فقط بر اساس متن‌های زیر به سؤال به فارسی پاسخ بده. "
        "اگر اطلاعات کافی نیست، بگو نمی‌دانم.\n\n"
        f"متن‌ها:\n{context}\n\nسؤال: {question}\nپاسخ:"
    )
    payload = json.dumps({
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2},
    }).encode("utf-8")
    request = Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=120) as response:
            return json.load(response)["response"]
    except HTTPError as error:
        raise RuntimeError(f"اولاما خطای HTTP {error.code} داد؛ مدل و سرویس را بررسی کنید.") from error
    except (URLError, TimeoutError) as error:
        raise RuntimeError("ارتباط با اولاما برقرار نشد؛ اجرای سرویس را بررسی کنید.") from error


if __name__ == "__main__":
    import sys
    # تنظیم انکودینگ خروجی کنسول برای کاراکترهای فارسی در ویندوز
    sys.stdout.reconfigure(encoding="utf-8")

    print("--- گام ۱: خواندن اطلاعات از فایل CSV ---")
    docs = load_documents_from_csv(CSV_PATH)
    print(f"تعداد {len(docs)} پرسش‌وپاسخ بارگذاری شد.")

    print("\n--- گام ۲: بارگذاری مدل امبدینگ ---")
    embed_model = get_embedding_model()

    print("\n--- گام ۳: ذخیره‌سازی در پایگاه داده برداری ChromaDB ---")
    faq_collection = setup_vector_database(docs, embed_model)
    bm25 = BM25Okapi([tokenize(doc["text"]) for doc in docs])

    print("\n--- گام ۴: بارگذاری مدل بازمرتب‌سازی ---")
    reranker = get_reranker()

    provider = TracerProvider(resource=Resource.create({"openinference.project.name": "faq-rag"}))
    exporter = OTLPSpanExporter(endpoint="http://127.0.0.1:4317", insecure=True, timeout=3)
    provider.add_span_processor(BatchSpanProcessor(exporter, schedule_delay_millis=1000))
    tracer = provider.get_tracer("faq-rag")

    print("\n--- گام ۵: جستجو و بازمرتب‌سازی ---")
    print("برای خروج، بدون نوشتن سؤال Enter را بزنید.")
    while True:
        question = input("\nسؤال شما: ").strip()
        if not question:
            break

        with tracer.start_as_current_span("پاسخ به سؤال") as root_span:
            root_span.set_attribute("openinference.span.kind", "CHAIN")
            root_span.set_attribute("input.value", question)
            with tracer.start_as_current_span("جستجوی سندها") as search_span:
                search_span.set_attribute("openinference.span.kind", "RETRIEVER")
                search_span.set_attribute("input.value", question)
                candidates = hybrid_search(question, faq_collection, embed_model, docs, bm25, top_k=10)
                search_span.set_attribute("output.value", json.dumps(candidates, ensure_ascii=False))

            print("\nترتیب اولیهٔ بازیابی ترکیبی:")
            for rank, (doc_id, score) in enumerate(zip(candidates["ids"][0], candidates["rrf_scores"][0]), start=1):
                print(f"{rank}. {doc_id} | امتیاز RRF: {score:.4f}")

            with tracer.start_as_current_span("بازمرتب‌سازی سندها") as rerank_span:
                rerank_span.set_attribute("openinference.span.kind", "RERANKER")
                rerank_span.set_attribute("input.value", json.dumps({"question": question, "ids": candidates["ids"][0]}, ensure_ascii=False))
                search_res = rerank_faq(question, candidates, reranker, top_k=3)
                rerank_span.set_attribute("output.value", json.dumps({"ids": search_res["ids"][0], "scores": search_res["rerank_scores"][0]}, ensure_ascii=False))

            print("\nسه سند انتخاب‌شده پس از بازمرتب‌سازی:")
            for i in range(len(search_res["ids"][0])):
                doc_id = search_res["ids"][0][i]
                distance = search_res["distances"][0][i]
                category = search_res["metadatas"][0][i]["category"]
                text = search_res["documents"][0][i]
                similarity = f"{1.0 - distance:.3f}" if distance is not None else "در نتایج اولیهٔ معنایی نبود"
                rerank_score = search_res["rerank_scores"][0][i]
                initial_rank = search_res["retrieval_ranks"][0][i]
                print(f"\n--- نتیجه {i + 1} | {doc_id} | رتبهٔ ترکیبی اولیه: {initial_rank} | شباهت برداری: {similarity} | امتیاز بازمرتب‌سازی: {rerank_score:.3f} | دسته: {category} ---")
                print(text)

            try:
                print("\nدر حال ساخت پاسخ با مدل زبانی...")
                with tracer.start_as_current_span("ساخت پاسخ") as answer_span:
                    answer_span.set_attribute("openinference.span.kind", "LLM")
                    answer_span.set_attribute("llm.model_name", "qwen3-vl:4b-instruct-q4_K_M")
                    answer_span.set_attribute("input.value", json.dumps({"question": question, "documents": search_res["documents"][0]}, ensure_ascii=False))
                    answer = generate_answer(question, search_res)
                    answer_span.set_attribute("output.value", answer)
                    root_span.set_attribute("output.value", answer)
                print("پاسخ مدل:", answer)
            except RuntimeError as error:
                print(f"خطا: {error}")
    provider.shutdown()
