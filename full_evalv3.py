import os
import json
import re
import time
import faiss
import numpy as np
import torch
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher
from sentence_transformers import SentenceTransformer
from sklearn.metrics import ndcg_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Define paths
QUERY_DIR = "C:\\Users\\asus\\PycharmProjects\\SparseDenseRanking\\query-relJudgments"
SPARSE_INDEX_PATH = "lucene_index"
DENSE_INDEX_PATH = "faiss_index"

# Load Cross-Encoder model
model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# ---------------------- #
# Step 1: Parse Queries
# ---------------------- #
def parse_trec_queries(file_path):
    """Parses TREC format query files and returns a dictionary of queries."""
    queries = {}
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        topics = re.findall(r"<top>(.*?)</top>", content, re.DOTALL)

        for topic in topics:
            num_match = re.search(r"<num>\s*Number:\s*(\d+)", topic)
            title_match = re.search(r"<title>\s*(.*?)\n", topic)

            if num_match and title_match:
                query_id = num_match.group(1).strip()
                queries[query_id] = title_match.group(1).strip()
    return queries

# ------------------------------ #
# Step 2: Parse Relevance Judgments
# ------------------------------ #
def parse_qrels(file_path):
    """Parses TREC relevance judgment files and returns a dictionary of query-document relevance."""
    qrels = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                query_id, _, doc_id, relevance = parts
                doc_id = doc_id.strip().lower()  # Normalize doc IDs
                if query_id not in qrels:
                    qrels[query_id] = {}
                qrels[query_id][doc_id] = int(relevance)
    return qrels

# ------------------------------ #
# Step 3: Load Queries & Judgments
# ------------------------------ #
query_files = ["q-topics-org-SET1.txt", "q-topics-org-SET2.txt", "q-topics-org-SET3.txt"]
queries = {}
for query_file in query_files:
    file_path = os.path.join(QUERY_DIR, query_file)
    if os.path.exists(file_path):
        queries.update(parse_trec_queries(file_path))

qrel_files = ["qrels.trec7.adhoc_350-400.txt", "qrel_301-350_complete.txt", "qrels.trec8.adhoc.parts1-5_400-450"]
qrels = {}
for qrel_file in qrel_files:
    file_path = os.path.join(QUERY_DIR, qrel_file)
    if os.path.exists(file_path):
        qrels.update(parse_qrels(file_path))

print(f"Loaded {len(queries)} queries and {len(qrels)} relevance judgments.")

# ------------------------------------- #
# Step 4: Define Retrieval Functions
# ------------------------------------- #
def search_sparse(query, index_path, top_k=10):
    """Performs BM25 search on the sparse index."""
    searcher = LuceneSearcher(index_path)
    searcher.set_bm25(k1=0.9, b=0.4)

    start_time = time.time()
    hits = searcher.search(query, k=top_k)
    end_time = time.time()

    results = [(hit.docid.strip().lower(), hit.score) for hit in hits]
    return results, end_time - start_time

def search_dense(query, index_path, model_name="sentence-transformers/all-mpnet-base-v2", top_k=10):
    """Performs vector similarity search on the dense FAISS index."""
    model = SentenceTransformer(model_name)
    index = faiss.read_index(index_path)
    doc_ids = np.load(index_path + "_doc_ids.npy", allow_pickle=True)

    query_embedding = model.encode(query, convert_to_numpy=True, normalize_embeddings=True).reshape(1, -1)

    start_time = time.time()
    distances, indices = index.search(query_embedding, top_k)
    end_time = time.time()

    results = [(doc_ids[i].strip().lower(), 1 / (1 + distances[0][j])) for j, i in enumerate(indices[0])]
    return results, end_time - start_time

def reciprocal_rank_fusion(results_list, k=60):
    """Merges multiple ranked lists using Reciprocal Rank Fusion (RRF)."""
    fused_scores = {}
    for results in results_list:
        for rank, (doc_id, score) in enumerate(results):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

def cross_encoder_rerank(query, documents, batch_size=16):
    """Reranks top documents using a Cross-Encoder."""
    scores = []
    doc_texts = []
    doc_ids = []

    for doc in documents:
        if isinstance(doc, tuple) and len(doc) == 2:  
            doc_id, text = doc
        else:
            print(f"Skipping invalid document format: {doc}")
            continue

        if not isinstance(text, str) or not text.strip():
            print(f"Skipping document {doc_id}: Invalid or empty text format.")
            continue

        doc_ids.append(doc_id)
        doc_texts.append(text)

    if not doc_texts:
        print("No valid documents found for reranking. Returning empty list.")
        return []  # No valid documents to rerank

    # Debugging: Print sample input
    print(f" Cross-Encoder Query: {query}")
    print(f" First 3 Docs for Reranking: {doc_texts[:3]}")

    reranked_results = []
    for i in range(0, len(doc_texts), batch_size):
        batch_texts = doc_texts[i:i + batch_size]
        batch_ids = doc_ids[i:i + batch_size]

        #  Batch encode queries and documents
        inputs = tokenizer.batch_encode_plus(
            [(query, doc) for doc in batch_texts],
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        #  Forward pass through the model
        with torch.no_grad():
            logits = model(**inputs).logits.squeeze(-1)

        #  Zip doc_ids and scores together, then sort by score (higher is better)
        batch_results = list(zip(batch_ids, logits.tolist()))
        reranked_results.extend(batch_results)

    return sorted(reranked_results, key=lambda x: x[1], reverse=True)

def hybrid_search(query, index_path, model_name="sentence-transformers/all-MiniLM-L6-v2", top_k=10, alpha=0.5):
    """Performs hybrid retrieval combining BM25 and vector search."""
    sparse_results, _ = search_sparse(query, index_path, top_k=top_k)
    dense_results, _ = search_dense(query, DENSE_INDEX_PATH, top_k=top_k)

    combined_scores = {}
    for doc_id, score in sparse_results:
        combined_scores[doc_id] = alpha * score
    for doc_id, score in dense_results:
        combined_scores[doc_id] = combined_scores.get(doc_id, 0) + (1 - alpha) * score

    return sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
# ------------------------------------- #
#  Step 5: Evaluation Metrics
# ------------------------------------- #
def mean_reciprocal_rank(ranking, relevant_docs):
    for i, doc_id in enumerate(ranking[:10]):
        if doc_id in relevant_docs:
            return 1 / (i + 1)
    return 0

def compute_ndcg(ranked_list, relevant_docs, k=10):
    relevance = [1 if doc_id in relevant_docs else 0 for doc_id in ranked_list[:k]]
    return ndcg_score([relevance], [list(range(len(relevance), 0, -1))]) if sum(relevance) > 0 else 0

# ------------------------------------- #
#  Step 6: Evaluate All Methods
# ------------------------------------- #

def evaluate(queries, qrels):
    """Evaluates BM25, Dense, Hybrid, RRF, and Cross-Encoder methods."""
    sparse_mrr, dense_mrr, hybrid_mrr, rrf_mrr, cross_mrr = [], [], [], [], []
    x=0 

    for query_id, query_text in queries.items():
        relevant_docs = qrels.get(query_id, {})
        if not relevant_docs:
            print(f" Skipping Query {query_id}: No relevant documents found.")
            continue

        print(f"\n Evaluating Query {query_id}: {query_text}")

        sparse_results, _ = search_sparse(query_text, SPARSE_INDEX_PATH, top_k=1000)
        dense_results, _ = search_dense(query_text, DENSE_INDEX_PATH, top_k=1000)
        hybrid_results = hybrid_search(query_text, SPARSE_INDEX_PATH, top_k=1000)
        rrf_results = reciprocal_rank_fusion([sparse_results, dense_results])
        rrf_text_results = [(doc_id, "DOCUMENT TEXT PLACEHOLDER") for doc_id, _ in rrf_results[:50]]
        cross_results = cross_encoder_rerank(query_text, rrf_text_results)
    
        sparse_mrr.append(mean_reciprocal_rank([doc for doc, _ in sparse_results], relevant_docs))
        dense_mrr.append(mean_reciprocal_rank([doc for doc, _ in dense_results], relevant_docs))
        hybrid_mrr.append(mean_reciprocal_rank([doc for doc, _ in hybrid_results], relevant_docs))
        rrf_mrr.append(mean_reciprocal_rank([doc for doc, _ in rrf_results], relevant_docs))
        cross_mrr.append(mean_reciprocal_rank([doc for doc, _ in cross_results], relevant_docs))
 
    print("\n **Final Evaluation Metrics**")
    print(f"Sparse MRR@10: {np.mean(sparse_mrr):.4f},Dense: {np.mean(dense_mrr):.4f}, RRF: {np.mean(rrf_mrr):.4f}, Hybrid: {np.mean(hybrid_mrr):.4f}, Cross: {np.mean(cross_mrr):.4f}")

evaluate(queries, qrels)