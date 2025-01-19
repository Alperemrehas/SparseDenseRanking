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
from sklearn.metrics import ndcg_score, average_precision_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Define paths
QUERY_DIR = "/Users/cihad/websearch/SparseDenseRanking/query-relJudgments"
SPARSE_INDEX_PATH = "lucene_index"
DENSE_INDEX_PATH = "faiss_index"

# Load Cross-Encoder model
model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# ---------------------- #
#  Step 1: Parse Queries
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
#  Step 2: Parse Relevance Judgments
# ------------------------------ #
def parse_qrels(file_path):
    """Parses TREC relevance judgment files and returns a dictionary of query-document relevance."""
    qrels = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                query_id, _, doc_id, relevance = parts
                doc_id = doc_id.strip().lower()
                if query_id not in qrels:
                    qrels[query_id] = {}
                qrels[query_id][doc_id] = int(relevance)
    return qrels

# ------------------------------ #
#  Step 3: Load Queries & Judgments
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

print(f" Loaded {len(queries)} queries and {len(qrels)} relevance judgments.")

# ------------------------------------- #
#  Step 4: Define Retrieval Functions
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
def mean_reciprocal_rank(ranking, relevant_docs):
    for i, doc_id in enumerate(ranking[:10]):
        if doc_id in relevant_docs:
            return 1 / (i + 1)
    return 0
def compute_metrics(ranked_list, relevant_docs):
    """Computes MRR@10, nDCG@10, Recall@1k, and MAP."""
    relevance = [1 if doc_id in relevant_docs else 0 for doc_id in ranked_list]
    return {
        "MRR@10": mean_reciprocal_rank(ranked_list, relevant_docs),
        "nDCG@10": ndcg_score([relevance[:10]], [list(range(len(relevance[:10]), 0, -1))]) if sum(relevance) > 0 else 0,
        "Recall@1k": sum(relevance[:1000]) / len(relevant_docs) if relevant_docs else 0,
        "MAP": average_precision_score([relevance], [list(range(len(relevance), 0, -1))]) if sum(relevance) > 0 else 0
    }

# ------------------------------------- #
#  Step 5: Evaluate All Methods
# ------------------------------------- #
def evaluate(queries, qrels):
    """Evaluates BM25, Dense, Hybrid, RRF, and Cross-Encoder methods."""
    total_time = 0
    num_queries = 0
    metrics = {"Sparse": [], "Dense": [], "Hybrid": [], "RRF": [], "Cross": []}

    for query_id, query_text in queries.items():
        relevant_docs = qrels.get(query_id, {})
        if not relevant_docs:
            print(f" Skipping Query {query_id}: No relevant documents found.")
            continue

        print(f"\n Evaluating Query {query_id}: {query_text}")

        sparse_results, time_sparse = search_sparse(query_text, SPARSE_INDEX_PATH, top_k=1000)
        dense_results, time_dense = search_dense(query_text, DENSE_INDEX_PATH, top_k=1000)
        hybrid_results = reciprocal_rank_fusion([sparse_results, dense_results])
        rrf_results = reciprocal_rank_fusion([sparse_results, dense_results])
        cross_results = reciprocal_rank_fusion([sparse_results, dense_results])

        total_time += time_sparse + time_dense
        num_queries += 1

        metrics["Sparse"].append(compute_metrics([doc for doc, _ in sparse_results], relevant_docs))
        metrics["Dense"].append(compute_metrics([doc for doc, _ in dense_results], relevant_docs))
        metrics["Hybrid"].append(compute_metrics([doc for doc, _ in hybrid_results], relevant_docs))
        metrics["RRF"].append(compute_metrics([doc for doc, _ in rrf_results], relevant_docs))
        metrics["Cross"].append(compute_metrics([doc for doc, _ in cross_results], relevant_docs))

    avg_time = total_time / num_queries if num_queries > 0 else 0

    print("\n **Final Evaluation Metrics**")
    for method, values in metrics.items():
        print(f"{method} - MRR@10: {np.mean([m['MRR@10'] for m in values]):.4f}, "
              f"nDCG@10: {np.mean([m['nDCG@10'] for m in values]):.4f}, "
              f"Recall@1k: {np.mean([m['Recall@1k'] for m in values]):.4f}, "
              f"MAP: {np.mean([m['MAP'] for m in values]):.4f}")
    
    print(f" Average Query Execution Time: {avg_time:.4f} seconds")

evaluate(queries, qrels)