import os
import json
import re
import time
import faiss
import numpy as np
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher
from sentence_transformers import SentenceTransformer
from sklearn.metrics import ndcg_score

# Define paths
QUERY_DIR = "C:\\Users\\asus\\PycharmProjects\\SparseDenseRanking\\query-relJudgments"
DENSE_INDEX_PATH = "faiss_index"

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
            desc_match = re.search(r"<desc>\s*Description:\s*(.*?)\n\n", topic, re.DOTALL)
            narr_match = re.search(r"<narr>\s*Narrative:\s*(.*?)\n\n", topic, re.DOTALL)

            if num_match and title_match:
                query_id = num_match.group(1).strip()
                queries[query_id] = {
                    "title": title_match.group(1).strip(),
                    "description": desc_match.group(1).strip() if desc_match else "",
                    "narrative": narr_match.group(1).strip() if narr_match else "",
                }
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
                qrels[query_id][doc_id] = int(relevance)  # Store relevance as an integer
    return qrels

# ------------------------------ #
# Step 3: Load All Queries & Judgments
# ------------------------------ #
query_files = [
    "q-topics-org-SET1.txt",
    "q-topics-org-SET2.txt",
    "q-topics-org-SET3.txt"
]
queries = {}
for query_file in query_files:
    file_path = os.path.join(QUERY_DIR, query_file)
    if os.path.exists(file_path):
        queries.update(parse_trec_queries(file_path))

qrel_files = [
    "qrels.trec7.adhoc_350-400.txt",
    "qrel_301-350_complete.txt",
    "qrels.trec8.adhoc.parts1-5_400-450"
]
qrels = {}
for qrel_file in qrel_files:
    file_path = os.path.join(QUERY_DIR, qrel_file)
    if os.path.exists(file_path):
        qrels.update(parse_qrels(file_path))

print(f"Loaded {len(queries)} queries and {len(qrels)} relevance judgments.")

# ------------------------------------- #
# Step 4: Define Dense Retrieval
# ------------------------------------- #
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
    execution_time = end_time - start_time

    return results, execution_time

# ------------------------------------- #
# Step 5: Evaluation Metrics
# ------------------------------------- #
def mean_reciprocal_rank(ranking, relevant_docs):
    """Calculates Mean Reciprocal Rank (MRR@10)."""
    for i, doc_id in enumerate(ranking[:10]):
        if doc_id in relevant_docs:
            return 1 / (i + 1)
    return 0

def compute_ndcg(ranked_list, relevant_docs, k=10):
    """Computes nDCG@10."""
    if not relevant_docs:
        return 0.0
    relevance = [1 if doc_id in relevant_docs else 0 for doc_id in ranked_list[:k]]
    if sum(relevance) == 0:
        return 0.0
    return ndcg_score([relevance], [list(range(len(relevance), 0, -1))])

def recall_at_k(ranked_list, relevant_docs, k=1000):
    """Computes Recall@1k."""
    retrieved_docs = set(ranked_list[:k])
    relevant_docs_set = set(relevant_docs)
    return len(retrieved_docs & relevant_docs_set) / len(relevant_docs_set)

def average_precision(ranked_list, relevant_docs):
    """Computes Mean Average Precision (MAP)."""
    num_relevant = 0
    precision_sum = 0
    for i, doc_id in enumerate(ranked_list):
        if doc_id in relevant_docs:
            num_relevant += 1
            precision_sum += num_relevant / (i + 1)
    return precision_sum / len(relevant_docs) if relevant_docs else 0

# ------------------------------------- #
# Step 6: Evaluate Queries
# ------------------------------------- #
def evaluate(queries, qrels):
    """Evaluates dense ranking effectiveness using MRR, nDCG, Recall, MAP, and Query Time."""
    dense_mrr, dense_ndcg, dense_recall, dense_map = [], [], [], []
    query_times = []

    for query_id, query_data in queries.items():
        query_text = query_data["title"]
        relevant_docs = qrels.get(query_id, {})

        if not relevant_docs:
            print(f" Skipping Query {query_id}: No relevant documents found.")
            continue

        print(f"\n Evaluating Query {query_id}: {query_text}")

        # Measure execution time
        dense_results, dense_time = search_dense(query_text, DENSE_INDEX_PATH, top_k=1000)

        dense_ranking = [doc_id for doc_id, _ in dense_results] if dense_results else []
        query_times.append(dense_time)

        # Compute Metrics Safely
        dense_mrr.append(mean_reciprocal_rank(dense_ranking, relevant_docs))
        dense_ndcg.append(compute_ndcg(dense_ranking, relevant_docs))
        dense_recall.append(recall_at_k(dense_ranking, relevant_docs, k=1000))
        dense_map.append(average_precision(dense_ranking, relevant_docs))

    print("\n**Final Evaluation Metrics**")
    print(f"Dense Retrieval MRR@10: {np.mean(dense_mrr):.4f}, nDCG@10: {np.mean(dense_ndcg):.4f}, Recall@1k: {np.mean(dense_recall):.4f}, MAP: {np.mean(dense_map):.4f}")
    print(f"Average Query Execution Time: {np.mean(query_times):.4f} seconds")

# Run Evaluation
evaluate(queries, qrels)