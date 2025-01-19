import os
import json
import re
import time
import faiss
import numpy as np
import subprocess
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher
from sentence_transformers import SentenceTransformer
from sklearn.metrics import ndcg_score

# Define paths
QUERY_DIR = "C:\\Users\\asus\\PycharmProjects\\SparseDenseRanking\\query-relJudgments"
SPARSE_INDEX_PATH = "lucene_index"
DENSE_INDEX_PATH = "faiss_index"

# ---------------------- #
# ✅ Step 1: Parse Queries 
# ---------------------- #
def parse_trec_queries(file_path):
    """
    Parses TREC format query files and returns a dictionary of queries.
    """
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
# ✅ Step 2: Parse Relevance Judgments
# ------------------------------ #
def parse_qrels(file_path):
    """
    Parses TREC relevance judgment files and returns a dictionary of query-document relevance.
    """
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
# ✅ Step 3: Load All Queries & Judgments
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

print(f"✅ Loaded {len(queries)} queries and {len(qrels)} relevance judgments.")

# ------------------------------------- #
# ✅ Step 4: Define Retrieval Functions
# ------------------------------------- #
def search_sparse(query, index_path, top_k=10):
    """Performs BM25 search on the sparse index."""
    searcher = LuceneSearcher(index_path)
    searcher.set_bm25(k1=0.9, b=0.4)

    hits = searcher.search(query, k=top_k)
    return [(hit.docid.strip().lower(), hit.score) for hit in hits]  # Normalize doc IDs

def search_dense(query, index_path, model_name="sentence-transformers/all-mpnet-base-v2", top_k=10):
    """Performs vector similarity search on the dense FAISS index."""
    model = SentenceTransformer(model_name)
    index = faiss.read_index(index_path)
    doc_ids = np.load(index_path + "_doc_ids.npy", allow_pickle=True)

    query_embedding = model.encode(query, convert_to_numpy=True, normalize_embeddings=True).reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)

    results = [(doc_ids[i].strip().lower(), 1 / (1 + distances[0][j])) for j, i in enumerate(indices[0])]
    return results

# ------------------------------------- #
# ✅ Step 5: Evaluation Metrics
# ------------------------------------- #
def mean_reciprocal_rank(ranking, relevant_docs):
    """Calculates Mean Reciprocal Rank (MRR)."""
    for i, doc_id in enumerate(ranking):
        if doc_id in relevant_docs:
            return 1 / (i + 1)
    return 0

def convert_to_relevance_vector(relevant_docs, ranking):
    """Convert a ranked list into a relevance score vector (1 if relevant, 0 if not)."""
    return [1 if doc_id in relevant_docs else 0 for doc_id in ranking]

# ------------------------------------- #
# ✅ Step 6: Evaluate Queries
# ------------------------------------- #

def evaluate(queries, qrels):
    """Evaluates ranking effectiveness using MRR and NDCG."""
    bm25_mrr, bm25_ndcg, dense_mrr, dense_ndcg = [], [], [], []
    x=0
    for query_id, query_data in queries.items():
        query_text = query_data["title"]
        relevant_docs = qrels.get(query_id, {})

        if not relevant_docs:
            print(f"⚠️ Skipping Query {query_id}: No relevant documents found.")
            continue

        print(f"\n🔍 Evaluating Query {query_id}: {query_text}")

        bm25_results = search_sparse(query_text, SPARSE_INDEX_PATH, top_k=10)
      #  dense_results = search_dense(query_text, DENSE_INDEX_PATH, top_k=10)

        bm25_ranking = [doc_id for doc_id, _ in bm25_results] if bm25_results else []
      #  dense_ranking = [doc_id for doc_id, _ in dense_results] if dense_results else []

        # Convert to binary relevance vectors
        bm25_relevance = convert_to_relevance_vector(relevant_docs, bm25_ranking)
      #  dense_relevance = convert_to_relevance_vector(relevant_docs, dense_ranking)

        # Compute MRR
        if bm25_ranking:
            bm25_mrr.append(mean_reciprocal_rank(bm25_ranking, relevant_docs))
     #   if dense_ranking:
     #       dense_mrr.append(mean_reciprocal_rank(dense_ranking, relevant_docs))

        # Compute NDCG (Ensure numeric values)
        if sum(bm25_relevance) > 0:
            bm25_ndcg.append(ndcg_score([bm25_relevance], [list(range(len(bm25_relevance), 0, -1))]))
     #   if sum(dense_relevance) > 0:
     #       dense_ndcg.append(ndcg_score([dense_relevance], [list(range(len(dense_relevance), 0, -1))]))
     #   x+=1
     #   if x==100:   
     #       break
    print("\n📊 **Final Evaluation Metrics**")
    print(f"BM25 MRR: {np.mean(bm25_mrr):.4f}, NDCG: {np.mean(bm25_ndcg) if bm25_ndcg else 0:.4f}")
   # print(f"Dense Retrieval MRR: {np.mean(dense_mrr):.4f}, NDCG: {np.mean(dense_ndcg) if dense_ndcg else 0:.4f}")

# Run Evaluation
evaluate(queries, qrels)