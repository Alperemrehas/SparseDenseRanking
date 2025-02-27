import os
import json
import time
import faiss
import numpy as np
import subprocess
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher
from sentence_transformers import SentenceTransformer
from sklearn.metrics import ndcg_score

# Define paths
DATASET_PATH = "documents.jsonl"  
SPARSE_INDEX_PATH = "lucene_index"
DENSE_INDEX_PATH = "faiss_index"

# Load dataset
def load_documents(dataset_path):
    documents = {}
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            documents[doc["id"]] = doc["contents"]
    return documents

documents = load_documents(DATASET_PATH)

### Step 1: Create Sparse Index using Pyserini ###
def create_sparse_index(documents, index_path):
    """Creates a sparse index using Pyserini via subprocess."""
    if not os.path.exists(index_path):
        os.makedirs(index_path)

    # Save documents in JSONL format
    with open("corpus.jsonl", "w", encoding="utf-8") as f:
        for doc_id, text in documents.items():
            f.write(json.dumps({"id": doc_id, "contents": text}) + "\n")

    print("Starting Sparse Indexing...")
    start_time = time.time()

    # Run Pyserini indexing command
    command = [
        "python", "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", "./",
        "--index", index_path,
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "4",
        "--storePositions", "--storeDocvectors", "--storeRaw"
    ]
    
    subprocess.run(command, check=True)
    end_time = time.time()
    print(f"Sparse index created in {end_time - start_time:.2f} seconds!")

# Run sparse indexing
create_sparse_index(documents, SPARSE_INDEX_PATH)

### Step 2: Create Dense Index using SBERT ###
def create_dense_index(documents, index_path, model_name="sentence-transformers/all-mpnet-base-v2"):
    """Creates a FAISS dense index using SBERT embeddings."""
    model = SentenceTransformer(model_name)
    embeddings = []
    doc_ids = list(documents.keys())

    print(f"Generating embeddings using {model_name}...")
    start_time = time.time()

    for doc_id in tqdm(doc_ids, desc="Encoding Documents"):
        embedding = model.encode(documents[doc_id], convert_to_numpy=True, normalize_embeddings=True)
        embeddings.append(embedding)

    embeddings = np.array(embeddings, dtype=np.float32)

    # Build FAISS index
    index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner Product for cosine similarity
    index.add(embeddings)

    # Save FAISS index
    faiss.write_index(index, index_path)
    np.save(index_path + "_doc_ids.npy", np.array(doc_ids))

    end_time = time.time()
    print(f"Dense index created in {end_time - start_time:.2f} seconds!")

# Run dense indexing
create_dense_index(documents, DENSE_INDEX_PATH)

### Step 3: Query Processing ###
def search_sparse(query, index_path, top_k=10):
    """Performs BM25 search on the sparse index."""
    searcher = LuceneSearcher(index_path)
    searcher.set_bm25(k1=0.9, b=0.4)

    start_time = time.time()
    hits = searcher.search(query, k=top_k)
    end_time = time.time()

    print(f"BM25 Query Execution Time: {end_time - start_time:.4f} seconds")
    return [(hit.docid, hit.score) for hit in hits]

def search_dense(query, index_path, model_name="sentence-transformers/all-mpnet-base-v2", top_k=10):
    """Performs vector similarity search on the dense FAISS index."""
    model = SentenceTransformer(model_name)
    index = faiss.read_index(index_path)
    doc_ids = np.load(index_path + "_doc_ids.npy", allow_pickle=True)

    query_embedding = model.encode(query, convert_to_numpy=True, normalize_embeddings=True).reshape(1, -1)

    start_time = time.time()
    distances, indices = index.search(query_embedding, top_k)
    end_time = time.time()

    print(f" Dense Query Execution Time: {end_time - start_time:.4f} seconds")
    results = [(doc_ids[i], 1 / (1 + distances[0][j])) for j, i in enumerate(indices[0])]
    return results

### Step 4: Evaluation ###
def mean_reciprocal_rank(ranking, relevant_docs):
    """Calculates Mean Reciprocal Rank (MRR)."""
    for i, doc_id in enumerate(ranking):
        if doc_id in relevant_docs:
            return 1 / (i + 1)
    return 0

def convert_to_relevance_vector(relevant_docs, ranking):
    """Convert a ranked list into a relevance score vector (1 if relevant, 0 if not)."""
    return [1 if doc_id in relevant_docs else 0 for doc_id in ranking]

def evaluate(queries, ground_truth):
    """Evaluates ranking effectiveness using MRR and NDCG."""
    bm25_mrr, bm25_ndcg, dense_mrr, dense_ndcg = [], [], [], []
    
    for query, relevant_docs in queries.items():
        print(f"\n Evaluating Query: {query}")

        bm25_results = search_sparse(query, SPARSE_INDEX_PATH, top_k=10)
        dense_results = search_dense(query, DENSE_INDEX_PATH, top_k=10)
        
        # Convert to ranking lists
        bm25_ranking = [doc_id for doc_id, _ in bm25_results]
        dense_ranking = [doc_id for doc_id, _ in dense_results]

        # Convert rankings to relevance scores
        bm25_relevance = convert_to_relevance_vector(relevant_docs, bm25_ranking)
        dense_relevance = convert_to_relevance_vector(relevant_docs, dense_ranking)

        # Compute metrics
        bm25_mrr.append(mean_reciprocal_rank(bm25_ranking, relevant_docs))
        dense_mrr.append(mean_reciprocal_rank(dense_ranking, relevant_docs))

        # Compute NDCG with numerical values
        bm25_ndcg.append(ndcg_score([bm25_relevance], [list(range(len(bm25_ranking), 0, -1))]))
        dense_ndcg.append(ndcg_score([dense_relevance], [list(range(len(dense_ranking), 0, -1))]))

    print("\n **Final Evaluation Metrics**")
    print(f"BM25 MRR: {np.mean(bm25_mrr):.4f}, NDCG: {np.mean(bm25_ndcg):.4f}")
    print(f"Dense Retrieval MRR: {np.mean(dense_mrr):.4f}, NDCG: {np.mean(dense_ndcg):.4f}")

# Example Queries for testing only
queries = {
    "renewable energy": ["1", "3"],
    "machine learning applications": ["2", "4"]
}

evaluate(queries, queries)