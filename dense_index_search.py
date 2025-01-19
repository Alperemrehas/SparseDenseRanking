import os
import json
import time
import faiss
import numpy as np
from tqdm import tqdm
#from pyserini.search import SimpleSearcher
#from pyserini.index import IndexReader
from pyserini.search.lucene import LuceneSearcher
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

DATASET_PATH = "documents.jsonl"  # JSONL file with {"id": "1", "text": "..."}
DENSE_INDEX_PATH = "faiss_index"

def load_documents(dataset_path):
    documents = {}
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            documents[doc["id"]] = doc["contents"]
    return documents

documents = load_documents(DATASET_PATH)

### Step 2: Create Dense Index using FAISS ###
def create_dense_index(documents, index_path, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = []
    doc_ids = list(documents.keys())

    for doc_id in tqdm(doc_ids, desc="Encoding Documents"):
        embedding = model.encode(documents[doc_id], convert_to_numpy=True)
        embeddings.append(embedding)

    embeddings = np.array(embeddings, dtype=np.float32)

    # Build FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance
    index.add(embeddings)

    # Save FAISS index
    faiss.write_index(index, index_path)
    np.save(index_path + "_doc_ids.npy", np.array(doc_ids))
    
create_dense_index(documents, DENSE_INDEX_PATH)


def search_dense(query, index_path, model_name="all-MiniLM-L6-v2", top_k=10):
    model = SentenceTransformer(model_name)
    index = faiss.read_index(index_path)
    doc_ids = np.load(index_path + "_doc_ids.npy", allow_pickle=True)

    query_embedding = model.encode(query, convert_to_numpy=True).reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)

    results = [(doc_ids[i], 1 / (1 + distances[0][j])) for j, i in enumerate(indices[0])]
    return results

queries = {
    "renewable energy": ["1", "3"],
    "machine learning applications": ["2", "4"]
}
queries = {
    "renewable energy": ["1", "3"],
    "machine learning applications": ["2", "4"]
}
for query in queries:
    print(f"Query: {query}")
    print("Dense Results:")
    dense_results = search_dense(query, DENSE_INDEX_PATH, top_k=10)
    print(dense_results)
    dense_ranking = [doc_id for doc_id, _ in dense_results]
