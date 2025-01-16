import os
import json
import time
import faiss
import numpy as np
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pyserini.index.lucene as IndexCollection
 
'''
Sparse index creation (uncomment and run this command if the index does not exist):
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input ./ \
  --index indexes/lucene_index \
  --generator DefaultLuceneDocumentGenerator \
  --threads 4 \
  --storePositions --storeDocvectors --storeRaw
'''
 
def main():
    # Initialize LuceneSearcher with the correct index path
    searcher = LuceneSearcher("/Users/cihad/websearch/SparseDenseRanking/indexes/lucene_index")
    searcher.set_bm25(k1=0.9, b=0.4)  # Configure BM25 parameters
 
    # Define queries and their associated labels
    queries = {
        "renewable energy": ["1", "3"],
        "machine learning applications": ["2", "4"]
    }
 
    for query in queries:
        print(f"Query: {query}")
        print("Sparse Results:")
        hits = searcher.search(query, k=10)  # Perform search
        result = [(hit.docid, hit.score) for hit in hits]  # Extract doc IDs and scores
        print(result)
 
if __name__ == "__main__":
    main()