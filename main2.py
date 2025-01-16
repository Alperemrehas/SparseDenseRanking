import os
import json
import time
import faiss
import numpy as np
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher
from pyserini.search.lucene import LuceneSearcher
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

'''
sparse index
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input ./ \
  --index indexes/lucene_index \
  --generator DefaultLuceneDocumentGenerator \
  --threads 4 \
  --storePositions --storeDocvectors --storeRaw
  
'''

searcher = LuceneSearcher("/Users/cihad/websearch/SparseDenseRanking/indexes/lucene_index")
searcher.set_bm25(k1=0.9, b=0.4)
queries = {
    "renewable energy": ["1", "3"],
    "machine learning applications": ["2", "4"]
}
for query in queries:
    print(f"Query: {query}")
    print("Sparse Results:")
    hits = searcher.search(query, k=10)  
    print(hits)
    result = [(hit.docid, hit.score) for hit in hits]
    print(result)
