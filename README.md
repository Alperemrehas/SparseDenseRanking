# SparseDenseRanking# SparseDenseRanking

A repository for exploring sparse (BM25) and dense (neural) ranking techniques over Istella22 and Financial Times datasets.

## Features
- Sparse and dense index creation using Lucene and neural models
- Evaluation of ranking effectiveness (e.g., NDCG, precision, recall)
- Support for Learning-to-Rank (LTR) models

## Setup
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt



allmethods.py —> includes  all base functions 
sparsesearcher.py --> includes sparse search query
dense_index_search.py —> includes dense index and dense search
Sparsedensecolab includes fais index creation and search 
combined_methods.py updated version of all methods
eval.py to evaluate searchers
eval_with_metrics.py added with many metrics
 

Dataset info :Sparse index :2025-01-19 02:55:21,208 INFO  [main] index.AbstractIndexer (AbstractIndexer.java:307) - Indexing Complete! 210,167 documents indexed
2025-01-19 02:55:21,209 INFO  [main] index.AbstractIndexer (AbstractIndexer.java:308) - ============ Final Counter Values ============
2025-01-19 02:55:21,209 INFO  [main] index.AbstractIndexer (AbstractIndexer.java:309) - indexed:          210,167
2025-01-19 02:55:21,209 INFO  [main] index.AbstractIndexer (AbstractIndexer.java:310) - unindexable:            0
2025-01-19 02:55:21,209 INFO  [main] index.AbstractIndexer (AbstractIndexer.java:311) - empty:                  0
2025-01-19 02:55:21,209 INFO  [main] index.AbstractIndexer (AbstractIndexer.java:312) - skipped:                0
2025-01-19 02:55:21,209 INFO  [main] index.AbstractIndexer (AbstractIndexer.java:313) - errors:                 0
2025-01-19 02:55:21,213 INFO  [main] index.AbstractIndexer (AbstractIndexer.java:316) - Total 210,167 documents indexed in 00:00:47
✅ Sparse index created in 49.14 seconds!

Dense index:Encoding Documents: 100%|██████████| 6568/6568 [1:30:08<00:00,  1.21it/s]
model_name="sentence-transformers/all-mpnet-base-v2"

Sparse search   :
📊 **Final Evaluation Metrics**
BM25 MRR: 0.9785, NDCG: 0.9825

📊 **Final Evaluation Metrics**
BM25 MRR@10: 0.9785, nDCG@10: 0.9760, Recall@1k: 0.1211, MAP: 0.0795
📌 Average Query Execution Time: 0.0974 seconds

Dense Search:📊 **Final Evaluation Metrics**
Dense Retrieval MRR: 0.9304, NDCG: 0.9377


📊 **Final Evaluation Metrics**
Dense Retrieval MRR@10: 0.9304, nDCG@10: 0.9189, Recall@1k: 0.0930, MAP: 0.0436
📌 Average Query Execution Time: 0.0379 seconds



Sparse MRR@10: 0.9785,Dense: 0.9304, RRF: 0.9813, Hybrid: 0.9780, Cross: 0.9602




