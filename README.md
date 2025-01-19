# SparseDenseRanking

A repository for exploring sparse (BM25) and dense (neural) ranking techniques over Istella22 and Financial Times datasets.

---

## Features
- Sparse and dense index creation using Lucene and neural models.
- Evaluation of ranking effectiveness using metrics like MRR, NDCG, Recall, and MAP.
- Combined approaches, including Reciprocal Rank Fusion (RRF) and Cross-Encoding.
- Utilities for query evaluation and relevance scoring.

---

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/SparseDenseRanking.git
   cd SparseDenseRanking
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Results
### Final Evaluation Metrics
- **Sparse**  
  - MRR@10: 0.9785  
  - nDCG@10: 0.9760  
  - Recall@1k: 0.1211  
  - MAP: 0.2543  

- **Dense**  
  - MRR@10: 0.9304  
  - nDCG@10: 0.9189  
  - Recall@1k: 0.0930  
  - MAP: 0.1422  

- **Hybrid**  
  - MRR@10: 0.9813  
  - nDCG@10: 0.9805  
  - Recall@1k: 0.1304  
  - MAP: 0.1389  

- **RRF**  
  - MRR@10: 0.9813  
  - nDCG@10: 0.9805  
  - Recall@1k: 0.1304  
  - MAP: 0.1389  

- **Cross-Encoding**  
  - MRR@10: 0.9813  
  - nDCG@10: 0.9805  
  - Recall@1k: 0.1304  
  - MAP: 0.1389  

---

### Dataset and Index Information
- **Sparse Index**  
  - Created using Lucene with 210,167 documents indexed in 49.14 seconds.  
  - BM25 Query Execution Time: 0.0974 seconds.  

- **Dense Index**  
  - Generated using `sentence-transformers/all-mpnet-base-v2` with FAISS.  
  - Query Execution Time: 0.0379 seconds.  

For sparse and dense index files, refer to [Google Drive](https://drive.google.com/drive/folders/1K9tUrY1xf-NgiPQEy6Xk--DBRMbtk9jc?usp=drive_link).

---

## Code Overview
### Key Modules
1. **`allmethods.py`**
   - **Purpose:** Contains base functions for loading datasets, creating sparse and dense indexes, querying, and evaluation.
   - **Highlights:**
     - Sparse index creation using Lucene (`create_sparse_index`).
     - Dense index creation with FAISS and SBERT embeddings (`create_dense_index`).
     - Ranking evaluation using MRR and NDCG metrics.

2. **`combined_methods.py`**
   - **Purpose:** Provides methods for parsing datasets and combining sparse and dense retrieval methods.
   - **Highlights:**
     - Hybrid retrieval techniques like Reciprocal Rank Fusion (RRF).
     - FAISS and Pyserini integrations for combined search.

3. **`dense_index_search.py`**
   - **Purpose:** Handles dense index creation and search functionality.
   - **Highlights:**
     - Document encoding with `sentence-transformers`.
     - Query execution using FAISS for dense vector similarity.

4. **`eval_dense.py`**
   - **Purpose:** Evaluates dense retrieval methods.
   - **Highlights:**
     - Metric calculation for NDCG, MRR, Recall@k, and MAP.
     - TREC-style query and relevance judgment parsing.

5. **`eval_sparse_with_diffrentmetrics.py`**
   - **Purpose:** Evaluates sparse retrieval with multiple metrics.
   - **Highlights:**
     - Query relevance scoring using BM25.
     - Advanced evaluation metrics implementation.

6. **`eval_with_mrr_sparse.py`**
   - **Purpose:** Evaluates sparse retrieval focusing on MRR.
   - **Highlights:**
     - Customized relevance vector conversion and ranking.

7. **`full_eval_allmethods.py`**
   - **Purpose:** Comprehensive evaluation across sparse, dense, hybrid, and cross-encoder methods.
   - **Highlights:**
     - Integration of Cross-Encoder ranking for fine-grained results.
     - RRF and hybrid search evaluations.

8. **`full_eval_sparse_densev1.py`**
   - **Purpose:** Variant of evaluation for sparse and dense retrieval methods.
   - **Highlights:**
     - Comparative metrics for sparse and dense indexes.
     - Hybrid retrieval combining BM25 and dense embeddings.

9. **`full_evalv3.py`**
   - **Purpose:** Final version of evaluation with optimized metrics.
   - **Highlights:**
     - Improved query execution pipeline.
     - Full implementation of cross-encoder reranking.

---

## Directory Structure
```
SparseDenseRanking/
├── README.md
├── allmethods.py
├── combined_methods.py
├── dense_index_search.py
├── eval_dense.py
├── eval_sparse_with_diffrentmetrics.py
├── eval_with_mrr_sparse.py
├── full_eval_allmethods.py
├── full_eval_sparse_densev1.py
├── full_evalv3.py
└── requirements.txt
```

---

## Contact
- **Cihad Tekinbaş**  
  Middle East Technical University  
  [cihad@metu.edu.tr](mailto:cihad@metu.edu.tr)  

- **Alper Emre Has**  
  Middle East Technical University  
  [alper.has@metu.edu.tr](mailto:alper.has@metu.edu.tr)
