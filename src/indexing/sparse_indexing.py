from pyserini.index import SimpleIndexer

def create_sparse_index(data_dir, index_dir):
    indexer = SimpleIndexer(data_dir, index_dir)
    indexer.set_analyzer('standard')  # Use a standard analyzer
    indexer.index()
    print(f"Sparse index created at {index_dir}")
