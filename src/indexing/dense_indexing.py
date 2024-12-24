from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def create_dense_index(docs, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(docs, convert_to_tensor=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    print(f"Dense index created with {len(docs)} documents.")
    return index
