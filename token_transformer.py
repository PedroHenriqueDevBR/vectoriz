import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class TokenTransformer:
    
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def transform(self, sentences: list[str]) -> np.ndarray:
        return self.model.encode(sentences)
    
    def format_query(self, query: str) -> np.ndarray:
        return self.model.encode([query]).reshape(1, -1)

    def get_np_vectors(self, embeddings: np.ndarray):
        return np.array(embeddings).astype('float32')
    
    def search(self, query: str, texts: list[str], index: faiss.IndexFlatL2, depth: int = 1) -> str:
        query_embedding = self.format_query(query)
        _, I = index.search(query_embedding, k=depth)
        context = ''
        
        for i in I[0]:
            context += texts[i] + '\n'
            
        return context.strip()