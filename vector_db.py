from typing import Optional
import faiss
import numpy as np
from files import FileArgument
from token_transformer import TokenTransformer
import os

class VectorDB:

    def __init__(self):
        self.index: faiss.IndexFlatL2 | None = None
        self.transformer = TokenTransformer()
        self.np_db_path = './data/embeddings.npz'
        self.faiss_db_path = './data/faiss_index.index'

    def create_index(self, embeddings_np: np.ndarray) -> faiss.IndexFlatL2:
        dimension = embeddings_np.shape[1]
        index: faiss.IndexFlatL2 = faiss.IndexFlatL2(dimension)
        index.add(embeddings_np)
        self.index = index
        self.save_faiss_index(index)
        return index
    
    def save_faiss_index(self, index: faiss.IndexFlatL2) -> None:
        faiss.write_index(index, self.faiss_db_path)
        print("Saved FAISS Index!")
        
    def load_faiss_index(self) -> Optional[faiss.IndexFlatL2]:
        if not os.path.exists(self.faiss_db_path):
            print(f"File {self.faiss_db_path} not found.")
            return None
        
        index = faiss.read_index(self.faiss_db_path)
        self.index = index
        print("Loaded FAISS Index!")
        return index

    def save_np_index(self, argument: FileArgument) -> np.ndarray:
        embeddings_np = self.transformer.get_np_vectors(argument.embeddings)
        np.savez(self.np_db_path, embeddings=embeddings_np, files=argument.filenames, texts=argument.text_list,)
        print("Saved NumPy Embeddings!")
        return embeddings_np

    def load_np_data(self) -> Optional[FileArgument]:
        if not os.path.exists(self.np_db_path):
            print(f"File {self.np_db_path} not found.")
            return None
        
        data = np.load(self.np_db_path)
        embeddings_np = data['embeddings']
        filenames = data['files']
        texts = data['texts']
        
        print("Loaded NumPy Embeddings!")
        return FileArgument(filenames=filenames, text_list=texts, embeddings=[], embeddings_np=embeddings_np)
