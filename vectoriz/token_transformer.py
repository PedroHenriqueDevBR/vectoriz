import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class TokenTransformer:

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
    def search(
        self,
        query: str,
        texts: list[str],
        index: faiss.IndexFlatL2,
        context_amount: int = 1,
    ) -> str:
        """
        Search texts based on semantic similarity to a given query.

        This method uses the vector embeddings of the query to search for the most similar texts 
        in a FAISS index and returns the matched texts as context.

        Parameters:
            query (str): The search query to find similar texts for.
            texts (list[str]): List of all texts that correspond to the vectors in the index.
            index (faiss.IndexFlatL2): A FAISS index containing vector embeddings of texts.
            context_amount (int, optional): Number of most similar texts to retrieve. Defaults to 1.

        Returns:
            str: A concatenated string of the most similar texts, separated by newlines.
        """
        
        query_embedding = self.query_to_embeddings(query)
        _, I = index.search(query_embedding, k=context_amount)
        context = ""

        for i in I[0]:
            context += texts[i] + "\n"

        return context.strip()
        
    def convert_vector_to_faiss_index(
        self, embeddings_np: np.ndarray
    ) -> faiss.IndexFlatL2:
        """
        Creates a FAISS index using the provided numpy array of embeddings.

        This method initializes a FAISS IndexFlatL2 (L2 distance/Euclidean) index with
        the dimensions from the input embeddings, adds the embeddings to the index.

        Args:
            embeddings_np (np.ndarray): A numpy array of embedding vectors to be indexed.
                The shape should be (n, dimension) where n is the number of vectors
                and dimension is the size of each vector.

        Returns:
            faiss.IndexFlatL2: The created FAISS index containing the embeddings.

        Note:
            This method also sets the index as an instance attribute and saves it to disk
            using the save_faiss_index method.
        """
        dimension = embeddings_np.shape[1]
        index: faiss.IndexFlatL2 = faiss.IndexFlatL2(dimension)
        index.add(embeddings_np)
        return index

    def transform_sentences_to_embeddings(self, sentences: list[str]) -> np.ndarray:
        """
        Transforms a list of sentences into embeddings using the model.

        Args:
            sentences (list[str]): A list of sentences to be transformed into embeddings.

        Returns:
            np.ndarray: A numpy array containing the embeddings for each sentence.
        """
        return self.model.encode(sentences)

    def query_to_embeddings(self, query: str) -> np.ndarray:
        """
        Converts a text query into embeddings using the model.

        Args:
            query (str): The text query to be transformed into embeddings.

        Returns:
            np.ndarray: The embedding representation of the query reshaped to
                        have dimensions (1, embedding_size).
        """
        return self.model.encode([query]).reshape(1, -1)

    def get_np_vectors(self, embeddings: list[float]) -> np.ndarray:
        """
        Converts input embeddings to a numpy array of float32 type.
        
        Args:
            embeddings (list[float]): The embeddings to convert.
            
        Returns:
            np.ndarray: A numpy array containing the embeddings as float32 values.
        """
        return np.array(embeddings).astype("float32")
