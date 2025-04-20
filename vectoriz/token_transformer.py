import faiss
import numpy as np
from typing import Self
from sentence_transformers import SentenceTransformer


class TokenData:
    """
    A class that holds text data along with their vector representations and indexing.
    This class is designed to store and manage tokenized texts, their corresponding
    embeddings, and a FAISS index for efficient similarity search.
    Attributes:
        texts (list[str]): List of text strings that have been tokenized.
        index (faiss.IndexFlatL2): A FAISS index using L2 (Euclidean) distance metric
                                  for similarity search.
        embeddings (np.ndarray, optional): Matrix of vector embeddings corresponding
                                          to the texts. Default is None.
    """

    def __init__(
        self,
        texts: list[str],
        index: faiss.IndexFlatL2,
        embeddings: np.ndarray = None,
    ):
        self.texts = texts
        self.index = index
        self.embeddings = embeddings

    @staticmethod
    def from_vector_db(vector_data) -> Self:
        """
        Creates a TokenData instance from a VectorDBClient.

        This static method extracts the necessary components from a VectorDBClient instance
        and uses them to instantiate a new TokenData object.

        Parameters
        ----------
        vector_data : VectorDBClient
            The VectorDBClient instance containing the FAISS index, embeddings, and text data.

        Returns
        -------
        TokenData
            A new TokenData instance initialized with texts, FAISS index, and embeddings from the
            VectorDBClient.
        """
        index = vector_data.faiss_index
        embeddings = vector_data.file_argument.embeddings
        texts = vector_data.file_argument.text_list
        return TokenData(texts, index, embeddings)

    @staticmethod
    def from_file_argument(file_argument, index: faiss.IndexFlatL2) -> Self:
        """
        Loads the FAISS index and numpy embeddings from a file argument.

        Args:
            file_argument (FileArgument): An instance of FileArgument containing
                                            the FAISS index and numpy embeddings.
        """
        embeddings = file_argument.embeddings
        texts = file_argument.text_list
        return TokenData(texts, index, embeddings)


class TokenTransformer:

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def search(
        self,
        query: str,
        index: faiss.IndexFlatL2,
        texts: list[str],
        context_amount: int = 1,
    ) -> str:
        query_embedding = self._query_to_embeddings(query)
        _, I = index.search(query_embedding, k=context_amount)
        context = ""

        for i in I[0]:
            context += texts[i] + "\n"

        return context.strip()

    def create_index(self, texts: list[str]) -> TokenData:
        """
        Creates a FAISS index from a list of text strings.

        This method converts the input texts to embeddings and then creates a
        FAISS IndexFlatL2 (L2 distance/Euclidean space) index from these embeddings.

        Args:
            texts (list[str]): A list of text strings to be indexed.

        Returns:
            faiss.IndexFlatL2: A FAISS index containing the embeddings of the input texts.
        """
        embeddings = self.text_to_embeddings(texts)
        index = self.embeddings_to_index(embeddings)
        return TokenData(texts, index, embeddings)

    def embeddings_to_index(self, embeddings_np: np.ndarray) -> faiss.IndexFlatL2:
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

    def text_to_embeddings(self, sentences: list[str]) -> np.ndarray:
        """
        Transforms a list of sentences into embeddings using the model.

        Args:
            sentences (list[str]): A list of sentences to be transformed into embeddings.

        Returns:
            np.ndarray: A numpy array containing the embeddings for each sentence.
        """
        return self.model.encode(sentences)

    def get_np_vectors(self, embeddings: list[float]) -> np.ndarray:
        """
        Converts input embeddings to a numpy array of float32 type.

        Args:
            embeddings (list[float]): The embeddings to convert.

        Returns:
            np.ndarray: A numpy array containing the embeddings as float32 values.
        """
        return np.array(embeddings).astype("float32")

    def _query_to_embeddings(self, query: str) -> np.ndarray:
        """
        Converts a text query into embeddings using the model.

        Args:
            query (str): The text query to be transformed into embeddings.

        Returns:
            np.ndarray: The embedding representation of the query reshaped to
                        have dimensions (1, embedding_size).
        """
        return self.model.encode([query]).reshape(1, -1)
