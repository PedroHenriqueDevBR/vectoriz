from typing import Optional
import faiss
import numpy as np
from .files import FileArgument
from .token_transformer import TokenTransformer
import os


class VectorDBClient:

    def __init__(
        self,
        faiss_index: Optional[faiss.IndexFlatL2],
        file_argument: Optional[FileArgument],
    ):
        """
        Initialize the SavedVectorData with a FAISS index and embeddings.

        Args:
            faiss_index (faiss.IndexFlatL2): The FAISS index containing the vector data.
            embeddings (np.ndarray): The numpy array of embeddings associated with the index.
        """
        self.faiss_index = faiss_index
        self.file_argument = file_argument

    def save_data(self, faiss_db_path: str, np_db_path: str) -> None:
        """
        Save the FAISS index and numpy embeddings to disk.

        Args:
            faiss_db_path (str): Path to save the FAISS index.
            np_db_path (str): Path to save the numpy embeddings.
        """
        if self.faiss_index is None or self.file_argument is None:
            raise ValueError("FAISS index or file argument is not initialized.")

        vectorDB = VectorDB()
        vectorDB.save_faiss_index(self.faiss_index, faiss_db_path)
        vectorDB.save_numpy_embeddings(self.file_argument, np_db_path)
        
    def load_data(self, faiss_db_path: str, np_db_path: str) -> None:
        """
        Load the FAISS index and numpy embeddings from disk.

        Args:
            faiss_db_path (str): Path to load the FAISS index from.
            np_db_path (str): Path to load the numpy embeddings from.
        """
        vectorDB = VectorDB()
        self.faiss_index = vectorDB.load_faiss_index(faiss_db_path)
        self.file_argument = vectorDB.load_numpy_embeddings(np_db_path)


class VectorDB:

    def __init__(self):
        """
        Constructor for the class.

        Initializes the following attributes:
        - transformer: A TokenTransformer instance for text transformation.
        """
        self.transformer = TokenTransformer()

    def load_saved_data(
        self, faiss_db_path: str, np_db_path: str
    ) -> Optional[VectorDBClient]:
        """
        Load previously saved FAISS index and numpy embeddings data.

        This function attempts to load a FAISS index and numpy embeddings from specified paths.
        It combines them into a SavedVectorData object if both are successfully loaded.

        Parameters:
        ----------
        faiss_db_path : str
            Path to the saved FAISS index file
        np_db_path : str
            Path to the saved numpy embeddings file

        Returns:
        -------
        Optional[SavedVectorData]
            A SavedVectorData object containing the loaded index and embeddings if successful,
            or None if either file could not be loaded.
        """
        index = self.load_faiss_index(faiss_db_path)
        file_argument = self.load_numpy_embeddings(np_db_path)

        if index is None or file_argument is None:
            return None
        return VectorDBClient(index, file_argument)

    def save_faiss_index(
        self,
        index: faiss.IndexFlatL2,
        faiss_db_path: str,
    ) -> None:
        """
        Save a FAISS index to disk.

        This method takes a FAISS index and saves it to the specified location.
        It ensures the filename has the correct extension and the folder path ends with a slash.

        Args:
            index (faiss.IndexFlatL2): The FAISS index to save
            faiss_db_path (str): The directory and the name of the file to save the index as where the index will be saved

        Returns:
            None: This method doesn't return anything

        Note:
            If the filename doesn't end with '.index', the extension will be added automatically.
            If the folder_path doesn't end with '/', it will be added automatically.
        """
        faiss_db_path = (
            faiss_db_path
            if faiss_db_path.endswith(".index")
            else faiss_db_path + ".index"
        )
        faiss.write_index(index, faiss_db_path)

    def load_faiss_index(self, faiss_db_path: str) -> Optional[faiss.IndexFlatL2]:
        """
        Load a FAISS index from a specified file path.

        Args:
            faiss_db_path (str): Path to the FAISS index file.

        Returns:
            Optional[faiss.IndexFlatL2]: The loaded FAISS index if the file exists,
            None otherwise.

        Note:
            If the file does not exist, a message will be printed to console.
        """
        if not os.path.exists(faiss_db_path):
            return None
        return faiss.read_index(faiss_db_path)

    def save_numpy_embeddings(
        self,
        argument: FileArgument,
        np_db_path: str,
    ) -> None:
        """
        Save embeddings, chunk names, and texts to a compressed numpy file (.npz).
        Args:
            argument (FileArgument): An object containing embeddings, ndarray_data, chunk_names, and text_list.
            np_db_path (str): Directory path and filename where the file will be saved.
        Returns:
            np.ndarray: The numpy array containing the embeddings, either from argument.ndarray_data
                        or generated from argument.embeddings.
        Notes:
            The saved .npz file will contain three arrays:
            - 'embeddings': The vector embeddings
            - 'chunk_names': The chunk names
            - 'texts': The text content
        """
        np_db_path = np_db_path if np_db_path.endswith(".npz") else np_db_path + ".npz"

        embeddings_np: np.ndarray = None
        if argument.ndarray_data is not None:
            embeddings_np = argument.ndarray_data
        else:
            embeddings_np = self.transformer.get_np_vectors(argument.embeddings)

        np.savez(
            np_db_path,
            embeddings=embeddings_np,
            chunk_names=argument.chunk_names,
            texts=argument.text_list,
        )

    def load_numpy_embeddings(self, np_db_path: str) -> Optional[FileArgument]:
        """
        Load embeddings from a NumPy archive file.

        This method reads embeddings, filenames, and text data from a .npz file
        created by a previous vectorization process.

        Args:
            np_db_path (str): Path to the NumPy archive file containing embeddings and metadata.

        Returns:
            Optional[FileArgument]: A FileArgument object containing the loaded data,
                                   or None if the specified file does not exist.
        """
        if not os.path.exists(np_db_path):
            return None

        data = np.load(self.np_db_path)
        embeddings_np = data["embeddings"]
        chunk_names = data["chunk_names"]
        texts = data["texts"]

        return FileArgument(
            chunk_names=chunk_names,
            text_list=texts,
            embeddings=[],
            ndarray_data=embeddings_np,
        )
