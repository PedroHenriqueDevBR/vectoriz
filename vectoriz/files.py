import os
import numpy as np
from typing import Optional
from .token_transformer import TokenTransformer


class FileArgument:
    def __init__(
        self,
        chunk_names: list[str],
        text_list: list[str],
        embeddings: list[float],
        ndarray_data: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initializes the FileProcessor instance with file data and embeddings.

        This constructor sets up an instance with chunk_names, their text content, and associated embeddings.
        It also initializes a TokenTransformer instance for potential token transformations.

        Parameters
        ----------
        chunk_names : list[str]
            List of chunk_names corresponding to processed files
        text_list : list[str]
            List of text content extracted from the files
        embeddings : list[float]
            List of embeddings (vector representations) of the text content
        ndarray_data : Optional[np.ndarray], default=None
            NumPy array representation of the embeddings for efficient vector operations

        Returns
        -------
        None
        """
        self.chunk_names: list[str] = chunk_names
        self.text_list: list[str] = text_list
        self.embeddings: list[float] = embeddings
        self.ndarray_data: np.ndarray = ndarray_data
        self.transformerInstance = TokenTransformer()

    def add_data(self, filename: str, text: str) -> None:
        """
        Adds text data to the vectorizer along with its filename and creates the corresponding embedding.
        This method appends the provided filename and text to their respective lists in the object,
        and also creates and stores the embedding vector for the text.
        Args:
            filename (str): The name of the file or identifier for the text data
            text (str): The text content to be added and embedded
        Returns:
            None: This method doesn't return anything, it updates the internal state of the object
        """

        self.chunk_names.append(filename)
        self.text_list.append(text)
        self.embeddings.append(self._create_embedding(text))

    def _create_embedding(self, text: str) -> np.ndarray:
        """
        Creates an embedding vector for the given text using the transformer model.
        This method transforms the input text into a numerical vector representation
        that captures semantic meaning, which can be used for similarity comparisons
        or as input to machine learning models.
        Args:
            text (str): The text to be embedded.
        Returns:
            np.ndarray: A numpy array containing the embedding vector for the input text.
        """

        return self.transformerInstance.transform_sentences_to_embeddings([text])[0]


class FilesFeature:

    def __init__(self):
        """
        Initialize a new instance of the class.

        This constructor initializes a FileArgument object with empty lists for 
        its three parameters.

        Attributes:
            argument (FileArgument): A FileArgument object containing three empty lists.
        """
        self.argument: FileArgument = FileArgument([], [], [])

    def add_response_data(self, filename: str, text: str) -> None:
        """
        Add response data to the argument.

        This method adds the given text data associated with a specific filename to the underlying argument object.

        Args:
            filename (str): The name of the file associated with the data.
            text (str): The text content to be added as data.

        Returns:
            None: This method doesn't return a value.
        """
        self.argument.add_data(filename, text)

    def load_txt_files(self, directory: str) -> None:
        """
        Load all .txt files from the specified directory and add them to the response data.

        This method iterates through each file in the given directory, reads the content of
        text files (.txt extension), and adds them to the response data using the filename
        as the key.

        Args:
            directory (str): Path to the directory containing text files to be loaded

        Returns:
            None

        Note:
            Only files with .txt extension will be processed.
            Prints a message for each successfully loaded file.
        """
        for file in os.listdir(directory):
            if file.endswith(".txt"):
                with open(os.path.join(directory, file), "r", encoding='utf-8') as fl:
                    text = fl.read()
                    self.add_response_data(file, text)
