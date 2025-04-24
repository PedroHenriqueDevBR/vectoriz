import os
import docx
import numpy as np
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from vectoriz.token_transformer import TokenTransformer

class FileArgument:
    def __init__(
        self,
        chunk_names: list[str] = [],
        text_list: list[str] = [],
        embeddings: list[float] = [],
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
        print(f"Adding data for file: {filename}")
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
        instance = TokenTransformer()
        return instance.text_to_embeddings([text])[0]


class FilesFeature:

    def _extract_txt_content(self, directory: str, file: str) -> Optional[str]:
        """
        Extract content from a text file and add it to the response data.

        This method opens a text file in read mode with UTF-8 encoding, reads its content,
        and adds the file name and its content to the response data.

        Parameters:
        ----------
        directory : str
            The directory path where the file is located.
        file : str
            The name of the text file to read.

        Returns:
        -------
        None
            This method doesn't return any value but updates the internal response data.

        Raises:
        ------
        FileNotFoundError
            If the specified file does not exist.
        UnicodeDecodeError
            If the file cannot be decoded using UTF-8 encoding.
        """
        with open(os.path.join(directory, file), "r", encoding="utf-8") as fl:
            text = fl.read()
            return text
        
    def _extract_markdown_content(self, path: str) -> dict[str, str]:
        """
        Extract content from a Markdown file and add it to the response data.

        This method opens a Markdown file in read mode with UTF-8 encoding, reads its content,
        and adds the file name and its content to the response data.

        Parameters:
        ----------
        path : str
            The name of the Markdown file to read.

        Returns:
        -------
        Optional[str]
            The content of the Markdown file or None if the file is empty.

        Raises:
        ------
        FileNotFoundError
            If the specified file does not exist.
        UnicodeDecodeError
            If the file cannot be decoded using UTF-8 encoding.
        """
        file = os.path.basename(path)
        with open(path, "r", encoding="utf-8") as fl:
            content = fl.read()
        return {"file": file, "content": content}

    def _extract_docx_content(self, directory: str, file: str) -> Optional[str]:
        """
        Extracts text content from a Microsoft Word document.
        This method opens a Word document, reads all paragraphs, and joins non-empty
        paragraphs into a single text string. The extracted content is then stored
        using the add_response_data method.
        Args:
            directory (str): The directory path where the Word file is located
            file (str): The filename of the Word document to process
        Returns:
            Optional[str]: The extracted text content or None if no content is found.
        Note:
            Empty paragraphs (those that contain only whitespace) are skipped.
            The python-docx library is required for this method to work.
        """
        file_path = os.path.join(directory, file)
        doc = docx.Document(file_path)
        full_text = []

        for paragraph in doc.paragraphs:
            content = paragraph.text.strip()
            if len(content) == 0:
                continue
            full_text.append(paragraph.text)
        return "\n".join(full_text)

    def load_txt_files_from_directory(self, directory: str, verbose: bool = False) -> FileArgument:
        """
        Load all text files from the specified directory and extract their content.
        This method scans the specified directory for files with the '.txt' extension
        and processes each of them using the extract_txt_content method.
        Parameters:
        ----------
        directory : str
            Path to the directory containing text files to be loaded.
        Returns:
        -------
        None
            This method does not return any value. It updates the internal state
            by processing text files found in the directory.
        """
        argument: FileArgument = FileArgument([], [], [])
        for file in os.listdir(directory):
            if not file.endswith(".txt"):
                if verbose:
                    print(f"Error file: {file}")
                continue
            
            text = self._extract_txt_content(directory, file)
            if text is None:
                if verbose:
                    print(f"Error file: {file}")
                continue
            
            argument.add_data(file, text)
            if verbose:
                    print(f"Loaded txt file: {file}")
        return argument

    def load_docx_files_from_directory(self, directory: str, verbose: bool = False) -> FileArgument:
        """
        Load all Word (.docx) files from the specified directory and extract their content.

        This method iterates through all files in the given directory, identifies those
        with a .docx extension, and processes them using the extract_word_content method.

        Args:
            directory (str): Path to the directory containing Word files to be processed

        Returns:
            None

        Examples:
            >>> processor = DocumentProcessor()
            >>> processor.load_word_files("/path/to/documents")
        """
        argument: FileArgument = FileArgument([], [], [])
        for file in os.listdir(directory):
            if not file.endswith(".docx"):
                if verbose:
                    print(f"Error file: {file}")
                continue
            
            text = self._extract_docx_content(directory, file)
            if text is None:
                if verbose:
                    print(f"Error file: {file}")
                continue
            
            argument.add_data(file, text)
            if verbose:
                print(f"Loaded Word file: {file}")
        return argument
    
    def load_markdown_files_from_directory(self, directory: str, verbose: bool = False) -> FileArgument:
        """
        Load all Markdown (.md) files from the specified directory and extract their content.

        This method iterates through all files in the given directory, identifies those
        with a .md extension, and processes them using the extract_markdown_content method.

        Args:
            directory (str): Path to the directory containing Markdown files to be processed

        Returns:
            None

        Examples:
            >>> processor = DocumentProcessor()
            >>> processor.load_markdown_files("/path/to/documents")
        """
        argument = FileArgument()
        paths = [
            os.path.join(directory, file)
            for file in os.listdir(directory)
            if file.endswith(".md")
        ]
        
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self._extract_markdown_content, paths))
            
        with ThreadPoolExecutor() as executor:
            executor.map(
                lambda result: argument.add_data(result.get("file"), result.get("content")),
                results,
            )

        return argument

    def load_all_files_from_directory(self, directory: str, verbose: bool =  False) -> FileArgument:
        """
        Load all supported files (.txt and .docx) from the specified directory and its subdirectories.

        This method walks through the directory tree, processing all text and Word files
        by adding them to the response data.

        Args:
            directory (str): Path to the directory containing files to be loaded

        Returns:
            None
        """
        argument: FileArgument = FileArgument([], [], [])
        for root, _, files in os.walk(directory):
            for file in files:
                readed = False
                if file.endswith(".txt"):
                    text = self._extract_txt_content(root, file)
                    if text is not None:
                        argument.add_data(file, text)
                        readed = True
                elif file.endswith(".docx"):
                    try:
                        text = self._extract_docx_content(root, file)
                        if text is not None:
                            argument.add_data(file, text)
                            readed = True
                    except Exception as e:
                        print(f"Error processing {file}: {str(e)}")

                if verbose and readed:
                    print(f"Loaded file: {file}")
                elif verbose and not readed:
                    print(f"Error file: {file}")
        
        return argument
