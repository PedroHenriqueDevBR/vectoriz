import pytest
import faiss
import numpy as np
from unittest.mock import Mock
from vectoriz.token_transformer import TokenData, TokenTransformer


class TestTokenData:

    def test_from_vector_db(self):
        mock_vector_data = Mock()

        mock_index = faiss.IndexFlatL2(5)
        mock_embeddings = np.random.random((3, 5)).astype("float32")
        mock_texts = ["text1", "text2", "text3"]

        mock_file_argument = Mock()
        mock_file_argument.embeddings = mock_embeddings
        mock_file_argument.text_list = mock_texts
        mock_vector_data.faiss_index = mock_index
        mock_vector_data.file_argument = mock_file_argument
        token_data = TokenData.from_vector_db(mock_vector_data)

        assert token_data.texts == mock_texts
        assert token_data.index == mock_index
        assert np.array_equal(token_data.embeddings, mock_embeddings)
        assert isinstance(token_data, TokenData)

    def test_from_file_argument(self):
        mock_file_argument = Mock()
        mock_file_argument.embeddings = np.random.random((3, 5)).astype("float32")
        mock_file_argument.text_list = ["text1", "text2", "text3"]
        mock_index = faiss.IndexFlatL2(5)
        token_data = TokenData.from_file_argument(mock_file_argument, mock_index)

        assert token_data.texts == mock_file_argument.text_list
        assert token_data.index == mock_index
        assert np.array_equal(token_data.embeddings, mock_file_argument.embeddings)
        assert isinstance(token_data, TokenData)


class TestTokenTransformer:

    def test_text_to_embeddings(self):
        transformer = TokenTransformer()
        sentences = ["This is a test sentence.", "Another test sentence."]
        embeddings = transformer.text_to_embeddings(sentences)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(sentences)
        assert embeddings.shape[1] > 0

    def test_text_to_embeddings_empty_list(self):
        transformer = TokenTransformer()
        sentences = []
        embeddings = transformer.text_to_embeddings(sentences)

        assert isinstance(embeddings, np.ndarray)

    def test_get_np_vectors(self):
        transformer = TokenTransformer()
        embeddings = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        result = transformer.get_np_vectors(embeddings)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (2, 3)
        assert np.array_equal(
            result, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        )

    def test_get_np_vectors_empty_list(self):
        transformer = TokenTransformer()
        embeddings = []
        result = transformer.get_np_vectors(embeddings)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (0,)

    def test_get_np_vectors_single_element(self):
        transformer = TokenTransformer()
        embeddings = [[1.5, 2.5]]
        result = transformer.get_np_vectors(embeddings)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (1, 2)
        assert np.array_equal(result, np.array([[1.5, 2.5]], dtype=np.float32))

    def test_query_to_embeddings(self):
        transformer = TokenTransformer()
        query = "This is a test query"
        result = transformer._query_to_embeddings(query)

        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 1
        assert result.shape[1] == 384

    def test_query_to_embeddings_empty_string(self):
        transformer = TokenTransformer()
        query = ""
        result = transformer._query_to_embeddings(query)

        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 1
        assert result.shape[1] == 384

    def test_query_to_embeddings_returns_correct_shape(self):
        transformer = TokenTransformer()
        query1 = "First query"
        query2 = "Second query with more words"
        result1 = transformer._query_to_embeddings(query1)
        result2 = transformer._query_to_embeddings(query2)

        assert result1.shape == result2.shape
        assert len(result1.shape) == 2
        assert result1.shape[0] == 1

    def test_search(self):
        transformer = TokenTransformer()
        texts = ["First document", "Second document", "Third document"]

        embeddings = transformer.text_to_embeddings(texts)
        index = transformer.embeddings_to_index(embeddings)

        result = transformer.search("first", index, texts, context_amount=1)
        assert isinstance(result, str)
        assert "First document" in result

        result = transformer.search("document", index, texts, context_amount=2)
        assert isinstance(result, str)
        assert len(result.strip().split("\n")) == 2

    def test_search_with_empty_texts(self):
        transformer = TokenTransformer()
        texts = []

        if texts:
            embeddings = transformer.text_to_embeddings(texts)
            index = transformer.embeddings_to_index(embeddings)
        else:
            index = faiss.IndexFlatL2(384)

        result = transformer.search("query", index, texts)
        assert result == ""

    def test_search_with_different_context_amounts(self):
        transformer = TokenTransformer()
        texts = ["Doc 1", "Doc 2", "Doc 3", "Doc 4", "Doc 5"]

        embeddings = transformer.text_to_embeddings(texts)
        index = transformer.embeddings_to_index(embeddings)

        result1 = transformer.search("Doc", index, texts, context_amount=1)
        result3 = transformer.search("Doc", index, texts, context_amount=3)
        result5 = transformer.search("Doc", index, texts, context_amount=5)

        assert len(result1.strip().split("\n")) == 1
        assert len(result3.strip().split("\n")) == 3
        assert len(result5.strip().split("\n")) == 5
        
    def test_search_with_list_arg(self):
        transformer = TokenTransformer()
        texts = ["Doc 1", "Doc 2", "Doc 3", "Doc 4", "Doc 5"]

        embeddings = transformer.text_to_embeddings(texts)
        index = transformer.embeddings_to_index(embeddings)

        result1 = transformer.search("Doc", index, texts, context_amount=1, as_list=True)
        result3 = transformer.search("Doc", index, texts, context_amount=3, as_list=True)
        result5 = transformer.search("Doc", index, texts, context_amount=5, as_list=True)

        assert len(result1) == 1
        assert len(result3) == 3
        assert len(result5) == 5

    def test_create_index(self):
        transformer = TokenTransformer()
        texts = ["First document", "Second document", "Third document"]

        token_data = transformer.create_index(texts)

        assert isinstance(token_data, TokenData)
        assert token_data.texts == texts
        assert isinstance(token_data.index, faiss.IndexFlatL2)
        assert isinstance(token_data.embeddings, np.ndarray)
        assert token_data.embeddings.shape[0] == len(texts)
        assert token_data.embeddings.shape[1] == 384

    def test_create_index_empty_list(self):
        transformer = TokenTransformer()
        texts = []

        with pytest.raises(ValueError, match="The input texts list is empty."):
            transformer.create_index(texts)

    def test_create_index_single_element(self):
        transformer = TokenTransformer()
        texts = ["Single document text"]

        token_data = transformer.create_index(texts)

        assert isinstance(token_data, TokenData)
        assert token_data.texts == texts
        assert isinstance(token_data.index, faiss.IndexFlatL2)
        assert isinstance(token_data.embeddings, np.ndarray)
        assert token_data.embeddings.shape == (1, 384)
