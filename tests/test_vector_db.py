import os
import pytest
import numpy as np
import faiss
from unittest.mock import patch, MagicMock
from vectoriz.vector_db import VectorDB, VectorDBClient
from vectoriz.files import FileArgument


class TestVectorDB:

    def test_load_saved_data_successful(self, tmp_path):
        vector_db = VectorDB()
        faiss_path = str(tmp_path / "test.index")
        np_path = str(tmp_path / "test.npz")

        mock_index = MagicMock(spec=faiss.IndexFlatL2)
        mock_file_argument = MagicMock(spec=FileArgument)

        with patch.object(
            vector_db, "load_faiss_index", return_value=mock_index
        ) as mock_load_index:
            with patch.object(
                vector_db, "load_numpy_embeddings", return_value=mock_file_argument
            ) as mock_load_embeddings:

                result = vector_db.load_saved_data(faiss_path, np_path)

                assert isinstance(result, VectorDBClient)
                assert result.faiss_index == mock_index
                assert result.file_argument == mock_file_argument
                mock_load_index.assert_called_once_with(faiss_path)
                mock_load_embeddings.assert_called_once_with(np_path)

    def test_load_saved_data_missing_faiss_index(self, tmp_path):
        vector_db = VectorDB()
        faiss_path = str(tmp_path / "nonexistent.index")
        np_path = str(tmp_path / "test.npz")

        mock_file_argument = MagicMock(spec=FileArgument)

        with patch.object(
            vector_db, "load_faiss_index", return_value=None
        ) as mock_load_index:
            with patch.object(
                vector_db, "load_numpy_embeddings", return_value=mock_file_argument
            ) as mock_load_embeddings:

                result = vector_db.load_saved_data(faiss_path, np_path)

                assert result is None
                mock_load_index.assert_called_once_with(faiss_path)
                mock_load_embeddings.assert_called_once_with(np_path)

    def test_load_saved_data_missing_numpy_embeddings(self, tmp_path):
        vector_db = VectorDB()
        faiss_path = str(tmp_path / "test.index")
        np_path = str(tmp_path / "nonexistent.npz")

        mock_index = MagicMock(spec=faiss.IndexFlatL2)

        with patch.object(
            vector_db, "load_faiss_index", return_value=mock_index
        ) as mock_load_index:
            with patch.object(
                vector_db, "load_numpy_embeddings", return_value=None
            ) as mock_load_embeddings:

                result = vector_db.load_saved_data(faiss_path, np_path)

                assert result is None
                mock_load_index.assert_called_once_with(faiss_path)
                mock_load_embeddings.assert_called_once_with(np_path)

    def test_load_saved_data_both_missing(self, tmp_path):
        vector_db = VectorDB()
        faiss_path = str(tmp_path / "nonexistent.index")
        np_path = str(tmp_path / "nonexistent.npz")

        with patch.object(
            vector_db, "load_faiss_index", return_value=None
        ) as mock_load_index:
            with patch.object(
                vector_db, "load_numpy_embeddings", return_value=None
            ) as mock_load_embeddings:

                result = vector_db.load_saved_data(faiss_path, np_path)

                assert result is None
                mock_load_index.assert_called_once_with(faiss_path)
                mock_load_embeddings.assert_called_once_with(np_path)

    def test_save_faiss_index(self, tmp_path):
        vector_db = VectorDB()
        mock_index = MagicMock(spec=faiss.IndexFlatL2)
        faiss_path = str(tmp_path / "test")

        with patch("faiss.write_index") as mock_write_index:
            vector_db.save_faiss_index(mock_index, faiss_path)

            mock_write_index.assert_called_once_with(mock_index, faiss_path + ".index")

    def test_save_faiss_index_with_extension(self, tmp_path):
        vector_db = VectorDB()
        mock_index = MagicMock(spec=faiss.IndexFlatL2)
        faiss_path = str(tmp_path / "test.index")

        with patch("faiss.write_index") as mock_write_index:
            vector_db.save_faiss_index(mock_index, faiss_path)

            mock_write_index.assert_called_once_with(mock_index, faiss_path)

    def test_save_faiss_index_integration(self, tmp_path):
        vector_db = VectorDB()
        dimension = 128
        index = faiss.IndexFlatL2(dimension)

        sample_vectors = np.random.random((10, dimension)).astype("float32")
        index.add(sample_vectors)
        faiss_path = str(tmp_path / "test.index")

        vector_db.save_faiss_index(index, faiss_path)

        assert os.path.exists(faiss_path)

        loaded_index = faiss.read_index(faiss_path)
        assert loaded_index.ntotal == index.ntotal

    def test_load_faiss_index_successful(self, tmp_path):
        vector_db = VectorDB()
        dimension = 128
        index = faiss.IndexFlatL2(dimension)

        sample_vectors = np.random.random((5, dimension)).astype("float32")
        index.add(sample_vectors)
        faiss_path = str(tmp_path / "test.index")

        faiss.write_index(index, faiss_path)

        result = vector_db.load_faiss_index(faiss_path)

        assert result is not None
        assert result.ntotal == index.ntotal
        assert result.d == index.d

    def test_load_faiss_index_missing_file(self, tmp_path):
        vector_db = VectorDB()
        nonexistent_path = str(tmp_path / "nonexistent.index")

        result = vector_db.load_faiss_index(nonexistent_path)

        assert result is None

    def test_load_faiss_index_with_mock(self):
        vector_db = VectorDB()
        test_path = "/mock/path/test.index"
        mock_index = MagicMock(spec=faiss.IndexFlatL2)

        with patch("os.path.exists", return_value=True) as mock_exists:
            with patch("faiss.read_index", return_value=mock_index) as mock_read:
                result = vector_db.load_faiss_index(test_path)

                mock_exists.assert_called_once_with(test_path)
                mock_read.assert_called_once_with(test_path)
                assert result == mock_index

    def test_save_numpy_embeddings_with_ndarray(self, tmp_path):
        vector_db = VectorDB()
        np_path = str(tmp_path / "test")

        embeddings_np = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        chunk_names = np.array(["chunk1", "chunk2"])
        texts = np.array(["text1", "text2"])

        file_arg = FileArgument(
            chunk_names=chunk_names,
            text_list=texts,
            embeddings=[],
            ndarray_data=embeddings_np,
        )

        with patch("numpy.savez") as mock_savez:
            vector_db.save_numpy_embeddings(file_arg, np_path)

            mock_savez.assert_called_once_with(
                np_path + ".npz",
                embeddings=embeddings_np,
                chunk_names=chunk_names,
                texts=texts,
            )

    def test_save_numpy_embeddings_with_embeddings_list(self, tmp_path):
        vector_db = VectorDB()
        np_path = str(tmp_path / "test")

        embeddings_list = [[[0.1, 0.2]], [[0.3, 0.4]]]
        chunk_names = np.array(["chunk1", "chunk2"])
        texts = np.array(["text1", "text2"])

        file_arg = FileArgument(
            chunk_names=chunk_names,
            text_list=texts,
            embeddings=embeddings_list,
            ndarray_data=None,
        )

        transformed_np = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)

        with patch(
            "vectoriz.token_transformer.TokenTransformer.get_np_vectors",
            return_value=transformed_np,
        ) as mock_transform:
            with patch("numpy.savez") as mock_savez:
                vector_db.save_numpy_embeddings(file_arg, np_path)

                mock_transform.assert_called_once_with(embeddings_list)
                mock_savez.assert_called_once_with(
                    np_path + ".npz",
                    embeddings=transformed_np,
                    chunk_names=chunk_names,
                    texts=texts,
                )

    def test_save_numpy_embeddings_with_extension(self, tmp_path):
        vector_db = VectorDB()
        np_path = str(tmp_path / "test.npz")

        embeddings_np = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        chunk_names = np.array(["chunk1", "chunk2"])
        texts = np.array(["text1", "text2"])

        file_arg = FileArgument(
            chunk_names=chunk_names,
            text_list=texts,
            embeddings=[],
            ndarray_data=embeddings_np,
        )

        with patch("numpy.savez") as mock_savez:
            vector_db.save_numpy_embeddings(file_arg, np_path)

            mock_savez.assert_called_once_with(
                np_path, embeddings=embeddings_np, chunk_names=chunk_names, texts=texts
            )

    def test_save_numpy_embeddings_integration(self, tmp_path):
        vector_db = VectorDB()
        np_path = str(tmp_path / "test.npz")

        embeddings_np = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        chunk_names = np.array(["chunk1", "chunk2"])
        texts = np.array(["text1", "text2"])

        file_arg = FileArgument(
            chunk_names=chunk_names,
            text_list=texts,
            embeddings=[],
            ndarray_data=embeddings_np,
        )

        vector_db.save_numpy_embeddings(file_arg, np_path)

        assert os.path.exists(np_path)
        loaded_data = np.load(np_path)
        assert "embeddings" in loaded_data
        assert "chunk_names" in loaded_data
        assert "texts" in loaded_data
        np.testing.assert_array_equal(loaded_data["embeddings"], embeddings_np)
        np.testing.assert_array_equal(loaded_data["chunk_names"], chunk_names)
        np.testing.assert_array_equal(loaded_data["texts"], texts)

    def test_load_numpy_embeddings_successful(self, tmp_path):
        vector_db = VectorDB()
        np_path = str(tmp_path / "test.npz")

        embeddings_np = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        chunk_names = np.array(["chunk1", "chunk2"])
        texts = np.array(["text1", "text2"])

        np.savez(
            np_path, embeddings=embeddings_np, chunk_names=chunk_names, texts=texts
        )

        result = vector_db.load_numpy_embeddings(np_path)

        assert result is not None
        np.testing.assert_array_equal(result.ndarray_data, embeddings_np)
        np.testing.assert_array_equal(result.chunk_names, chunk_names)
        np.testing.assert_array_equal(result.text_list, texts)
        assert result.embeddings == []

    def test_load_numpy_embeddings_missing_file(self, tmp_path):
        vector_db = VectorDB()
        nonexistent_path = str(tmp_path / "nonexistent.npz")

        result = vector_db.load_numpy_embeddings(nonexistent_path)

        assert result is None

    def test_load_numpy_embeddings_with_mock(self):
        vector_db = VectorDB()
        test_path = "/mock/path/test.npz"

        mock_data = {
            "embeddings": np.array([[0.1, 0.2], [0.3, 0.4]]),
            "chunk_names": np.array(["name1", "name2"]),
            "texts": np.array(["text1", "text2"]),
        }

        with patch("os.path.exists", return_value=True) as mock_exists:
            with patch("numpy.load", return_value=mock_data) as mock_load:
                result = vector_db.load_numpy_embeddings(test_path)

                mock_exists.assert_called_once_with(test_path)
                mock_load.assert_called_once_with(test_path)
                assert isinstance(result, FileArgument)
                np.testing.assert_array_equal(
                    result.ndarray_data, mock_data["embeddings"]
                )
                np.testing.assert_array_equal(
                    result.chunk_names, mock_data["chunk_names"]
                )
                np.testing.assert_array_equal(result.text_list, mock_data["texts"])


class TestVectorDBClient:
    def test_save_data_successful(self, tmp_path):
        # Setup
        mock_index = MagicMock(spec=faiss.IndexFlatL2)
        mock_file_argument = MagicMock(spec=FileArgument)
        client = VectorDBClient(mock_index, mock_file_argument)
        
        faiss_path = str(tmp_path / "test")
        np_path = str(tmp_path / "test")
        
        # Test with mocks to verify function calls
        with patch.object(VectorDB, 'save_faiss_index') as mock_save_faiss:
            with patch.object(VectorDB, 'save_numpy_embeddings') as mock_save_numpy:
                client.save_data(faiss_path, np_path)
                
                mock_save_faiss.assert_called_once_with(mock_index, faiss_path)
                mock_save_numpy.assert_called_once_with(mock_file_argument, np_path)
    
    def test_save_data_not_initialized(self):
        # Test with None values
        client1 = VectorDBClient(None, MagicMock(spec=FileArgument))
        client2 = VectorDBClient(MagicMock(spec=faiss.IndexFlatL2), None)
        client3 = VectorDBClient(None, None)
        
        with pytest.raises(ValueError, match="FAISS index or file argument is not initialized."):
            client1.save_data("test.index", "test.npz")
            
        with pytest.raises(ValueError, match="FAISS index or file argument is not initialized."):
            client2.save_data("test.index", "test.npz")
            
        with pytest.raises(ValueError, match="FAISS index or file argument is not initialized."):
            client3.save_data("test.index", "test.npz")
            
    def test_save_data_integration(self, tmp_path):
        # Setup real objects for integration test
        dimension = 128
        index = faiss.IndexFlatL2(dimension)
        
        # Add some vectors to the index
        sample_vectors = np.random.random((5, dimension)).astype("float32")
        index.add(sample_vectors)
        
        # Create file argument
        chunk_names = np.array(["chunk1", "chunk2", "chunk3", "chunk4", "chunk5"])
        texts = np.array(["text1", "text2", "text3", "text4", "text5"])
        file_arg = FileArgument(
            chunk_names=chunk_names,
            text_list=texts,
            embeddings=[],
            ndarray_data=sample_vectors
        )
        
        # Create client
        client = VectorDBClient(index, file_arg)
        
        # Define paths
        faiss_path = str(tmp_path / "test.index")
        np_path = str(tmp_path / "test.npz")
        
        # Save the data
        client.save_data(faiss_path, np_path)
        
        # Verify files were created
        assert os.path.exists(faiss_path)
        assert os.path.exists(np_path)
        
        # Load and verify the data
        loaded_index = faiss.read_index(faiss_path)
        loaded_data = np.load(np_path)
        
        assert loaded_index.ntotal == index.ntotal
        assert loaded_index.d == index.d
        assert "embeddings" in loaded_data
        assert "chunk_names" in loaded_data
        assert "texts" in loaded_data
        np.testing.assert_array_equal(loaded_data["embeddings"], sample_vectors)
        np.testing.assert_array_equal(loaded_data["chunk_names"], chunk_names)
        np.testing.assert_array_equal(loaded_data["texts"], texts)
    
    def test_load_data_successful(self, tmp_path):
        # Setup - create data files first
        dimension = 128
        index = faiss.IndexFlatL2(dimension)
        sample_vectors = np.random.random((5, dimension)).astype("float32")
        index.add(sample_vectors)
        
        chunk_names = np.array(["chunk1", "chunk2", "chunk3", "chunk4", "chunk5"])
        texts = np.array(["text1", "text2", "text3", "text4", "text5"])
        
        # Save test data
        faiss_path = str(tmp_path / "test.index")
        np_path = str(tmp_path / "test.npz")
        faiss.write_index(index, faiss_path)
        np.savez(np_path, embeddings=sample_vectors, chunk_names=chunk_names, texts=texts)
        
        # Create client and load data
        client = VectorDBClient()
        client.load_data(faiss_path, np_path)
        
        # Verify data was loaded correctly
        assert client.faiss_index is not None
        assert client.file_argument is not None
        assert client.faiss_index.ntotal == 5
        assert client.faiss_index.d == dimension
        np.testing.assert_array_equal(client.file_argument.ndarray_data, sample_vectors)
        np.testing.assert_array_equal(client.file_argument.chunk_names, chunk_names)
        np.testing.assert_array_equal(client.file_argument.text_list, texts)

    def test_load_data_with_mocks(self):
        client = VectorDBClient()
        mock_index = MagicMock(spec=faiss.IndexFlatL2)
        mock_file_arg = MagicMock(spec=FileArgument)
        
        with patch.object(VectorDB, 'load_faiss_index', return_value=mock_index) as mock_load_faiss:
            with patch.object(VectorDB, 'load_numpy_embeddings', return_value=mock_file_arg) as mock_load_numpy:
                client.load_data("test.index", "test.npz")
                
                mock_load_faiss.assert_called_once_with("test.index")
                mock_load_numpy.assert_called_once_with("test.npz")
                assert client.faiss_index == mock_index
                assert client.file_argument == mock_file_arg

    def test_load_data_missing_files(self, tmp_path):
        client = VectorDBClient()
        nonexistent_faiss = str(tmp_path / "nonexistent.index")
        nonexistent_np = str(tmp_path / "nonexistent.npz")
        
        # Test loading non-existent files
        client.load_data(nonexistent_faiss, nonexistent_np)
        
        # When files don't exist, the values should be None
        assert client.faiss_index is None
        assert client.file_argument is None

    def test_load_data_partial_missing(self, tmp_path):
        # Setup - create only one file
        dimension = 128
        index = faiss.IndexFlatL2(dimension)
        faiss_path = str(tmp_path / "test.index")
        nonexistent_np = str(tmp_path / "nonexistent.npz")
        faiss.write_index(index, faiss_path)
        
        # Test loading with one missing file
        client = VectorDBClient()
        client.load_data(faiss_path, nonexistent_np)
        
        # Verify partial loading
        assert client.faiss_index is not None
        assert client.file_argument is None
