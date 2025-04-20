import pytest
import numpy as np
from unittest.mock import patch
from vectoriz.files import FileArgument


def test_add_data_appends_to_lists():
    file_arg = FileArgument([], [], [])
    filename = "test.txt"
    text = "Test content"
    
    with patch.object(FileArgument, '_create_embedding', return_value=np.array([0.1, 0.2, 0.3])):
        file_arg.add_data(filename, text)
        
        assert file_arg.chunk_names == [filename]
        assert file_arg.text_list == [text]
        assert len(file_arg.embeddings) == 1
        np.testing.assert_array_equal(file_arg.embeddings[0], np.array([0.1, 0.2, 0.3]))

def test_add_data_multiple_entries():
    file_arg = FileArgument(["existing.txt"], ["existing content"], [np.array([0.5, 0.5, 0.5])])
    filename = "new.txt"
    text = "New content"
    
    with patch.object(FileArgument, '_create_embedding', return_value=np.array([0.7, 0.8, 0.9])):
        file_arg.add_data(filename, text)
        assert file_arg.chunk_names == ["existing.txt", "new.txt"]
        assert file_arg.text_list == ["existing content", "New content"]
        assert len(file_arg.embeddings) == 2
        np.testing.assert_array_equal(file_arg.embeddings[1], np.array([0.7, 0.8, 0.9]))

def test_add_data_calls_create_embedding():
    file_arg = FileArgument([], [], [])
    filename = "test.txt"
    text = "Test content"
    
    with patch.object(FileArgument, '_create_embedding') as mock_create_embedding:
        mock_create_embedding.return_value = np.array([0.1, 0.2, 0.3])
        file_arg.add_data(filename, text)
        mock_create_embedding.assert_called_once_with(text)
