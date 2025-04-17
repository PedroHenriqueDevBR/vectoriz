from vectoriz.files import FileArgument
import unittest
import numpy as np
from unittest.mock import patch, MagicMock


class TestFileArgumentTestCase(unittest.TestCase):
    
    def setUp(self):
        self.file_arg = FileArgument([], [], [], None)
    
    @patch('vectoriz.files.FileArgument._create_embedding')
    def test_add_data_appends_to_lists(self, mock_create_embedding):
        """Test that add_data correctly appends to the internal lists."""
        # Setup
        mock_create_embedding.return_value = np.array([0.1, 0.2, 0.3])
        
        # Execute
        self.file_arg.add_data("test_file.txt", "This is test content")
        
        # Assert
        self.assertEqual(self.file_arg.chunk_names, ["test_file.txt"])
        self.assertEqual(self.file_arg.text_list, ["This is test content"])
        self.assertEqual(len(self.file_arg.embeddings), 1)
        np.testing.assert_array_equal(self.file_arg.embeddings[0], np.array([0.1, 0.2, 0.3]))
        mock_create_embedding.assert_called_once_with("This is test content")
    
    @patch('vectoriz.files.FileArgument._create_embedding')
    def test_add_data_multiple_entries(self, mock_create_embedding):
        """Test that add_data correctly handles multiple entries."""
        # Setup
        mock_create_embedding.side_effect = [
            np.array([0.1, 0.2, 0.3]),
            np.array([0.4, 0.5, 0.6])
        ]
        
        # Execute
        self.file_arg.add_data("file1.txt", "Content 1")
        self.file_arg.add_data("file2.txt", "Content 2")
        
        # Assert
        self.assertEqual(self.file_arg.chunk_names, ["file1.txt", "file2.txt"])
        self.assertEqual(self.file_arg.text_list, ["Content 1", "Content 2"])
        self.assertEqual(len(self.file_arg.embeddings), 2)
        np.testing.assert_array_equal(self.file_arg.embeddings[0], np.array([0.1, 0.2, 0.3]))
        np.testing.assert_array_equal(self.file_arg.embeddings[1], np.array([0.4, 0.5, 0.6]))


if __name__ == '__main__':
    unittest.main()
