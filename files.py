import os
from typing import Optional
import numpy as np
from token_transformer import TokenTransformer


class FileArgument:
    def __init__(self, filenames: list[str], text_list: list[str], embeddings: list[float], embeddings_np: Optional[np.ndarray] = None) -> None:
        self.filenames: list[str] = filenames
        self.text_list: list[str] = text_list
        self.embeddings: list[float] = embeddings
        self.embeddings_np: np.ndarray = embeddings_np
        self.transformer = TokenTransformer()
        
    def save(self, filename: str, text: str) -> None:
        self.filenames.append(filename)
        self.text_list.append(text)
        self.embeddings.append(self._create_embedding(text))
            
    def full_content(self) -> str:
        return '; '.join(self.text_list)
    
    def _create_embedding(self, text: str) -> np.ndarray:
        return self.transformer.transform([text])[0]


class FilesFeature:
    
    def __init__(self):
        self.argument: FileArgument = FileArgument([], [], [])
        
    def add_response_data(self, filename: str, text: str) -> None:
        self.argument.save(filename, text)
        
    def load_files(self, directory: str) -> None:
        for file in os.listdir(directory):
            if file.endswith('.txt'):
                with open(os.path.join(directory, file), 'r') as fl:
                    text = fl.read()
                    self.add_response_data(file, text)
                    print(f"Loaded file: {file}")