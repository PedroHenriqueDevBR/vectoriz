# Vectoriz

Vectoriz is available on PyPI and can be installed via pip:

<div align="center">
    <a href="https://pypi.org/project/vectoriz/"><img src="https://badge.fury.io/py/vectoriz.svg" alt="PyPI version"></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.12%2B-blue" alt="Python Version"></a>
    <a href="https://github.com/PedroHenriqueDevBR/vectoriz/issues"><img src="https://img.shields.io/github/issues/PedroHenriqueDevBR/vectoriz" alt="GitHub issues"></a>
    <a href="https://github.com/PedroHenriqueDevBR/vectoriz/stargazers"><img src="https://img.shields.io/github/stars/PedroHenriqueDevBR/vectoriz" alt="GitHub stars"></a>
    <a href="https://github.com/PedroHenriqueDevBR/vectoriz/network"><img src="https://img.shields.io/github/forks/PedroHenriqueDevBR/vectoriz" alt="GitHub forks"></a>
</div>

## Project description

For install another versions you can go to: <a href="https://pypi.org/project/vectoriz/#history">PyPI versions</a>

```bash
pip install vectoriz
```

A tool for generating vector embeddings for Retrieval-Augmented Generation (RAG) applications.

## Overview

This project provides utilities to create, manage, and optimize vector embeddings for use in RAG systems. It streamlines the process of converting documents and data sources into vector representations suitable for semantic search and retrieval.

## Features

- Document processing and chunking
- Vector embedding generation using various models
- Vector database integration
- Optimization tools for RAG performance
- Easy-to-use API for embedding creation

## Installation

```bash
git clone https://github.com/PedroHenriqueDevBR/vectoriz.git
cd vectoriz
pip install -r requirements.txt
```

## Usage

```python
# initial informations
index_db_path = "./data/faiss_db.index" # path to save/load index
np_db_path = "./data/np_db.npz" # path to save/load numpy data
directory_path = "/home/username/Documents/" # Path where the files (.txt, .docx) are saved

# Class instance
transformer = TokenTransformer()
files_features = FilesFeature()

# Load files and create a argument class (pack with embedings, chunk_names and text_list)
argument = files_features.load_all_files_from_directory(directory_path)

# Created FAISS index to be used in queries
token_data = transformer.create_index(argument.text_list)
index = token_data.index

# To load files from VectorDB use
vector_client = VectorDBClient()
vector_client.load_data(self.index_db_path, self.np_db_path)
index = vector_client.faiss_index
argument = vector_client.file_argument

# To save data on VectorDB use
vector_client = VectorDBClient(index, argument)
vector_client.save_data(index_db_path, np_db_path)

# To search information on index
query = input(">>> ")
amoount_content = 1
response = self.transformer.search(query, self.index, self.argument.text_list, amoount_content)
print(response)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.