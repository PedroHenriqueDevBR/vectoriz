# RAG-vector-creator

## Overview
This project implements a RAG (Retrieval-Augmented Generation) system for creating and managing vector embeddings from documents using FAISS and NumPy libraries. It efficiently transforms text data into high-dimensional vector representations that enable semantic search capabilities, similarity matching, and context-aware document retrieval for enhanced question answering applications.

## Features

- Document ingestion and preprocessing
- Vector embedding generation using state-of-the-art models
- Efficient storage and retrieval of embeddings
- Integration with LLM-based generation systems

## Installation

```bash
pip install -r requirements.txt
python app.py
```

## Build lib

To build the lib run the commands:

```
python setup.py sdist bdist_wheel
```

To test the install run:
```
pip install .
```

## License

MIT
