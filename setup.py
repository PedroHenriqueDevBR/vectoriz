from setuptools import setup, find_packages

setup(
    name="vectoriz",
    version="0.0.3",
    author="PedroHenriqueDevBR",
    author_email="pedro.henrique.particular@gmail.com",
    description="Python library for creating vectorized data from text or files.",
    long_description = open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/PedroHenriqueDevBR/vectoriz",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",
    install_requires=[
        "faiss-cpu==1.10.0",
        "numpy==2.2.4",
        "sentence-transformers==4.0.2",
        "python-docx==1.1.2"
    ],
)