import sys, os
import faiss
from typing import Optional
from files import FilesFeature
from vector_db import VectorDB
from token_transformer import TokenTransformer
from ia_model import IaModel

class App:
    def __init__(self):
        self.file_features = FilesFeature()
        self.token_transformer = TokenTransformer()
        self.vector_db = VectorDB()
        self.file_argument = None
        self.index: faiss.IndexFlatL2 = None
        self.ia_model = IaModel()

    def run(self):
        menu = '''
        1. Load files from directory
        2. Search
        0. Exit
        '''
        
        if self.load_db_data():
            print("Database loaded successfully!")
            print("- " * 20)
        else:
            print("Failed to load database.")
            print("- " * 20)
        
        while True:
            print(menu)
            choice = input("Select an option: ")
            if choice == '1':
                self.load_files()
            elif choice == '2':
                self.search()
            elif choice == '0':
                print("Exiting...")
                sys.exit(0)
            else:
                print("Invalid option. Please try again.")


    def search(self):
        query = input("Query: ")
        if not self.file_argument:
            print("No files loaded. Please load files first.")
            return

        context = self.token_transformer.search(query, self.file_argument.text_list, self.index, depth=1)
        generate_ia_response = input("Generate response? (y/n): ")
        if generate_ia_response.lower() == 'y':
            question = f'Baseado nas informações a seguir, {context}, responda a seguinte pergunta: {query}'
            print(f"Question: {question}")
            ia_response = self.ia_model.generate(question)
            print('- ' * 20)
            print(f"IA Response: {ia_response}")
        
    def load_db_data(self) -> bool:
        index = self.vector_db.load_faiss_index()
        if index is not None:
            self.index = index

        np_data = self.vector_db.load_np_data()
        if np_data is not None:
            if index is None:
                self.index = self.vector_db.create_index(np_data.embeddings_np)
            self.file_argument = np_data
            return True

        return False

    def load_files(self):
        directory_path = input("Directory path: ")
        self.file_features.load_files(directory_path)

        if self.file_features.argument:
            self.file_argument = self.file_features.argument
            print("Files loaded successfully!")

        embeddings_np = self.vector_db.save_np_index(self.file_argument)
        self.file_argument.embeddings_np = embeddings_np
        self.index = self.vector_db.create_index(self.file_argument.embeddings_np)        
        print("Index created successfully!")
        
if __name__ == "__main__":
    app = App()
    app.run()
