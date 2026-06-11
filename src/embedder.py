import os
import shutil
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from loader import load_and_split_docs

def create_vector_store():
    """
    Gera embeddings para os chunks de texto e os salva no banco de dados vetorial ChromaDB.
    """
    # Obtém os chunks de texto carregados e divididos pelo loader.py
    chunks = load_and_split_docs()

    if not chunks:
        print("Nenhum chunk gerado. Adicione PDFs na pasta 'docs/' para processar.")
        return None

    # Remove banco existente para evitar duplicatas ao reprocessar
    if os.path.exists("chroma_db"):
        shutil.rmtree("chroma_db")

    print("Inicializando o modelo de embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    print("Criando o banco de vetores ChromaDB e armazenando os embeddings...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    
    print(f"Banco de dados criado com sucesso. {len(chunks)} vetores foram armazenados em 'chroma_db/'.")
    
    return vectorstore

if __name__ == "__main__":
    # Roda a função principal quando o script for executado
    create_vector_store()
