import os
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
        
    print("Inicializando o modelo de embeddings...")
    # Configura o modelo de embeddings, otimizado para vários idiomas (incluindo Português).
    # Este modelo roda localmente de forma gratuita.
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    print("Criando o banco de vetores ChromaDB e armazenando os embeddings...")
    # Cria a base de dados Chroma e armazena de forma persistente na pasta 'chroma_db'
    # Ao especificar o persist_directory, o ChromaDB salva os vetores automaticamente no disco.
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
