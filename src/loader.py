import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split_docs(docs_dir="docs"):
    """
    Lê os PDFs da pasta fornecida e divide o conteúdo em pequenos blocos de texto (chunks).
    """
    # Garante que a pasta existe (se não existir, não falha ao tentar carregar)
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
        
    print(f"Lendo PDFs da pasta '{docs_dir}'...")
    
    # Carrega os PDFs do diretório especificado
    loader = PyPDFDirectoryLoader(docs_dir)
    documents = loader.load()
    
    # Configura o divisor de texto conforme a especificação do projeto
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80
    )
    
    # Divide os documentos em pedaços menores (chunks) para processamento
    chunks = text_splitter.split_documents(documents)
    
    # Imprime os resultados no terminal
    print(f"Total de documentos carregados: {len(documents)}")
    print(f"Total de chunks (pedaços) gerados: {len(chunks)}")
    
    return chunks

if __name__ == "__main__":
    # Teste local: permite rodar `python src/loader.py` para visualizar o processo.
    load_and_split_docs()
