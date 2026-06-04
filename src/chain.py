import os
from dotenv import load_dotenv
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Carrega as variáveis do arquivo .env (como a GOOGLE_API_KEY) para a memória
load_dotenv()

def get_qa_chain(provider="gemini"):
    """
    Configura a cadeia (chain) de perguntas e respostas RAG, interligando
    o banco de dados ChromaDB com o modelo selecionado (Gemini ou GPT-4o-mini) via um Prompt customizado.
    """
    
    # Precisamos do mesmo modelo de embeddings para buscar as respostas corretas
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    # Carrega o banco de vetores ChromaDB persistido anteriormente no diretório 'chroma_db'
    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )
    
    # Configura o retriever (recuperador) para trazer os 4 pedaços mais parecidos
    # com a pergunta do usuário
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # Instancia o LLM com base no provedor selecionado
    if provider == "gemini":
        if "GOOGLE_API_KEY" not in os.environ:
            raise ValueError("A variável GOOGLE_API_KEY não foi encontrada. Configure seu arquivo .env.")
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("A variável OPENAI_API_KEY não foi encontrada. Configure seu arquivo .env.")
        llm = ChatOpenAI(
            model="gpt-4o-mini-2024-07-18",
            temperature=0
        )
    else:
        raise ValueError(f"Provedor desconhecido: {provider}")
    
    # Define o template do prompt com instruções rigorosas para não alucinar
    template = """
Você é um assistente útil e preciso. Responda à pergunta do usuário APENAS com base no contexto fornecido abaixo.
Se a informação necessária para responder não estiver presente no contexto, você deve responder EXATAMENTE: 
"Não encontrei essa informação nos materiais fornecidos."

Contexto:
{context}

Pergunta:
{question}

Resposta:
"""
    
    # Transforma o template em um formato que o LangChain entende
    PROMPT = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    # Cria e retorna a corrente (chain) de QA 
    # return_source_documents=True indica que queremos ver quais chunks basearam a resposta
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain
