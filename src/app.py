import streamlit as st
import os
import re
import time
import json
from dotenv import load_dotenv

# Configura a página principal do Streamlit antes de outras importações
st.set_page_config(page_title="Assistente RAG Inteligente", page_icon="🤖", layout="wide")

from chain import get_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI

# Carrega as variáveis de ambiente (especialmente GOOGLE_API_KEY)
load_dotenv()

# Caminho absoluto para o arquivo de perguntas de teste
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
perguntas_path = os.path.join(project_root, "eval", "perguntas_teste.json")

# Função auxiliar para retentativa exponencial em caso de erro 429 (Resource Exhausted)
def invoke_with_retry(func, *args, max_retries=4, initial_delay=3, **kwargs):
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                # Aguarda o reset da quota de requisições por minuto (RPM)
                time.sleep(delay)
                delay += 3  # Incremento linear simples para backoff
            else:
                raise e
    # Última tentativa direta caso esgote as retentativas no loop
    return func(*args, **kwargs)

# Inicializa o pipeline de RAG em cache
@st.cache_resource(show_spinner=False)
def load_chain(provider):
    if provider == "gemini":
        if not os.environ.get("GOOGLE_API_KEY"):
            return None
    elif provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            return None
    return get_qa_chain(provider=provider)

# Inicializa o LLM avaliador em cache
@st.cache_resource(show_spinner=False)
def load_evaluator_llm(provider):
    if provider == "gemini":
        if not os.environ.get("GOOGLE_API_KEY"):
            return None
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0
        )
    elif provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            return None
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4o-mini-2024-07-18",
            temperature=0
        )
    return None

# Carrega a base de dados de testes
def carregar_perguntas_teste():
    try:
        with open(perguntas_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Erro ao carregar perguntas de teste em '{perguntas_path}': {e}")
        return []

perguntas_teste = carregar_perguntas_teste()

# --- Funções de Avaliação ---

def obter_pergunta_teste_correspondente(prompt):
    """
    Verifica se o prompt do usuário corresponde a alguma pergunta do benchmark
    (ignora maiúsculas/minúsculas e pontuação).
    """
    def limpar(s):
        return re.sub(r'[^\w\s]', '', s).strip().lower()
    
    prompt_limpo = limpar(prompt)
    for item in perguntas_teste:
        if limpar(item["pergunta"]) == prompt_limpo:
            return item
    return None

def avaliar_precisao_llm(pergunta, resposta_gerada, resposta_esperada):
    """
    Compara a resposta gerada com a esperada (gabarito) usando o Gemini e retorna a precisão (0-100%).
    """
    frase_recusa = "Não encontrei essa informação nos materiais fornecidos."
    if frase_recusa in resposta_gerada and frase_recusa in resposta_esperada:
        return 100
        
    prompt_eval = f"""
Você é um avaliador especialista em sistemas RAG.
Compare a Resposta Gerada pelo assistente com a Resposta Esperada (Gabarito) para a pergunta fornecida.

Pergunta: {pergunta}
Resposta Esperada (Gabarito): {resposta_esperada}
Resposta Gerada: {resposta_gerada}

Dê uma nota de 0 a 100 representando a Precisão (fidelidade de conteúdo e correção de fatos) da Resposta Gerada em relação à Resposta Esperada.
Instruções:
- Se a resposta gerada estiver correta e contiver a essência da resposta esperada, a nota deve ser alta (90-100).
- Se estiver parcialmente correta, atribua uma nota intermediária (50-80).
- Se estiver incorreta ou contiver informações não suportadas ou incorretas, a nota deve ser baixa (0-40).
- Responda APENAS com um número inteiro entre 0 e 100. Não escreva mais nada além do número.

Nota (0-100):"""
    
    try:
        resultado = invoke_with_retry(evaluator_llm.invoke, prompt_eval).content.strip()
        nota = int(re.search(r'\d+', resultado).group())
        return min(max(nota, 0), 100)
    except Exception:
        return 50

def avaliar_fidelidade_ao_contexto(contexto, resposta):
    """
    Avalia o quanto a resposta é fiel ao contexto recuperado (Faithfulness / Groundedness).
    """
    frase_recusa = "Não encontrei essa informação nos materiais fornecidos."
    if frase_recusa in resposta:
        return 100

    prompt_fidelidade = f"""
Analise a resposta fornecida em relação ao contexto de suporte e avalie se a resposta é 100% suportada e fiel ao contexto.
Se a resposta contiver fatos que não estão no contexto de suporte, diminua a nota de fidelidade.

Contexto:
{contexto}

Resposta:
{resposta}

Dê uma nota de fidelidade/groundedness de 0 a 100.
Responda APENAS com um número inteiro entre 0 e 100. Não escreva mais nada além do número.

Nota (0-100):"""
    try:
        resultado = invoke_with_retry(evaluator_llm.invoke, prompt_fidelidade).content.strip()
        nota = int(re.search(r'\d+', resultado).group())
        return min(max(nota, 0), 100)
    except Exception:
        return 100

# --- Interface Gráfica Streamlit ---

# Configuração da Barra Lateral
with st.sidebar:
    st.markdown("### 🤖 Provedor de LLM")
    provedor_opcao = st.selectbox(
        "Escolha o Modelo de Linguagem:",
        options=["Google Gemini (gemini-2.5-flash)", "OpenAI (gpt-4o-mini-2024-07-18)"],
        index=0
    )
    provider = "gemini" if "Google Gemini" in provedor_opcao else "openai"
    
    # Verifica se a respectiva chave de API está configurada
    if provider == "gemini" and not os.environ.get("GOOGLE_API_KEY"):
        st.error("⚠️ Chave `GOOGLE_API_KEY` não encontrada no arquivo `.env`.")
        st.stop()
    elif provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        st.error("⚠️ Chave `OPENAI_API_KEY` não encontrada no arquivo `.env`.")
        st.stop()
        
    st.markdown("---")
    st.markdown("### ⚙️ Configurações RAG")
    habilitar_auditoria = st.checkbox(
        "Habilitar Auditoria em Tempo Real",
        value=False,
        help="Analisa cada resposta com o modelo em tempo real para calcular a precisão/fidelidade. Isso consome 1 chamada de API adicional por mensagem."
    )
    st.caption("ℹ️ Desative esta opção para economizar sua cota de requisições por minuto (RPM) da API gratuita.")

    st.markdown("---")
    st.markdown("### 📄 Upload de Documentos")

    docs_dir = os.path.join(project_root, "docs")
    os.makedirs(docs_dir, exist_ok=True)

    existing_pdfs = sorted([f for f in os.listdir(docs_dir) if f.lower().endswith(".pdf")])
    if existing_pdfs:
        with st.expander(f"📁 {len(existing_pdfs)} PDF(s) na base"):
            for pdf_name in existing_pdfs:
                st.caption(f"• {pdf_name}")

    if "upload_key" not in st.session_state:
        st.session_state.upload_key = 0

    uploaded_files = st.file_uploader(
        "Envie arquivos PDF",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"pdf_uploader_{st.session_state.upload_key}",
    )

    if uploaded_files:
        nomes = []
        for arquivo in uploaded_files:
            caminho = os.path.join(docs_dir, arquivo.name)
            with open(caminho, "wb") as out:
                out.write(arquivo.getbuffer())
            nomes.append(arquivo.name)

        with st.spinner("Processando documentos e atualizando base de conhecimento..."):
            from langchain_community.document_loaders import PyPDFLoader
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from langchain_chroma import Chroma
            from langchain_huggingface import HuggingFaceEmbeddings

            all_chunks = []
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
            for nome in nomes:
                loader = PyPDFLoader(os.path.join(docs_dir, nome))
                docs = loader.load()
                all_chunks.extend(text_splitter.split_documents(docs))

            if all_chunks:
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                )
                vectorstore = Chroma(
                    persist_directory="chroma_db",
                    embedding_function=embeddings
                )
                vectorstore.add_documents(all_chunks)
                load_chain.clear()

        st.success(f"✅ {len(nomes)} PDF(s) processado(s): {', '.join(nomes)}")
        st.session_state.upload_key += 1
        st.rerun()

# Carrega a pipeline RAG e o avaliador
try:
    qa_chain = load_chain(provider)
    evaluator_llm = load_evaluator_llm(provider)
except Exception as e:
    st.error(f"Erro ao carregar o modelo ou a base de dados: {e}. Certifique-se de ter PDFs em `docs/` e de ter rodado o script `embedder.py` primeiro.")
    st.stop()

# Abas principais
tab1, tab2 = st.tabs(["💬 Chat Assistente", "📊 Painel de Avaliação"])

# --- ABA 1: CHAT ASSISTENTE ---
with tab1:
    st.write("Faça perguntas com base nos documentos PDF inseridos na pasta `docs/`.")
    
    # Inicializa o histórico de chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Exibe histórico
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "metrics" in msg:
                st.markdown(msg["metrics"], unsafe_allow_html=True)

    # Input do usuário
    if prompt := st.chat_input("Pergunte algo sobre os PDFs..."):
        # Adiciona e exibe pergunta do usuário
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Resposta da IA
        with st.chat_message("assistant"):
            with st.spinner("Buscando informações e gerando resposta..."):
                try:
                    # Mede latência
                    start_time = time.time()
                    result = invoke_with_retry(qa_chain.invoke, {"query": prompt})
                    latency = time.time() - start_time
                    
                    resposta = result["result"]
                    fontes = result.get("source_documents", [])
                    
                    st.markdown(resposta)
                    
                    # Painel de fontes consultadas formatado de forma premium
                    frase_recusa = "Não encontrei essa informação nos materiais fornecidos."
                    if fontes and resposta.strip() != frase_recusa:
                        with st.expander("🔎 Ver Fontes Consultadas"):
                            st.write("")
                            for i, doc in enumerate(fontes):
                                origem = doc.metadata.get('source', 'Desconhecida')
                                nome_arquivo = os.path.basename(origem.replace('\\', '/'))
                                pagina = doc.metadata.get('page', 'Desconhecida')
                                num_pagina = doc.metadata.get('page_label', str(pagina + 1) if isinstance(pagina, int) else pagina)
                                
                                conteudo_limpo = re.sub(r'\s+', ' ', doc.page_content.strip())
                                
                                with st.container(border=True):
                                    col_info, col_texto = st.columns([1, 3])
                                    with col_info:
                                        st.markdown(f"##### 📄 Fonte {i+1}")
                                        st.caption(f"**Arquivo:**\n`{nome_arquivo}`")
                                        st.caption(f"**Página:** `{num_pagina}`")
                                    with col_texto:
                                        st.markdown("**Trecho extraído do PDF:**")
                                        st.markdown(f"> *\"{conteudo_limpo}\"*")
                    
                    # Avaliação das métricas da resposta em tempo real
                    contexto_completo = "\n---\n".join([doc.page_content for doc in fontes])
                    pergunta_teste = obter_pergunta_teste_correspondente(prompt)
                    
                    if pergunta_teste:
                        e_recusa = (pergunta_teste["dificuldade"] == "fora do escopo")
                        if e_recusa:
                            recusa_correta = (frase_recusa in resposta)
                            precisao_texto = "100%" if recusa_correta else "0%"
                            status_recusa = "✅ Recusa Correta" if recusa_correta else "❌ Falha na Recusa"
                        else:
                            if habilitar_auditoria:
                                precisao = avaliar_precisao_llm(prompt, resposta, pergunta_teste["resposta_esperada"])
                                precisao_texto = f"{precisao}%"
                            else:
                                precisao_texto = "<i>Ative a Auditoria na barra lateral para calcular</i>"
                            status_recusa = "N/A"
                        
                        metrics_html = f"""
                        <div style="background-color: rgba(28, 107, 240, 0.1); border-left: 5px solid #1c6bf0; padding: 10px; border-radius: 4px; margin-top: 15px;">
                            <span style="font-weight: bold; color: #1c6bf0;">📊 Métricas de Avaliação (Benchmark):</span><br>
                            ⚡ <b>Latência:</b> <code>{latency:.2f}s</code> | 
                            🎯 <b>Precisão (vs Gabarito):</b> <code>{precisao_texto}</code> | 
                            🛡️ <b>Recusa:</b> <code>{status_recusa}</code>
                        </div>
                        """
                    else:
                        if habilitar_auditoria:
                            fidelidade = avaliar_fidelidade_ao_contexto(contexto_completo, resposta)
                            fidelidade_texto = f"{fidelidade}%"
                        else:
                            fidelidade_texto = "<i>Ative a Auditoria na barra lateral para calcular</i>"
                            
                        status_recusa = "✅ Recusa Ativada" if frase_recusa in resposta else "N/A"
                        
                        metrics_html = f"""
                        <div style="background-color: rgba(28, 200, 138, 0.1); border-left: 5px solid #1cc88a; padding: 10px; border-radius: 4px; margin-top: 15px;">
                            <span style="font-weight: bold; color: #1cc88a;">📊 Métricas da Resposta:</span><br>
                            ⚡ <b>Latência:</b> <code>{latency:.2f}s</code> | 
                            🎯 <b>Precisão (Fidelidade ao Contexto):</b> <code>{fidelidade_texto}</code> | 
                            🛡️ <b>Recusa:</b> <code>{status_recusa}</code>
                        </div>
                        """
                        
                    st.markdown(metrics_html, unsafe_allow_html=True)
                    
                    # Salva no histórico com a métrica formatada
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": resposta,
                        "metrics": metrics_html
                    })
                    
                except Exception as e:
                    st.error(f"Ocorreu um erro durante a geração da resposta: {e}")

# --- ABA 2: PAINEL DE AVALIAÇÃO (BENCHMARK) ---
with tab2:
    st.header("📊 Avaliação e Diagnóstico de Performance (RAG)")
    st.write(
        "Este painel executa automaticamente o conjunto de testes localizado em `eval/perguntas_teste.json` e "
        "compara os retornos do assistente com o gabarito. Ele calcula o tempo de resposta, precisão de conteúdo e o índice de recusas corretas."
    )
    
    if len(perguntas_teste) == 0:
        st.warning("⚠️ Nenhuma pergunta de teste carregada. Verifique seu arquivo `eval/perguntas_teste.json`.")
    else:
        st.info(f"O arquivo contém **{len(perguntas_teste)} perguntas** prontas para teste.")
        
        # Botão para disparar a bateria de testes
        if st.button("🚀 Rodar Bateria de Testes (Benchmark)", key="run_benchmark"):
            resultados = []
            tempo_total_latencia = 0.0
            soma_precisao = 0
            recusas_esperadas = 0
            recusas_corretas = 0
            in_scope_count = 0
            
            # Container de progresso
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Loop das perguntas
            for i, item in enumerate(perguntas_teste):
                # Aguarda 4 segundos antes de cada chamada (exceto a primeira) para respeitar o limite de cota gratuita (15 RPM)
                if i > 0:
                    time.sleep(4)
                
                pergunta = item["pergunta"]
                status_text.text(f"Processando pergunta {i+1} de {len(perguntas_teste)}: '{pergunta[:50]}...'")
                
                # Executa o pipeline RAG medindo latência
                start_time = time.time()
                try:
                    result = invoke_with_retry(qa_chain.invoke, {"query": pergunta})
                except Exception as e:
                    result = {"result": f"Erro na execução: {e}", "source_documents": []}
                latency = time.time() - start_time
                
                tempo_total_latencia += latency
                resposta_gerada = result["result"]
                resposta_esperada = item["resposta_esperada"]
                dificuldade = item["dificuldade"]
                
                # Regras de avaliação para precisão e recusa
                e_recusa = (dificuldade == "fora do escopo")
                frase_recusa = "Não encontrei essa informação nos materiais fornecidos."
                
                if e_recusa:
                    recusas_esperadas += 1
                    recusa_correta = (frase_recusa in resposta_gerada)
                    if recusa_correta:
                        recusas_corretas += 1
                        precisao = 100
                    else:
                        precisao = 0
                else:
                    recusa_correta = None
                    in_scope_count += 1
                    precisao = avaliar_precisao_llm(pergunta, resposta_gerada, resposta_esperada)
                    soma_precisao += precisao
                
                resultados.append({
                    "ID": i + 1,
                    "Dificuldade": dificuldade.capitalize(),
                    "Pergunta": pergunta,
                    "Resposta Gerada": resposta_gerada,
                    "Resposta Esperada": resposta_esperada,
                    "Latência": f"{latency:.2f}s",
                    "Precisão": f"{precisao}%",
                    "Recusa Correta": "Sim" if recusa_correta is True else ("Não" if recusa_correta is False else "-")
                })
                
                # Atualiza a barra de progresso
                progress_bar.progress((i + 1) / len(perguntas_teste))
            
            status_text.success("🎉 Bateria de testes concluída!")
            
            # Cálculos finais
            avg_latency = tempo_total_latencia / len(perguntas_teste)
            avg_precision = (soma_precisao / in_scope_count) if in_scope_count > 0 else 0
            recusa_index = (recusas_corretas / recusas_esperadas * 100) if recusas_esperadas > 0 else 0
            
            # Renderiza as métricas resumidas de forma premium
            st.write("")
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric(label="⚡ Latência Média", value=f"{avg_latency:.2f} s", delta=None)
            with col_m2:
                # Compara com a meta de > 90%
                delta_precisao = f"{avg_precision - 90:.1f}% vs Meta (90%)" if avg_precision >= 90 else f"{avg_precision - 90:.1f}% vs Meta (90%)"
                st.metric(label="🎯 Precisão Média (Dentro de Escopo)", value=f"{avg_precision:.1f}%", delta=delta_precisao, delta_color="normal")
            with col_m3:
                # Compara com a meta de 100% de recusa
                delta_recusa = "Meta atingida! (100%)" if recusa_index == 100 else f"{recusa_index - 100:.1f}% vs Meta"
                st.metric(label="🛡️ Índice de Recusa Correta", value=f"{recusa_index:.1f}%", delta=delta_recusa)
                
            # Exibe a tabela detalhada de resultados
            st.write("")
            st.markdown("### 📋 Tabela Detalhada de Resultados")
            st.dataframe(resultados, use_container_width=True, hide_index=True)
