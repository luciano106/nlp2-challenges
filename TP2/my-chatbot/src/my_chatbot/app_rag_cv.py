"""
TP1 - RAG sobre CV en PDF con Streamlit, LangChain y Groq
========================================================

Esta aplicación implementa un flujo de Retrieval-Augmented Generation (RAG)
que permite:
 - Cargar un CV en PDF o DOCX
 - Indexarlo en Pinecone usando embeddings de HuggingFace o FastEmbed
 - Consultar el contenido del CV mediante un LLM de Groq con contexto recuperado

Requisitos (instalar):
   - pip install streamlit langchain langchain-groq langchain-community langchain-text-splitters pinecone langchain-pinecone sentence-transformers pypdf docx2txt
   - Verificar README.md para más detalles (el proyecto utiliza poetry)

Ejecución:
    streamlit run app_rag_cv.py

Variables de entorno necesarias:
    GROQ_API_KEY       # obtener en https://console.groq.com
    PINECONE_API_KEY   # obtener en https://app.pinecone.io
"""

import hashlib
import os
from typing import Any, List

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_pinecone import Pinecone as PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()


def _get_text_splitter_class():
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter as _R
    except Exception:  # pragma: no cover
        from langchain.text_splitter import (
            RecursiveCharacterTextSplitter as _R,  # type: ignore
        )
    return _R


# ------------------------------------------------------------
# Utilidades
# ------------------------------------------------------------

def compute_file_sha1(file_bytes: bytes) -> str:
    sha1 = hashlib.sha1()
    sha1.update(file_bytes)
    return sha1.hexdigest()


def ensure_pinecone_index(index_name: str, dimension: int, metric: str = "cosine", recreate_if_mismatch: bool = False) -> None:
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    existing = {idx.name for idx in pc.list_indexes()}
    if index_name in existing:
        try:
            current_stats = pc.Index(index_name).describe_index_stats()
            current_dim = current_stats.get("dimension")
            if current_dim is not None and current_dim != dimension:
                if recreate_if_mismatch:
                    pc.delete_index(index_name)
                    pc.create_index(
                        name=index_name,
                        dimension=dimension,
                        metric=metric,
                        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                    )
                else:
                    raise RuntimeError(
                        f"Dimensión del índice '{index_name}' ({current_dim}) != embeddings ({dimension}). Cambia 'Nombre del índice Pinecone' o activa 'Recrear índice si difiere'."
                    )
        except Exception:
            # Si falla obtener stats, intentamos usar el índice tal cual
            pass
    else:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )


def get_embeddings_dimension(embeddings: Any) -> int:
    sample_vec = embeddings.embed_query("dimension-check")
    return len(sample_vec)


def index_pdf_into_pinecone(
    pdf_bytes: bytes,
    chunk_size: int,
    chunk_overlap: int,
    index_name: str,
    namespace: str,
    embeddings: Any,
    recreate_if_mismatch: bool,
    metadata_hash: str,
) -> PineconeVectorStore:
    import tempfile

    # Detectar extensión/loader según el archivo subido
    file_suffix = ".pdf"
    if st.session_state.get("uploaded_filename", "").lower().endswith(".docx"):
        file_suffix = ".docx"

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        if file_suffix == ".docx":
            loader = Docx2txtLoader(tmp_path)
            documents = loader.load()
            if not any((d.page_content or "").strip() for d in documents):
                raise RuntimeError("El DOCX no contiene texto extraíble.")
        else:
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            # Fallback a PyMuPDF si no se extrajo texto (PDF escaneado/imagen)
            if not any((d.page_content or "").strip() for d in documents):
                try:
                    from langchain_community.document_loaders import (
                        PyMuPDFLoader,  # requires pymupdf
                    )
                    loader2 = PyMuPDFLoader(tmp_path)
                    documents = loader2.load()
                except Exception:
                    pass
            if not any((d.page_content or "").strip() for d in documents):
                raise RuntimeError(
                    "El PDF no contiene texto extraíble. Si es escaneado, instala 'pymupdf' y reintenta, o sube un PDF con texto."
                )
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    TextSplitter = _get_text_splitter_class()
    text_splitter = TextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", ".", " "]
    )
    splits = text_splitter.split_documents(documents)
    # Debug: mostrar cantidad de fragmentos generados
    try:
        st.sidebar.write({
            "splits_generados": len(splits),
            "chars_total": sum(len((s.page_content or "")) for s in splits),
        })
    except Exception:
        pass

    # Asegurar metadatos para filtrar por namespace/hash
    for d in splits:
        d.metadata = d.metadata or {}
        d.metadata["file_hash"] = namespace
        d.metadata["index_name"] = index_name

    # Crear/validar índice y hacer upsert directo (como en el script de debug)
    ensure_pinecone_index(
        index_name,
        get_embeddings_dimension(embeddings),
        recreate_if_mismatch=recreate_if_mismatch,
    )

    from pinecone import Pinecone

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    idx = pc.Index(index_name)

    # Embeddings por fragmento
    texts = [s.page_content for s in splits]
    vectors = []
    embedded = embeddings.embed_documents(texts)
    id_prefix = f"cv-{namespace[:8]}-"
    for i, vec in enumerate(embedded):
        meta = splits[i].metadata or {}
        # Reducir metadata a tipos serializables simples
        safe_meta = {k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))}
        # Guardar SIEMPRE el hash real del archivo, aunque el namespace sea vacío
        safe_meta.setdefault("file_hash", metadata_hash)
        # Guardar el texto del fragmento bajo la clave 'text' para que el vectorstore pueda reconstruir el Document
        safe_meta["text"] = splits[i].page_content
        vectors.append({
            "id": f"{id_prefix}{i:06d}",
            "values": vec,
            "metadata": safe_meta,
        })

    BATCH = 100
    for i in range(0, len(vectors), BATCH):
        idx.upsert(vectors=vectors[i:i + BATCH], namespace=namespace)

    # Polling de conteo para reflejar en UI
    try:
        vc = 0
        for _ in range(20):
            if namespace:
                stats_ns = idx.describe_index_stats(namespace=namespace)
                vc = stats_ns.get("namespaces", {}).get(namespace, {}).get("vector_count", 0)
            else:
                stats = idx.describe_index_stats()
                vc = stats.get("total_vector_count", 0)
            try:
                st.sidebar.write({"vector_count_ns": vc})
            except Exception:
                pass
            if vc > 0:
                break
            import time
            time.sleep(1)
    except Exception:
        pass

    # Devolver vectorstore apuntando al índice/namespace (si namespace vacío, sin parámetro)
    if namespace:
        return PineconeVectorStore(index_name=index_name, embedding=embeddings, namespace=namespace, text_key="text")
    return PineconeVectorStore(index_name=index_name, embedding=embeddings, text_key="text")


def load_vectorstore_pinecone(
    index_name: str,
    namespace: str,
    embeddings: Any,
    recreate_if_mismatch: bool,
) -> PineconeVectorStore:
    ensure_pinecone_index(index_name, get_embeddings_dimension(embeddings), recreate_if_mismatch=recreate_if_mismatch)
    return PineconeVectorStore(index_name=index_name, embedding=embeddings, namespace=namespace)


def format_context(docs: List[Any]) -> str:
    formatted_chunks: List[str] = []
    for i, d in enumerate(docs, start=1):
        metadata = d.metadata or {}
        page = metadata.get("page", "?")
        source = metadata.get("source", "PDF")
        formatted_chunks.append(f"[Fragmento {i} | {source} p.{page}]\n{d.page_content}")
    return "\n\n".join(formatted_chunks)


def build_prompt(system_instructions: str) -> ChatPromptTemplate:
    template = (
        "Responde exlusivamente con información presente en el contexto.\n"
        "Si la información solicitada no aparece en el contexto, responde: 'No se encuentra esa información en el CV'.\n"
        "Responde en español, de forma concisa y profesional e incluye una cita breve con la página (p.X).\n\n"
        "Contexto:\n{context}\n\n"
        "Pregunta: {question}"
    )
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_instructions),
        ("human", template),
    ])
    return prompt


def get_default_embeddings():
    """Devuelve FastEmbedEmbeddings forzando backend sin torch.

    Requiere fastembed y onnxruntime (en Mac M‑series: onnxruntime-silicon).
    """
    from langchain_community.embeddings import FastEmbedEmbeddings

    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")


# ------------------------------------------------------------
# Aplicación Streamlit
# ------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="TP1 - RAG sobre CV (PDF)", page_icon="📄", layout="wide")
    st.title("📄 TP1: RAG sobre CV con Groq + LangChain + Pinecone")
    st.caption("Carga un CV en PDF, indexa en Pinecone y consulta con evidencia del documento.")

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ_API_KEY no está configurada. Exporta la variable de entorno antes de continuar.")
        st.code("export GROQ_API_KEY='tu-clave'", language="bash")
        st.stop()

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        st.error("PINECONE_API_KEY no está configurada. Exporta la variable de entorno antes de continuar.")
        st.code("export PINECONE_API_KEY='tu-clave'", language="bash")
        st.stop()

    # Sidebar: configuración
    st.sidebar.title("⚙️ Configuración")
    model_name = st.sidebar.selectbox(
        "Modelo Groq",
        options=[
            "llama3-8b-8192",
            "mixtral-8x7b-32768",
            "gemma-7b-it",
        ],
        index=0,
        help="Selecciona el LLM a utilizar para generar respuestas."
    )
    temperature = st.sidebar.slider("Temperatura", 0.0, 1.0, 0.0, 0.05)
    top_k = st.sidebar.slider("Documentos a recuperar (k)", 1, 8, 4)
    chunk_size = st.sidebar.slider("Tamaño de chunk", 300, 2000, 800, 50)
    chunk_overlap = st.sidebar.slider("Solapamiento", 0, 400, 150, 10)

    index_name = st.sidebar.text_input("Nombre del índice Pinecone", value="cv-index")
    recreate_index = st.sidebar.checkbox("Recrear índice si difiere la dimensión", value=True,
                                         help="Si cambiaste de modelo de embeddings, recrea el índice automáticamente.")

    system_instructions = st.sidebar.text_area(
        "Instrucciones del sistema",
        value=(
            "Eres un asistente que responde preguntas sobre el contenido de un CV. "
            "No inventes información y cita fragmentos del documento.")
    )

    st.sidebar.markdown("---")
    uploaded_pdf = st.sidebar.file_uploader("Sube el CV (PDF o DOCX)", type=["pdf", "docx"], accept_multiple_files=False)
    index_status_placeholder = st.sidebar.empty()
    clear_ns = st.sidebar.checkbox("Limpiar namespace antes de reindexar", value=False, help="Elimina vectores del CV actual (hash) antes de (re)indexar")
    debug_retrieval = st.sidebar.checkbox("Mostrar debug de recuperación", value=True)

    if "namespace" not in st.session_state:
        st.session_state.namespace = None

    # Construcción del LLM
    try:
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=800,
        )
        st.sidebar.success("Modelo Groq listo ✅")
    except Exception as e:
        st.sidebar.error(f"Error al inicializar el modelo Groq: {e}")
        st.stop()

    # Embeddings (una sola vez)
    try:
        embeddings = get_default_embeddings()
        st.sidebar.caption("Embeddings cargados correctamente")
    except Exception as e:
        st.sidebar.error(str(e))
        st.stop()

    vectorstore = None

    if uploaded_pdf is not None:
        file_bytes = uploaded_pdf.read()
        st.session_state.uploaded_filename = uploaded_pdf.name
        file_hash = compute_file_sha1(file_bytes)
        namespace = ""  # Siempre usar namespace por defecto en Pinecone
        st.session_state.namespace = namespace
        if debug_retrieval:
            st.sidebar.caption(f"Namespace (hash): {namespace}")

        if st.sidebar.button("(Re)indexar PDF"):
            index_status_placeholder.info("Indexando PDF en Pinecone… esto puede tardar unos segundos")
            # Reintentos silenciosos para errores transitorios
            import time as _t
            last_error = None
            for attempt in range(3):
                try:
                    if clear_ns:
                        try:
                            pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
                            pc.Index(index_name).delete(delete_all=True, namespace=namespace)
                            index_status_placeholder.caption("Namespace limpiado ✅")
                        except Exception:
                            # Silenciar errores transitorios de limpieza
                            pass
                    vectorstore = index_pdf_into_pinecone(
                        pdf_bytes=file_bytes,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        index_name=index_name,
                        namespace=namespace,
                        embeddings=embeddings,
                        recreate_if_mismatch=recreate_index,
                        metadata_hash=file_hash,
                    )
                    index_status_placeholder.success("Índice actualizado en Pinecone ✅")
                    # Intento de leer stats (silencioso si falla)
                    try:
                        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
                        _ = pc.Index(index_name).describe_index_stats()
                    except Exception:
                        pass
                    last_error = None
                    break
                except Exception as e:
                    last_error = e
                    index_status_placeholder.info("Reintentando indexación…")
                    _t.sleep(2)
            if last_error is not None:
                index_status_placeholder.error("No se pudo completar la indexación tras varios intentos.")
        else:
            try:
                vectorstore = load_vectorstore_pinecone(
                    index_name=index_name,
                    namespace=namespace,
                    embeddings=embeddings,
                    recreate_if_mismatch=recreate_index,
                )
                index_status_placeholder.success("Índice existente en Pinecone listo ✅")
            except Exception as e:
                index_status_placeholder.error(f"No se pudo acceder al índice: {e}")

    # Interfaz de consulta
    st.markdown("### 🔎 Consulta el CV")
    question = st.text_input(
        "Escribe tu pregunta",
        placeholder="Ej.: ¿Cuáles son las principales habilidades técnicas?",
    )

    col1, col2 = st.columns([3, 2])
    with col1:
        ask_clicked = st.button("Preguntar", type="primary")
    with col2:
        clear_clicked = st.button("Limpiar")

    if clear_clicked:
        st.session_state.pop("last_answer", None)
        st.session_state.pop("last_docs", None)

    can_search = vectorstore is not None and bool(question.strip())

    if ask_clicked and not vectorstore:
        st.warning("Sube y (re)indexa un PDF antes de preguntar.")

    if ask_clicked and can_search:
        with st.spinner("Buscando en el CV y generando respuesta…"):
            try:
                # Recuperación sin filtros (namespace por defecto)
                retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
                docs = retriever.get_relevant_documents(question)

                context = format_context(docs)

                st.session_state.last_docs = docs
                if debug_retrieval:
                    st.sidebar.write({
                        "docs_recuperados": len(docs),
                        "index_name": index_name,
                        "namespace": st.session_state.namespace,
                        "k": top_k,
                    })

                if not docs:
                    st.session_state.last_answer = "No se encuentra esa información en el CV."
                else:
                    prompt = build_prompt(system_instructions)
                    chain = prompt | llm
                    response = chain.invoke({"context": context, "question": question})
                    st.session_state.last_answer = response.content if hasattr(response, "content") else str(response)
            except Exception as e:
                st.error(f"Error durante la consulta: {e}")

    # Mostrar resultados
    if st.session_state.get("last_answer"):
        st.markdown("### 🧠 Respuesta")
        st.write(st.session_state["last_answer"])

    with st.expander("📚 Fragmentos recuperados (contexto)"):
        docs_view = st.session_state.get("last_docs") or []
        st.caption(f"Total fragmentos: {len(docs_view)}")
        for i, d in enumerate(docs_view, start=1):
            meta = d.metadata or {}
            page = meta.get("page", "?")
            st.markdown(f"**Fragmento {i} — página {page}**")
            st.write(d.page_content)
            st.caption(str(meta))
            st.markdown("---")

    st.markdown("---")
    with st.expander("ℹ️ Notas técnicas"):
        st.markdown(
            "- Vectorstore: Pinecone (serverless) con namespaces por hash de archivo.\n"
            "- Embeddings: fallback a FastEmbed `BAAI/bge-small-en-v1.5`.\n"
            "- Splitter: RecursiveCharacterTextSplitter.\n"
            "- LLM: Groq (configurable)."
        )


if __name__ == "__main__":
    main()


