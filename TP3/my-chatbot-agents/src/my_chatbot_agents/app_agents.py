"""
TP3 – Sistema de Agentes RAG por Persona (Streamlit + LangChain + Pinecone)

Características:
- Permite cargar múltiples CVs (PDF/DOCX), uno por persona
- Crea un agente por persona (enrutador) y responde según a quién se consulta
- Si la query no nombra a nadie, usa el agente del Alumno (persona por defecto)
- Si se consultan múltiples personas en la misma query, fusiona contextos y responde de forma comparativa

Ejecución:
  poetry run streamlit run CEIA-PNL2/nlp2-challenges/TP3/my-chatbot-agents/src/my_chatbot_agents/app_agents.py
"""

# ========================================
# IMPORTS
# ========================================

import hashlib
import os
import re
import unicodedata
from typing import Any, Dict, List, Tuple

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_pinecone import Pinecone as PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# ========================================
# CONFIG / ENV
# ========================================
load_dotenv()


# ========================================
# UTILIDADES (hash, embeddings, splitter, loaders, índice)
# ========================================
def compute_sha1(data: bytes) -> str:
    h = hashlib.sha1()
    h.update(data)
    return h.hexdigest()


def get_embeddings():
    from langchain_community.embeddings import FastEmbedEmbeddings

    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")


def get_splitter():
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except Exception:  # pragma: no cover
        from langchain.text_splitter import (
            RecursiveCharacterTextSplitter,  # type: ignore
        )
    return RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=100, add_start_index=True, separators=["\n\n", "\n", ". ", ".", " "]
    )


def ensure_index(index_name: str, dim: int) -> None:
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    names = {i.name for i in pc.list_indexes()}
    if index_name not in names:
        pc.create_index(index_name, dimension=dim, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))


def load_docs_from_upload(tmp_path: str) -> List[Any]:
    ext = os.path.splitext(tmp_path)[1].lower()
    if ext == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader

        docs = Docx2txtLoader(tmp_path).load()
        return docs
    from langchain_community.document_loaders import PyPDFLoader

    try:
        docs = PyPDFLoader(tmp_path).load()
    except Exception:
        docs = []
    if not any((d.page_content or "").strip() for d in docs):
        try:
            from langchain_community.document_loaders import PyMuPDFLoader

            docs = PyMuPDFLoader(tmp_path).load()
        except Exception:
            pass
    return docs


# ========================================
# INDEXACIÓN EN PINECONE (por persona)
# ========================================
def upsert_person_cv(
    person: str,
    file_bytes: bytes,
    filename: str,
    index_name: str,
    namespace: str,
) -> Tuple[PineconeVectorStore, str]:
    import tempfile

    # Elegir sufijo real segun nombre/extension
    suffix = os.path.splitext(filename)[1].lower()
    if suffix not in [".pdf", ".docx"]:
        # Detección simple por cabecera PDF
        suffix = ".pdf" if file_bytes[:4] == b"%PDF" else ".docx"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        docs = load_docs_from_upload(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    splitter = get_splitter()
    splits = splitter.split_documents(docs)
    for d in splits:
        d.metadata = d.metadata or {}
        d.metadata["person"] = person
        d.metadata["text"] = d.page_content

    emb = get_embeddings()
    ensure_index(index_name, len(emb.embed_query("dim")))

    from pinecone import Pinecone as PC

    pc = PC(api_key=os.environ["PINECONE_API_KEY"])
    idx = pc.Index(index_name)

    texts = [s.page_content for s in splits]
    vecs = emb.embed_documents(texts)
    vectors: List[Dict] = []
    def to_ascii_id(text: str) -> str:
        txt = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
        txt = re.sub(r"[^A-Za-z0-9_.-]+", "-", txt).strip("-").lower()
        return txt or "persona"
    person_key = to_ascii_id(person)
    for i, v in enumerate(vecs):
        vectors.append({
            "id": f"{person_key}-{i:06d}",
            "values": v,
            "metadata": {"person": person, "text": splits[i].page_content},
        })
    BATCH = 100
    for i in range(0, len(vectors), BATCH):
        idx.upsert(vectors=vectors[i : i + BATCH], namespace=namespace)

    return PineconeVectorStore(index_name=index_name, embedding=emb, namespace=namespace, text_key="text"), person


# ========================================
# ENRUTADO (LLM opcional) Y MATCHING DETERMINÍSTICO
# ========================================
def build_router_prompt() -> ChatPromptTemplate:
    template = (
        "Eres un enrutador. Dado el texto de una consulta y la lista de personas disponibles, "
        "devuelve únicamente los nombres de las personas relevantes separadas por coma. "
        "Si no se menciona persona explícita, devuelve solo 'Alumno'.\n\n"
        "Personas disponibles: {people}\n"
        "Consulta: {question}"
    )
    return ChatPromptTemplate.from_messages([
        SystemMessage(content="Devuelve solo los nombres, sin explicación."),
        ("human", template),
    ])


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", s.strip().lower())


def person_aliases(person: str, alumno_alias: str | None) -> List[str]:
    # Aliases básicos: nombre completo, tokens por camelCase y separadores
    aliases: List[str] = []
    raw = person.strip()
    aliases.append(raw)
    # camelCase → insertar espacio entre minúscula+Mayúscula
    camel_spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", raw)
    if camel_spaced != raw:
        aliases.append(camel_spaced)
    # split por separadores
    parts = re.split(r"[^A-Za-z]+", camel_spaced)
    if len(parts) >= 2:
        aliases.append(" ".join(parts))
    # extraer primera palabra (nombre)
    if parts and parts[0]:
        aliases.append(parts[0])
    # casos Alumno
    if raw.lower() == "alumno":
        aliases.extend(["alumno", "estudiante"])
        if alumno_alias:
            aliases.append(alumno_alias)
    # limpiar duplicados
    seen = set()
    out: List[str] = []
    for a in aliases:
        a_norm = _norm(a)
        if a_norm and a_norm not in seen:
            seen.add(a_norm)
            out.append(a)
    return out


def build_alias_map(people: List[str], alumno_alias: str | None) -> Dict[str, List[str]]:
    return {p: person_aliases(p, alumno_alias) for p in people}


def extract_people(question: str, people: List[str], alias_map: Dict[str, List[str]]) -> List[str]:
    q_norm = f" {_norm(question)} "
    found: List[str] = []
    for p in people:
        for alias in alias_map.get(p, [p]):
            a = _norm(alias)
            # Para alias multi-palabra, exigir que todas estén presentes
            tokens = [t for t in a.split(" ") if t]
            if tokens and all(re.search(rf"\b{re.escape(t)}\b", q_norm) for t in tokens):
                found.append(p)
                break
    return found or ["Alumno"]


# ========================================
# APLICACIÓN STREAMLIT (UI)
# ========================================
def main() -> None:
    st.set_page_config(page_title="TP3 – Agentes por Persona", page_icon="🧑‍🤝‍🧑", layout="wide")
    st.title("🧑‍🤝‍🧑 TP3: Agentes RAG por Persona")
    st.caption("Carga múltiples CVs (uno por persona). El sistema enruta la consulta al/los agente(s) correspondientes.")

    if not os.getenv("GROQ_API_KEY") or not os.getenv("PINECONE_API_KEY"):
        st.error("Configura GROQ_API_KEY y PINECONE_API_KEY (puedes copiar desde .env-local)")
        st.stop()

    # Sidebar
    st.sidebar.header("Configuración")
    index_name = st.sidebar.text_input("Índice Pinecone", value="cv-index-agentes")
    temperature = st.sidebar.slider("Temperatura", 0.0, 1.0, 0.0, 0.05)
    top_k = st.sidebar.slider("k (por persona)", 1, 6, 3)
    clear_ns = st.sidebar.checkbox("Limpiar namespace antes de reindexar", value=False)

    # Personas y cargas
    st.sidebar.subheader("Personas y CVs")
    default_people = ["Alumno", "Persona1", "Persona2"]
    people_str = st.sidebar.text_input("Lista de personas (coma)", value=",".join(default_people))
    people: List[str] = [p.strip() for p in people_str.split(",") if p.strip()]
    alumno_alias = st.sidebar.text_input("Nombre real del Alumno (alias)", value="", help="Ej.: Luciano")

    uploaded_files = st.sidebar.file_uploader("Sube CVs (PDF/DOCX)", type=["pdf", "docx"], accept_multiple_files=True)

    # Estado
    if "namespace" not in st.session_state:
        st.session_state.namespace = ""  # _default_
    if "people_to_vs" not in st.session_state:
        st.session_state.people_to_vs = {}

    # LLM
    llm = ChatGroq(groq_api_key=os.environ["GROQ_API_KEY"], model_name="llama3-8b-8192", temperature=temperature)

    # Indexación
    if uploaded_files and st.sidebar.button("(Re)indexar CVs"):
        st.info("Indexando CVs…")
        # Limpiar namespace global si se requiere
        if clear_ns:
            try:
                pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
                pc.Index(index_name).delete(delete_all=True, namespace=st.session_state.namespace)
            except Exception:
                pass
        st.session_state.people_to_vs = {}
        for f in uploaded_files:
            person_guess = os.path.splitext(os.path.basename(f.name))[0]
            # Si el nombre del archivo coincide con alguien de la lista, usarlo; si no, Alumno por defecto
            target_person = next((p for p in people if p.lower() in person_guess.lower()), "Alumno")
            vs, _ = upsert_person_cv(target_person, f.read(), f.name, index_name, st.session_state.namespace)
            st.session_state.people_to_vs[target_person] = vs
        st.success("CVs indexados ✅")

    # Enrutamiento y respuesta
    st.markdown("### 🔎 Pregunta")
    question = st.text_input("Escribe tu consulta", placeholder="¿Cuál fue su última experiencia laboral?")
    ask = st.button("Preguntar", type="primary")

    if ask and question.strip():
        # Enrutamiento determinístico con aliases (Alumno usa alias opcional)
        alias_map = build_alias_map(people, alumno_alias or None)
        names = extract_people(question, people, alias_map)

        # Recuperar contexto por persona
        per_person_context: Dict[str, List[Any]] = {}
        for name in names:
            vs = st.session_state.people_to_vs.get(name)
            if not vs:
                continue
            # Filtrar por metadata de persona para aislar el contexto correcto
            retr = vs.as_retriever(search_kwargs={
                "k": top_k,
                "filter": {"person": {"$eq": name}},
            })
            per_person_context[name] = retr.get_relevant_documents(question)

        # Construir respuesta
        if not any(per_person_context.values()):
            st.warning("No hay contexto recuperado. Asegúrate de haber indexado los CVs.")
            return

        def fmt(name: str, docs: List[Any]) -> str:
            chunks = []
            for i, d in enumerate(docs, 1):
                chunks.append(f"[{name} - frag.{i}]\n{d.page_content}")
            return "\n\n".join(chunks)

        context_blocks = []
        for name, docs in per_person_context.items():
            if docs:
                context_blocks.append(fmt(name, docs))
        full_context = "\n\n".join(context_blocks)

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=(
                "Responde exclusivamente con la información del contexto. "
                "Si no hay datos, indica que no se encuentra en los CVs. Cuando hay varias personas, compara claramente."
            )),
            ("human", "Contexto:\n{ctx}\n\nPregunta:\n{q}"),
        ])
        chain = prompt | llm
        out = chain.invoke({"ctx": full_context, "q": question})

        st.markdown("### 🧠 Respuesta")
        st.write(out.content if hasattr(out, "content") else str(out))

        with st.expander("📚 Contexto por persona"):
            for name, docs in per_person_context.items():
                st.markdown(f"**{name}** — {len(docs)} fragmentos")
                for i, d in enumerate(docs, 1):
                    st.write(d.page_content)
                    st.caption(str(d.metadata))
                    st.markdown("---")


# ========================================
# EJECUCIÓN
# ========================================
if __name__ == "__main__":
    main()
