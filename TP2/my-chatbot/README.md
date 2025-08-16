# TP2 – Chatbot RAG sobre CV (Streamlit + LangChain + Pinecone)

## Descripción
Aplicación Streamlit que implementa Retrieval-Augmented Generation (RAG) para consultar un CV (PDF o DOCX). Indexa fragmentos en Pinecone y genera respuestas con Groq, citando el contexto utilizado.

## Requisitos
- Python 3.10–3.11 (gestionado por Poetry en este proyecto)
- Cuenta y API Key de Pinecone
- API Key de Groq

## Configuración con Poetry
1) Instalar dependencias del proyecto:
```bash
poetry install
```
2) Exportar variables de entorno (en la misma terminal/shell donde uses Poetry):
```bash
export GROQ_API_KEY="tu-clave-groq"
export PINECONE_API_KEY="tu-clave-pinecone"
```

## Ejecutar la app
```bash
poetry run streamlit run CEIA-PNL2/nlp2-challenges/TP2/my-chatbot/src/my_chatbot/app_rag_cv.py
```

## Uso en la UI
1) En la barra lateral:
   - Nombre del índice Pinecone: por defecto `cv-index` (puedes cambiarlo si lo deseas).
   - Temperatura: 0 (recomendado para factualidad).
   - Chunking: `Tamaño de chunk` y `Solapamiento` (valores sugeridos: 400 / 100).
   - Limpiar namespace antes de reindexar: úsalo solo cuando vayas a reindexar el mismo CV o cambies el chunking.
   - Mostrar debug de recuperación: está habilitado por defecto (puedes desactivarlo).

2) Carga del archivo:
   - Sube un CV en PDF o DOCX.
   - Pulsa “(Re)indexar PDF”. Verás un mensaje “Indexando…” y luego “Índice actualizado…”.

3) Consulta:
   - Escribe tu pregunta y pulsa “Preguntar”.
   - Bajo la respuesta, abre “📚 Fragmentos recuperados (contexto)” para ver los trozos usados (con metadatos de página si el loader los expone).

## Notas importantes
- La app usa el namespace por defecto en Pinecone para simplificar y evitar confusiones con filtros. No necesitas seleccionar namespace ni hash.
- El texto del fragmento se guarda en la metadata (`text_key="text"`) para que el retriever pueda reconstruir los `Document` al consultar.
- Embeddings por defecto: FastEmbed `BAAI/bge-small-en-v1.5` (sin torch). En macOS Intel, `onnxruntime` es suficiente; en Apple Silicon se sugiere `onnxruntime-silicon`.

## Solución de problemas
- 0 fragmentos recuperados al preguntar:
  - Asegúrate de haber pulsado “(Re)indexar PDF” y de no ver errores.
  - Si tu PDF es escaneado (imágenes), instala PyMuPDF para mejor extracción:
    ```bash
    poetry add pymupdf -n
    ```
    Reindexa y prueba nuevamente.
  - Baja el chunking (p. ej. 400/100) y reindexa.
- Límite de índices serverless en Pinecone (403):
  - Reutiliza un índice existente (por ejemplo `cv-index`) en lugar de crear uno nuevo, o elimina uno que no uses.
- Claves/credenciales:
  - Verifica que exportaste `GROQ_API_KEY` y `PINECONE_API_KEY` en la misma sesión donde ejecutas Poetry/Streamlit.

## Estructura relevante
- `src/my_chatbot/app_rag_cv.py`: aplicación Streamlit (carga/particiona documentos, embeddings, upsert directo a Pinecone y consulta).
- `pyproject.toml`: dependencias gestionadas por Poetry.

## Personalización
- Modelos Groq: seleccionables en la barra lateral.
- Embeddings: el código usa FastEmbed por defecto. Para cambiar, modifica `get_default_embeddings()` en `app_rag_cv.py`.
- Prompt: editable en la barra lateral (“Instrucciones del sistema”).

## Verificación rápida (opcional)
Comprueba que los embeddings funcionan en tu entorno:
```bash
poetry run python - <<'PY'
from langchain_community.embeddings import FastEmbedEmbeddings
E = FastEmbedEmbeddings(model_name='BAAI/bge-small-en-v1.5')
print('ok', len(E.embed_query('hola mundo')))
PY
```
