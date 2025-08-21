# TP3 – Chatbot de Agentes por Persona (Streamlit + LangChain + Pinecone)

## Objetivo
Responder preguntas sobre múltiples CVs con un agente por persona. Si la query no nombra a nadie, responde el agente del Alumno. Si se nombran varias personas, fusiona contextos de cada una.

## Requisitos
- Python 3.10–3.11 (Poetry)
- Variables de entorno:
  - `GROQ_API_KEY`
  - `PINECONE_API_KEY`

## Instalación (Poetry)
```bash
poetry install
# opcional si usas PDFs escaneados (mejor extracción)
poetry add pymupdf -n
```
Cargar variables (mismo shell):
```bash
export GROQ_API_KEY="..."
export PINECONE_API_KEY="..."
# o copiar desde .env-local
```

## Ejecutar
```bash
poetry run streamlit run ../TP3/my-chatbot-agents/src/my_chatbot_agents/app_agents.py
```

## Uso
1) En la barra lateral configura:
   - Índice Pinecone (ej: `cv-index-agentes`)
   - Lista de personas separadas por coma (ej: `Alumno,Martin,Camila`)
   - Alias del Alumno (opcional), ej: `Luciano`
   - k por persona y temperatura (0 recomendado)
   - (Opcional) Limpiar namespace antes de reindexar
2) Sube 1 CV por persona (PDF o DOCX) y pulsa “(Re)indexar CVs”.
3) Pregunta. Si no nombras a nadie → responde Alumno. Si nombras una o más personas → responde combinando contextos.

### Convenciones de nombres (match robusto)
- La asignación de CV a persona se infiere del nombre del archivo (sin extensión):
  - Coincidencia por nombre completo, con o sin separadores.
  - camelCase se parte automáticamente (`MartinTorres` → `Martin Torres`).
  - También se acepta el primer nombre (ej: `Martin`).
- Ejemplos con lista `Alumno,Martin,Camila`:
  - `CV_Alumno.pdf` → Alumno
  - `MartinTorres.pdf`, `Martin_Torres.pdf` o `Martin.pdf` → Martin
  - `CamilaRivas.docx`, `Camila_Rivas.docx` o `Camila.docx` → Camila
- Si ningún nombre coincide, el CV se asigna a `Alumno`.

### Enrutado de preguntas
- Si la pregunta contiene “Martin”, “Martin Torres” o “MartinTorres” → agente de Martin.
- Si contiene “Camila” / “Camila Rivas” / “CamilaRivas” → agente de Camila.
- Si contiene “Luciano” y definiste Alias del Alumno = Luciano → agente del Alumno.
- Si contiene múltiples nombres (ej. “Luciano y Martin”) → combina contextos de Alumno y Martin.
- Si no contiene nombres → usa agente del Alumno.

## Detalles técnicos
- Splitter: RecursiveCharacterTextSplitter (400/100 por defecto).
- Embeddings: FastEmbed `BAAI/bge-small-en-v1.5` (sin torch).
- Vector DB: Pinecone, namespace por defecto (`""`).
- Upsert directo con metadata: `{ person: <nombre>, text: <chunk> }` y `text_key="text"` para recuperar el contenido.

## Buenas prácticas
- Mantén un único índice por modelo de embeddings.
- Si cambias chunking o reindexas el mismo CV, marca “Limpiar namespace antes de reindexar”.
- Evita nombres ambiguos (subcadenas). Prefiere `NombreApellido` o `Nombre_Apellido`.

## Solución de problemas
- “No trae fragmentos”: reindexa; si el PDF es escaneado, instala `pymupdf`.
- “IDs no ASCII” en Pinecone: los IDs se normalizan internamente, pero revisa nombres con acentos extraños.
- “No enruta al Alumno”: define el alias real del Alumno (ej. `Luciano`) en la barra lateral.
