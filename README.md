# Code Charlie — Streamlit App

Standalone compliance research chatbot for building codes (DBC, CIBSE, EN 81, BCO, ASME/ADA/IBC, HTM, ISO, DoH, BMU, CSI, Machinery Directive).

Port of the **Code Charlie** agent that lives inside the KARR-AI dashboard, packaged as a single-file Streamlit app for quick access without spinning up the full KARR-AI stack. Shares the same Supabase project as KARR-AI — KARR-AI remains the source of truth for compliance embeddings.

## Architecture

- **LangGraph agent** in [agent/](agent/) — `classify_and_scope` → `code_charlie_compliance_rag` ReAct loop. Identical retrieval/rerank/citation logic to KARR-AI's Code Charlie.
- **Models**: `gpt-5.5` (ReAct), `gpt-5.4-nano` (classifier / rerank / titles), `text-embedding-3-large` (retrieval). Configurable via env vars.
- **Database**: shared Supabase project with KARR-AI.
  - `compliance_embeddings` — read-only, owned by KARR-AI
  - `code_charlie_streamlit_sessions` — created by [migrations/001_code_charlie_streamlit_sessions.sql](migrations/001_code_charlie_streamlit_sessions.sql), sidebar metadata for this app
  - `checkpoints` / `checkpoint_writes` — LangGraph state, auto-created by PostgresSaver on first run
- **Auth**: single password gate ([lib/gate.py](lib/gate.py)). All sessions owned by a fixed `GATE_USER_ID` UUID.

## Local setup

```powershell
cd C:\Users\karra\Desktop\Code-Charlie

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

Copy-Item .env.example .env
# Edit .env — paste values from KARR-AI's apps/api/.env (same Supabase + OpenAI),
# set a fresh GATE_PASSWORD, and generate one UUID for GATE_USER_ID:
#   python -c "import uuid; print(uuid.uuid4())"

# Apply the migration to your shared Supabase project (run in Supabase SQL editor):
#   migrations/001_code_charlie_streamlit_sessions.sql

streamlit run app.py
```

Open http://localhost:8501, enter the gate password, and start chatting.

## Streamlit Cloud deploy

1. Push this repo to a public GitHub repository.
2. Sign in at https://share.streamlit.io and click **New app**.
3. Point it at this repo, branch `main`, file `app.py`.
4. Open **Settings → Secrets** and paste the contents of [.streamlit/secrets.toml.example](.streamlit/secrets.toml.example) with real values filled in.
5. Deploy. The app will be reachable at `https://<your-slug>.streamlit.app`.

The `code_charlie_streamlit_sessions` table must exist in the shared Supabase project before first run.

## Embedding updates

KARR-AI is the source of truth for `compliance_embeddings`. When you ingest new compliance documents from KARR-AI's pipeline, the Streamlit app sees the updates immediately — both apps read from the same table.

## File map

```
app.py                              # Streamlit entry (gate + sidebar + chat)
requirements.txt
.env.example
.streamlit/
  config.toml                       # theme + server settings
  secrets.toml.example              # Streamlit Cloud secrets template
agent/
  graph.py                          # LangGraph builder + invoke helpers
  state.py                          # CodeCharlieState schema
  checkpointer.py                   # PostgresSaver singleton
  messages.py                       # message dict helpers
  nodes/
    routing.py                      # classify_and_scope node
    compliance.py                   # ReAct compliance RAG node
    compliance_helpers.py           # search / lookup / formatters
    context_expansion.py            # adjacent/sibling/parent chunk fetch
    query_rewriting.py              # multi-query, meta detect, HyDE
    keyword_filter.py               # has_compliance_keywords
core/
  config.py                         # pydantic Settings
  supabase_client.py                # service-role client
lib/
  gate.py                           # password gate
  sessions.py                       # sessions table CRUD
migrations/
  001_code_charlie_streamlit_sessions.sql
```
