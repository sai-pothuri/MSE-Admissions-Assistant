# MSE Admissions Assistant

RAG-based chatbot answering prospective student questions for CMU's Master of Software Engineering (MSE) program, with a faculty-facing admin console for knowledge base management. See `CLAUDE.md` for the full architecture and design spec, and `plan.md` for the phased implementation roadmap.

## Status

Phase 2 — guardrail layer. The `/query` pipeline now runs: pre-classification (off-limits topic check) → query category classification → category-filtered retrieval → confidence gate → generation → numerical verification (tuition/deadlines only). Ingestion auto-tags each chunk's category via Claude rather than inheriting it from the source folder. No admin console or eval harness yet (see `plan.md`).

Guardrail config lives in `backend/app/config/`: category taxonomy (`taxonomy.py`), off-limits topics (`offlimits.yaml`, edit without touching code), and thresholds (`thresholds.py` — confidence gate cutoffs are provisional pending Phase 3's eval-based calibration).

## Local development setup

1. Copy `.env.example` to `.env` in `backend/` and fill in `ANTHROPIC_API_KEY` and `VOYAGE_API_KEY`.
2. Start Qdrant:
   ```
   cd infra && docker compose up -d
   ```
3. Create a virtualenv and install backend dependencies:
   ```
   cd backend
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```
4. Create the Qdrant collection (one-time, or after wiping Qdrant storage):
   ```
   python -m app.scripts.init_collection
   ```
5. Ingest the knowledge base (source PDFs live in `data/knowledge_base/general/`):
   ```
   python -m app.scripts.ingest ../data/knowledge_base
   ```
6. Run the API:
   ```
   uvicorn app.main:app --reload
   ```
7. Check the health endpoint:
   ```
   curl http://localhost:8000/health
   ```
8. Ask a question:
   ```
   curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"question": "What are the admission requirements for the MSE program?"}'
   ```

Run the test suite (all mocked, no live API calls or Qdrant needed):
```
cd backend && source .venv/bin/activate && pytest
```

## Repo structure

```
/backend        # FastAPI app: ingestion, retrieval, guardrails, generation, admin API
/frontend       # Chatbot UI + admin console (Phase 5)
/data           # git-tracked knowledge base source PDFs, by category
/infra          # Docker Compose, Qdrant config, deployment scripts
/eval           # eval question set + harness (Phase 3)
```
