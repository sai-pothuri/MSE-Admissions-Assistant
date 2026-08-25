# MSE Admissions Assistant

RAG-based chatbot answering prospective student questions for CMU's Master of Software Engineering (MSE) program, with a faculty-facing admin console for knowledge base management. See `CLAUDE.md` for the full architecture and design spec, and `plan.md` for the phased implementation roadmap.

## Status

Phase 0 — repo scaffolding. No business logic yet.

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
4. Run the API:
   ```
   uvicorn app.main:app --reload
   ```
5. Check the health endpoint:
   ```
   curl http://localhost:8000/health
   ```

## Repo structure

```
/backend        # FastAPI app: ingestion, retrieval, guardrails, generation, admin API
/frontend       # Chatbot UI + admin console (Phase 5)
/data           # git-tracked knowledge base source PDFs, by category
/infra          # Docker Compose, Qdrant config, deployment scripts
/eval           # eval question set + harness (Phase 3)
```
