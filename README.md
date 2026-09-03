# MSE Admissions Assistant

RAG-based chatbot answering prospective student questions for CMU's Master of Software Engineering (MSE) program, with a faculty-facing admin console for knowledge base management. See `CLAUDE.md` for the full architecture and design spec, and `plan.md` for the phased implementation roadmap.

## Status

Phase 4 — admin console backend (API only, no UI yet — see `plan.md`). The `/query` pipeline runs: pre-classification (off-limits topic check) → query category classification → category-filtered retrieval (falling back to unfiltered whenever it scores higher, not only when the filter returns nothing) → confidence gate → generation → numerical verification (tuition/deadlines only). Ingestion auto-tags each chunk's category via Claude rather than inheriting it from the source folder.

Guardrail config lives in `backend/app/config/`: category taxonomy (`taxonomy.py`), off-limits topics (`offlimits.yaml`, edit without touching code), and thresholds (`thresholds.py` — `min_top1_score=0.40`, calibrated against the eval set in Phase 3; see `plan.md` for the reasoning).

## Admin console API

Password-protected (`ADMIN_PASSWORD`/`ADMIN_SESSION_SECRET` in `.env`), session via signed cookie:

```
POST   /admin/login                        {"password": "..."}  -> sets session cookie
POST   /admin/logout
GET    /admin/files                        list uploaded PDFs
POST   /admin/files/upload                 multipart file -> git commit + index into staging
PUT    /admin/files/{filename}              replace + re-index into staging (409 + confirm=true if it has manual re-tags)
DELETE /admin/files/{filename}              remove file + its chunks from staging and prod
GET    /admin/files/edit-log                git history + Qdrant-only changes (retags, promotions)
GET    /admin/chunks/preview?filename=...    dry-run extract+chunk+auto-tag, no indexing
GET    /admin/chunks?filename=...&collection=staging|prod
PATCH  /admin/chunks/{point_id}?category=... manual re-tag (in place, sets auto_tagged=false)
POST   /admin/staging/promote/{filename}     copy a file's staged chunks to prod
```

Uploads always land in the staging collection first — nothing reaches the live `/query` pipeline (which only reads prod) until explicitly promoted.

## Evaluation

`eval/questions.jsonl` holds 76 questions spanning normal, out-of-scope, injection, numerical-trap, ambiguous, off-limits-disguised, adversarial, and format-breaking cases, grounded in the real source PDFs. Run it against a live stack (Qdrant populated, API keys set):

```
python eval/harness.py                        # runs every question, writes eval/results/
python eval/calibrate_threshold.py             # sweeps confidence-gate thresholds against a harness run
```

Every guardrail/prompt/threshold change should be re-checked against this set before being considered done, per `CLAUDE.md`.

## Local development setup

1. Copy `.env.example` to `.env` in `backend/` and fill in `ANTHROPIC_API_KEY`, `VOYAGE_API_KEY`, `ADMIN_PASSWORD`, and `ADMIN_SESSION_SECRET` (e.g. `openssl rand -hex 32`).
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
4. Create the Qdrant collections (prod + staging; one-time, or after wiping Qdrant storage):
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
