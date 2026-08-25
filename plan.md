# MSE Admissions Assistant — Implementation Plan

## Context

CLAUDE.md fully specifies a RAG-based MSE admissions chatbot with a faculty admin console, but the repo currently contains only that spec file — no code exists yet. This is a greenfield build. The goal of this plan is to translate the CLAUDE.md spec into a sequenced, buildable roadmap: what to build first so we get a working end-to-end system quickly, and how later phases (guardrails, eval, admin console, frontend, deployment) layer on top without requiring rework.

**Decisions confirmed with the user:**
- Frontend: **React + Vite + TypeScript**.
- Model split: **Claude Haiku** for high-volume classification calls (auto-tagging, pre-classification, query classification), **Claude Sonnet** for final answer generation.
- Git workflow: each phase gets its own branch off `main`; when a phase's work is complete, open a PR for the user to review and merge (not auto-merged).
- Source documents: the real content is a general student handbook, an FAQ doc, and details on two programs within the department — none of it pre-sorted into the 5 category folders. Decision: don't force manual category sorting. All source docs land in a new `data/knowledge_base/general/` intake folder; folder placement is organizational only. The authoritative category per chunk comes from Phase 2's per-chunk auto-tagging into the Qdrant payload, regardless of which folder the source file sits in. The FAQ (currently .docx) will be converted to PDF by the user before ingestion — the pipeline stays PDF-only (PyMuPDF) per CLAUDE.md, no docx extraction path added.

## Guiding principle for sequencing

Dependency chain: **config/taxonomy → ingestion → retrieval → generation → guardrails wrapping retrieval+generation → eval harness to calibrate guardrails → admin console (needs staging collection + git-log design already proven) → frontend → containerization/deployment.**

Guardrails and the admin console both sit on primitives (ingestion, retrieval, Qdrant collection design) that must be stable first. Frontend is deferred until the API surface is stable so it isn't chasing a moving target. Each phase produces something runnable/testable before the next phase depends on it.

---

## Phase 0 — Repo & environment scaffolding

**Goal:** empty repo → runnable skeleton, no business logic.

**Deliverables:** monorepo layout per CLAUDE.md, Python project setup (deps + lint/format/type-check), Qdrant running locally via Docker Compose, FastAPI app booting with a health check, `.env.example`.

```
backend/pyproject.toml            # fastapi, uvicorn, pymupdf, qdrant-client, voyageai, anthropic, pydantic-settings, pytest, ruff, mypy
backend/app/main.py
backend/app/api/routes/health.py
backend/app/config/settings.py    # pydantic BaseSettings: secrets, URLs, collection names
infra/docker-compose.yml          # qdrant only, to start
data/knowledge_base/{admissions,curriculum,tuition,deadlines,faculty,general}/.gitkeep
eval/.gitkeep
.gitignore
.env.example
README.md
```

Note: `git init` + first commit is a state-changing action the user runs (or approves explicitly), not something done silently.

**Status: done** — committed and pushed to `main` (`https://github.com/sai-pothuri/MSE-Admissions-Assistant`). The `general/` folder above was added after this phase closed (see Source documents decision) and will be created as a small follow-up commit on `main`, not a new phase branch.

---

## Phase 1 — Minimal vertical slice: ingestion → retrieval → generation

**Branch:** `phase-1-ingestion-retrieval-generation`

**Goal:** prove the core RAG loop end-to-end against the real source documents (handbook, FAQ, two program detail docs — all in `data/knowledge_base/general/`). **No admin console, no guardrails yet** — category is folder-derived as a placeholder, so everything will show up tagged `general` until Phase 2's real per-chunk auto-tagging replaces it; no off-limits check, no confidence gate, no numerical verification. Smallest possible working system.

**Deliverables:** CLI-driven ingestion of the source PDFs into a single Qdrant collection; a `POST /query` endpoint doing unfiltered vector search + Claude generation with citations; manual smoke test proving grounded, cited answers.

```
backend/app/clients/{qdrant_client,voyage_client,anthropic_client}.py
backend/app/services/ingestion/pdf_extraction.py     # PyMuPDF: text + page_number per block
backend/app/services/ingestion/chunking.py            # structure-aware, ~300-500 tokens, overlap
backend/app/services/ingestion/embedding.py            # Voyage AI wrapper
backend/app/services/ingestion/indexer.py               # extract → chunk → embed → upsert
backend/app/scripts/ingest.py                            # CLI: python -m app.scripts.ingest data/knowledge_base
backend/app/scripts/init_collection.py                    # creates Qdrant collection, correct vector size/distance
backend/app/services/retrieval/vector_search.py            # top_k similarity search, no filter yet
backend/app/services/generation/prompt_templates.py
backend/app/services/generation/generator.py                 # Claude Sonnet, low temp, grounding + citation system prompt
backend/app/models/schemas.py                                  # QueryRequest, QueryResponse, ChunkPayload
backend/app/api/routes/query.py                                 # POST /query
backend/tests/unit/test_chunking.py
backend/tests/unit/test_indexer.py                                # mocked Qdrant/Voyage
backend/tests/integration/test_query_endpoint.py
```

### Phase 1 task order
1. Get the source PDFs (handbook, FAQ converted to PDF, two program docs) into `data/knowledge_base/general/`.
2. Create `phase-1-ingestion-retrieval-generation` branch off `main`.
3. Core backend deps already installed from Phase 0; reactivate venv.
4. `infra/docker-compose.yml` with Qdrant; bring it up locally.
5. `backend/app/config/settings.py` already exists from Phase 0 — extend if needed.
6. Client wrappers (Qdrant, Voyage, Anthropic).
7. `init_collection.py` — vector size matched to chosen Voyage embedding model, cosine distance.
8. `pdf_extraction.py` — unit test against a sample PDF.
9. `chunking.py` — unit test on extracted text.
10. `embedding.py` — Voyage embed wrapper (document + query variants).
11. `indexer.py` — folder-derived category placeholder (`general` for all docs at this stage); real auto-tagging deferred to Phase 2.
12. `ingest.py` — run against the real source PDFs into local Qdrant.
13. `vector_search.py` — unfiltered top_k search.
14. `prompt_templates.py` + `generator.py` — Claude Sonnet, context-only + citation system prompt.
15. `schemas.py` + `query.py` — wire retrieval+generation into `POST /query`.
16. Manual smoke test: real questions against the handbook/FAQ/program docs via curl/HTTPie, confirm grounded answers with correct citations.
17. Unit + integration tests.
18. README section: bring up Qdrant, run ingestion, start API, smoke-test.
19. Push branch, open PR into `main` for review.

---

## Phase 2 — Guardrail layer

**Branch:** `phase-2-guardrails`

**Goal:** wrap Phase 1 with the guardrails that are the actual engineering value-add of this project, each independently unit-testable.

**Deliverables:** centralized config for taxonomy/off-limits/thresholds; real Claude-Haiku-based auto-tagging replacing the folder-derived placeholder; full per-query pipeline: pre-classify → classify category → filtered search → confidence gate → generate → conditional numerical verification.

```
backend/app/config/taxonomy.py             # CATEGORIES = admissions, curriculum, tuition, deadlines, faculty, other
backend/app/config/offlimits.yaml           # explicit, editable off-limits topics + example phrasings + redirect copy
backend/app/config/thresholds.py             # confidence_threshold, score_gap_threshold, top_k, numerical-verification category list
backend/app/services/ingestion/auto_tagging.py    # Claude Haiku, one call per chunk
backend/app/services/guardrails/pre_classifier.py     # off-limits check, runs BEFORE retrieval (Claude Haiku)
backend/app/services/guardrails/query_classifier.py    # category classification for Qdrant filter (Claude Haiku)
backend/app/services/guardrails/confidence_gate.py       # top-1 score + score gap vs threshold
backend/app/services/guardrails/numerical_verification.py  # extract numbers from answer, verify literal match in cited chunk text
backend/app/services/guardrails/fallback.py                 # fallback/decline response builder
backend/tests/unit/test_pre_classifier.py
backend/tests/unit/test_query_classifier.py
backend/tests/unit/test_confidence_gate.py
backend/tests/unit/test_numerical_verification.py
backend/tests/unit/test_auto_tagging.py
```

`/query` calls these stages in sequence, short-circuiting on pre-classifier rejection or confidence-gate failure (generation model is never called in those cases, per spec).

**Testability pattern:** guardrail functions take explicit typed config/values as arguments rather than reaching into global config internally, e.g. `confidence_gate.evaluate(top1_score: float, score_gap: float, config: ConfidenceConfig) -> GateResult` — tests pass synthetic scores, no network calls needed. Same pattern for `pre_classifier.check(query, offlimits_config, classify_fn)` so the Claude call is injectable/mockable.

**Numerical verification approach:** regex-based number extraction (currency, percentages, dates) with literal substring matching against cited chunk text — no semantic normalization (e.g. "$50,000" won't match "fifty thousand"). Flagging this as the practical, low-complexity default; can be revisited if false positives show up in eval.

---

## Phase 3 — Eval harness & threshold calibration

**Branch:** `phase-3-eval-harness`

**Goal:** build the eval set, then use it to calibrate the confidence threshold introduced in Phase 2.

```
eval/questions.jsonl          # {id, question, category_expected, type, expected_behavior: answer|decline}
                               # type: normal | out_of_scope | injection | numerical_trap | ambiguous | off_limits_disguised | adversarial | format_breaking
eval/harness.py                 # runs each question through the full pipeline; logs category, confidence score, answered/declined, latency
eval/calibrate_threshold.py       # sweeps threshold values, reports FP/FN rate, recommends cutoff
```

Scoring approach: manual expected-behavior labels drive pass/fail for off-limits/injection/decline-type questions (deterministic); factual correctness on "should answer" questions is judged via a lightweight LLM-as-judge pass against the source PDF content, since manual grading of 50–100 answers isn't practical to keep re-running. This becomes the fixed baseline every later guardrail/prompt change is checked against.

---

## Phase 4 — Admin console backend (API only)

**Branch:** `phase-4-admin-console-backend`

**Goal:** git-backed file management, chunk preview/re-tag, staging/promotion, edit log, re-indexing, and the swappable auth module — all server-side, no UI yet.

```
backend/app/services/admin/file_manager.py     # upload/replace/delete + git commit per change
backend/app/services/admin/git_log.py            # parses git history into structured edit-log entries
backend/app/services/admin/manual_edit_log.py      # supplemental log for Qdrant-only changes (re-tags, promotions)
backend/app/services/admin/chunk_preview.py           # dry-run extract+chunk+auto-tag, no upsert
backend/app/services/admin/retag.py                      # in-place Qdrant payload update, auto_tagged: false
backend/app/services/admin/staging.py                      # promotion: staging collection → prod collection
backend/app/services/admin/reindex.py                        # delete-by-source_file, re-ingest; warns if manual corrections exist
backend/app/auth/base.py                                     # AuthProvider interface: authenticate, create_session, validate_session
backend/app/auth/password_auth.py                               # current implementation
backend/app/auth/session.py
backend/app/api/routes/{admin_auth,admin_files,admin_chunks,admin_staging}.py
backend/app/api/deps.py                                              # current_user dependency wrapping AuthProvider
backend/tests/unit/test_reindex_warning.py
backend/tests/integration/test_admin_upload_flow.py
```

Key design points:
- **Staging vs prod**: two Qdrant collections (`mse_kb_staging`, `mse_kb_prod`); ingestion targets staging by default, promotion endpoint copies/moves points.
- **Auth boundary**: every admin route depends on `AuthProvider` via FastAPI dependency injection, never calls `password_auth` directly — swapping to CMU SSO later means replacing one module + rewiring one dependency, not touching route handlers. Password auth is a single shared faculty password for now (matches CLAUDE.md's "simple password auth"), sessions via signed cookie.
- **Git commit message format**: structured, parseable trailer format per change, e.g. `[admissions] upload: fall2026-requirements.pdf` with a trailer block (`Action:`, `Category:`, `File:`) so `git_log.py` can parse it reliably rather than pattern-matching free text.
- **Re-index warning**: before deleting old chunks, check if any had `auto_tagged: false`; if so, surface a clear warning (via API response the frontend renders as a confirm modal) before proceeding.

---

## Phase 5 — Frontend (React + Vite + TypeScript)

**Branch:** `phase-5-frontend`

**Goal:** chatbot UI + admin console UI, built once the backend API surface is stable.

```
frontend/package.json
frontend/src/pages/Chat/ChatWidget.tsx
frontend/src/pages/Admin/Login.tsx
frontend/src/pages/Admin/FileManager.tsx
frontend/src/pages/Admin/ChunkPreview.tsx
frontend/src/pages/Admin/StagingPromotion.tsx
frontend/src/pages/Admin/EditLog.tsx
frontend/src/api/client.ts
```

Chat widget: question input, answer + citations display, visible decline/fallback state (not a silent failure). Admin console: login, per-category file list/upload, chunk preview table with category-override dropdown, staging-vs-prod indicator, promote button, re-index button with manual-correction warning modal, edit-log viewer.

---

## Phase 6 — Containerization & deployment

**Branch:** `phase-6-containerization-deployment`

**Goal:** full Docker Compose orchestration for self-hosting on a CMU department server.

```
backend/Dockerfile
frontend/Dockerfile
infra/docker-compose.yml          # qdrant + backend + frontend, volumes for /data and qdrant storage
infra/qdrant/config.yaml
infra/scripts/deploy.sh
infra/scripts/init_collections.py
```

Frontend is built as a static bundle and served as its own container (or via a lightweight reverse proxy alongside the backend) — consistent with the single-server Docker Compose target, no separate hosting platform introduced.

---

## Phase 7 — Hardening & polish

**Branch:** `phase-7-hardening-polish`

- Structured logging for guardrail decisions (which stage rejected/approved, scores, category) — needed for debugging and the auditability goal of this project.
- Basic rate limiting on the public `/query` endpoint (not in CLAUDE.md's spec, but reasonable given self-hosting on a shared department server — flagging as an addition, can be dropped if out of scope).
- README completion, architecture diagram, runbook.
- Security review pass once there's a diff to review (the `security-review` skill fits here).
- Final eval run + documented threshold sign-off.

---

## Centralized config surfaces (per CLAUDE.md requirement)

| Config | Location | Format | Consumers |
|---|---|---|---|
| Category taxonomy | `backend/app/config/taxonomy.py` | Python constant/Enum | `auto_tagging.py`, `query_classifier.py`, admin re-tag dropdown, `eval/questions.jsonl` labels |
| Off-limits topics | `backend/app/config/offlimits.yaml` | YAML, loaded into a typed model at startup | `pre_classifier.py` only |
| Confidence threshold / score gap / top_k | `backend/app/config/thresholds.py` | Python constants, values updated by `eval/calibrate_threshold.py`, versioned in git | `confidence_gate.py`, `vector_search.py` |

---

## Verification approach

- **Phase 1**: manual smoke test with curl/HTTPie against the real source documents (handbook, FAQ, program docs) — ask questions spanning their actual content, confirm grounded answers with correct source + page citations; run `pytest backend/tests` for unit/integration coverage.
- **Phase 2**: unit tests per guardrail stage with synthetic inputs (no network calls); manual test of a few off-limits and low-confidence queries to confirm short-circuiting works (generation not invoked).
- **Phase 3**: run `eval/harness.py` end-to-end, review the FP/FN report from `calibrate_threshold.py`, confirm the chosen threshold is checked into `thresholds.py`.
- **Phase 4**: integration test of full upload → git commit → re-index → warning flow; verify staging→prod promotion moves points without affecting prod until promoted.
- **Phase 5**: manual browser testing of chat widget and admin console against a running backend (dev server), covering golden path + a decline case + a re-index-with-warning case.
- **Phase 6**: `docker compose up` from a clean checkout, confirm health checks pass and the full flow works end-to-end in containers.
- Every guardrail/prompt/threshold change from Phase 2 onward is re-checked against the Phase 3 eval set before being considered done, per CLAUDE.md.