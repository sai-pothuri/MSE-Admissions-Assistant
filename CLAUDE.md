# MSE Admissions Assistant

A RAG-based chatbot that answers prospective student questions for CMU's Master of Software Engineering (MSE) program, with a faculty-facing admin console for knowledge base management. Built as a portfolio/resume project demonstrating production-oriented system design (guardrails, staged deployments, auditability) — not just a RAG demo.

## Project Context

- Knowledge base is currently **3 PDFs** (small scale). Do not over-engineer around this — favor simple, correct implementations over premature scaling infrastructure. See "Scale-appropriate engineering" below.
- The real engineering value of this project is in the **guardrail and evaluation layers**, not retrieval tuning. When effort is ambiguous, prioritize guardrails, hallucination mitigation, and eval coverage over retrieval sophistication.
- This is a fresh repo, single monorepo, intended to eventually run self-hosted on a CMU department server.

## Tech Stack (decided)

- **Backend**: Python, FastAPI
- **Vector DB**: Qdrant, self-hosted via Docker (not Qdrant Cloud)
- **Embeddings**: Voyage AI
- **Generation**: Anthropic API (Claude)
- **Frontend**: not yet decided — propose an approach (React is a reasonable default given the ecosystem and admin console complexity) and confirm before scaffolding
- **Metadata / edit log / versioning storage**: Git-based. Source PDFs and their metadata live in a git-tracked directory structure; edits (upload/replace/delete/retag) are recorded as git commits with structured commit messages. Do not introduce Postgres/SQLite for this unless a specific feature genuinely requires relational queries git can't express.
- **Orchestration**: Docker Compose (single-server deployment target — do not introduce Kubernetes or multi-node assumptions)
- **Admin console auth**: simple password auth for now (not SSO/Shibboleth). Build this so it can be swapped for CMU SSO later without a rewrite — keep auth logic in one clearly isolated module.

## Repo Structure

Monorepo. Suggested layout (adjust as needed, but keep backend/frontend/data cleanly separated):

```
/backend        # FastAPI app: ingestion, retrieval, guardrails, generation, admin API
/frontend       # Chatbot UI + admin console
/data
  /knowledge_base
    /admissions
    /curriculum
    /tuition
    /deadlines
    /faculty
  # git-tracked; each subfolder holds source PDFs for that category
/infra          # Docker Compose, Qdrant config, deployment scripts
/eval           # eval question set + harness (see Evaluation section)
CLAUDE.md
README.md
```

## Core Architecture

### Ingestion pipeline
1. PDF uploaded (via admin console) into `/data/knowledge_base/<category>/`, committed to git.
2. Text extraction (PyMuPDF preferred over pypdf for better layout/table fidelity — tuition tables and deadline lists are hallucination-sensitive, extraction quality matters here).
3. Chunking: prefer structure-aware chunking (split on headings/paragraphs) over fixed-size splitting where feasible, to avoid splitting tables or deadline lists mid-content. Target ~300–500 tokens per chunk with modest overlap.
4. Auto-tagging: classify each chunk into one category (`admissions`, `curriculum`, `tuition`, `deadlines`, `faculty`, `other`) via a single Claude call per chunk. Store `category`, `auto_tagged: true/false`, `source_file`, `page_number` in the Qdrant payload.
5. Embed chunks (Voyage AI) and upsert into Qdrant with the payload above.

### Retrieval + generation pipeline (per query)
1. **Pre-classification / topic boundary check** — reject or redirect out-of-scope or off-limits questions (admissions predictions, legal/visa advice, faculty personal commentary, etc.) before retrieval runs. Maintain the off-limits list as an explicit, easily editable config, not buried in a prompt string.
2. **Query category classification** — classify the incoming question into the same category taxonomy, used to filter the Qdrant search.
3. **Filtered vector search** — search Qdrant with the category filter applied, `top_k` in the 3–5 range (small corpus; no need for larger k).
4. **Confidence gate** — check top-1 cosine similarity score (and ideally the score gap between top results) against a calibrated threshold. Below threshold → return a fallback response ("I don't have that information — contact ...") without calling the generation model. This threshold must be calibrated against the eval set, not guessed.
5. **Generation** — call Claude with retrieved chunks, a low temperature, and a system prompt that enforces: answer only from provided context, cite source, refuse if not clearly supported.
6. **Numerical verification pass** — for `tuition`/`deadlines` category answers specifically, verify every number in the generated answer literally appears in the cited chunk(s) before returning it. This is a deliberate, higher-cost step reserved for high-stakes categories — do not apply it universally if it meaningfully increases latency/cost without benefit elsewhere.

### Admin console
- File upload/replace/delete, writing to the git-tracked `/data/knowledge_base/` structure with a commit per change.
- Chunk preview: show extracted chunks with auto-assigned category before/after indexing.
- Manual re-tag: dropdown per chunk to override category. On change: update Qdrant payload in place (no re-embedding needed), set `auto_tagged: false`, log the change.
- Staging mode: use a **separate Qdrant collection** for staged content (not a payload flag on the production collection). Faculty can query against staged content before promoting. Promotion = copy/move points from staging collection to production collection.
- Edit log: derive primarily from git history on `/data/knowledge_base/`; supplement with an explicit log of Qdrant-only changes (e.g., manual re-tags that don't touch the source PDF) if git history alone doesn't capture them.
- Re-indexing behavior: re-indexing a file fully replaces that file's chunks (delete-by-`source_file`-filter in Qdrant, then re-run ingestion). Manual category corrections on the old chunks are NOT automatically preserved — surface a clear warning in the UI when re-indexing a file that has manually-corrected chunks.

## Scale-appropriate engineering

The knowledge base is 3 PDFs today. Do not add complexity that only pays off at much larger scale:
- No need for approximate nearest neighbor tuning, sharding, or Qdrant clustering.
- No need for a caching layer in front of retrieval.
- No need for async/batch embedding pipelines — synchronous, sequential processing of a few files is fine.
- Do design the retrieval/ingestion code so it isn't *hostile* to future growth (e.g., don't hardcode "exactly 3 files" assumptions), but don't build scaffolding for scale that doesn't exist yet.

## Evaluation

Maintain an eval set in `/eval` with ~50–100 questions spanning:
- Normal in-scope questions (all 5 categories)
- Out-of-scope but plausible questions
- Prompt injection / jailbreak attempts
- Numerical precision traps (tuition, deadlines)
- Ambiguous / multi-hop questions
- Off-limits topics disguised as neutral questions
- Adversarial/social-engineering phrasing
- Format-breaking inputs (very long input, non-English, typos)

Every change to the guardrail logic, prompts, or confidence threshold should be checked against this eval set before being considered done. When calibrating the confidence threshold, use this set to find the score cutoff that best separates "should answer" from "should decline," and report false positive/negative rates.

## Coding conventions

- Python: type hints throughout, FastAPI route handlers kept thin (business logic in service modules, not in route functions).
- Keep the off-limits topic list, category taxonomy, and confidence threshold as centralized config (not scattered magic strings/numbers) — these will be tuned iteratively.
- Every guardrail stage (pre-classification, confidence gate, numerical verification) should be independently unit-testable, not only testable end-to-end.
- Prefer explicit, readable code over cleverness — this is a portfolio project; clarity of design intent matters as much as functionality.

## What to ask before building

If a request from the user is ambiguous about scale, cost, or scope (e.g., "should this be async," "should we add caching," "should we support multiple embedding providers"), default to the simplest option consistent with the current 3-PDF scale, and note the tradeoff rather than silently over-building.
