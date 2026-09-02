from fastapi import FastAPI

from app.api.routes import admin_auth, admin_chunks, admin_files, admin_staging, health, query

app = FastAPI(title="MSE Admissions Assistant")

app.include_router(health.router)
app.include_router(query.router)
app.include_router(admin_auth.router)
app.include_router(admin_files.router)
app.include_router(admin_chunks.router)
app.include_router(admin_staging.router)
