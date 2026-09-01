from fastapi import FastAPI

from app.api.routes import health, query

app = FastAPI(title="MSE Admissions Assistant")

app.include_router(health.router)
app.include_router(query.router)
