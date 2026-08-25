from fastapi import FastAPI

from app.api.routes import health

app = FastAPI(title="MSE Admissions Assistant")

app.include_router(health.router)
