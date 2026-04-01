"""Coredge LLM Benchmarking Platform — FastAPI entry point."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .database import init_db
from .routers import endpoints, evaluations, results


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title="Coredge LLM Benchmark",
    description="LLM evaluation and benchmarking platform for sovereign cloud inference",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174", "http://localhost:5173", "http://127.0.0.1:5174","https://coredge-llm-bench.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(endpoints.router)
app.include_router(evaluations.router)
app.include_router(results.router)


@app.get("/api/health")
async def health():
    return {"status": "ok", "app": "Coredge LLM Benchmark"}
