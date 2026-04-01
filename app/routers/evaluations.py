"""API routes for launching and tracking evaluations."""
import asyncio
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from ..database import get_db, async_session
from ..models.evaluation import Evaluation
from ..models.model_endpoint import ModelEndpoint
from ..models.results import BenchmarkResult, ToolCallResult, QualityResult, PerformanceResult
from ..services.runner import run_evaluation

router = APIRouter(prefix="/api/evaluations", tags=["evaluations"])


class EvalCreate(BaseModel):
    model_endpoint_id: int
    modules: list[str]  # ["capability", "tool_call", "quality", "performance"]


async def _run_eval_background(eval_id: int):
    """Background task for running evaluation."""
    async with async_session() as db:
        await run_evaluation(db, eval_id)


@router.post("", status_code=201)
async def create_evaluation(
    data: EvalCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    # Validate endpoint
    endpoint = await db.get(ModelEndpoint, data.model_endpoint_id)
    if not endpoint:
        raise HTTPException(status_code=404, detail="Model endpoint not found")

    valid_modules = {"capability", "tool_call", "quality", "performance"}
    for m in data.modules:
        if m not in valid_modules:
            raise HTTPException(status_code=400, detail=f"Invalid module: {m}")

    evaluation = Evaluation(
        model_endpoint_id=data.model_endpoint_id,
        modules=data.modules,
        status="pending",
    )
    db.add(evaluation)
    await db.commit()
    await db.refresh(evaluation)

    # Start background eval
    background_tasks.add_task(_run_eval_background, evaluation.id)

    return {"id": evaluation.id, "status": "pending"}


@router.get("")
async def list_evaluations(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Evaluation).order_by(Evaluation.created_at.desc())
    )
    evals = result.scalars().all()

    items = []
    for e in evals:
        endpoint = await db.get(ModelEndpoint, e.model_endpoint_id)
        items.append({
            "id": e.id,
            "model_endpoint_id": e.model_endpoint_id,
            "model_name": endpoint.name if endpoint else "Unknown",
            "status": e.status,
            "modules": e.modules,
            "progress": e.progress,
            "created_at": str(e.created_at) if e.created_at else None,
            "started_at": str(e.started_at) if e.started_at else None,
            "completed_at": str(e.completed_at) if e.completed_at else None,
            "error": e.error,
        })
    return items


@router.get("/{eval_id}")
async def get_evaluation(eval_id: int, db: AsyncSession = Depends(get_db)):
    evaluation = await db.get(Evaluation, eval_id)
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    endpoint = await db.get(ModelEndpoint, evaluation.model_endpoint_id)

    # Fetch results
    benchmarks = (await db.execute(
        select(BenchmarkResult).where(BenchmarkResult.evaluation_id == eval_id)
    )).scalars().all()

    tool_calls = (await db.execute(
        select(ToolCallResult).where(ToolCallResult.evaluation_id == eval_id)
    )).scalars().all()

    quality = (await db.execute(
        select(QualityResult).where(QualityResult.evaluation_id == eval_id)
    )).scalars().all()

    performance = (await db.execute(
        select(PerformanceResult).where(PerformanceResult.evaluation_id == eval_id)
    )).scalars().all()

    return {
        "id": evaluation.id,
        "model_name": endpoint.name if endpoint else "Unknown",
        "model_id": endpoint.model_id if endpoint else None,
        "status": evaluation.status,
        "modules": evaluation.modules,
        "progress": evaluation.progress,
        "created_at": str(evaluation.created_at) if evaluation.created_at else None,
        "started_at": str(evaluation.started_at) if evaluation.started_at else None,
        "completed_at": str(evaluation.completed_at) if evaluation.completed_at else None,
        "error": evaluation.error,
        "results": {
            "benchmarks": [
                {
                    "benchmark_name": b.benchmark_name,
                    "score": b.score,
                    "total": b.total,
                    "correct": b.correct,
                    "details": b.details,
                }
                for b in benchmarks
            ],
            "tool_call": [
                {
                    "accuracy": t.accuracy,
                    "precision": t.precision,
                    "recall": t.recall,
                    "false_positive_rate": t.false_positive_rate,
                    "total_tests": t.total_tests,
                    "details": t.details,
                }
                for t in tool_calls
            ],
            "quality": [
                {
                    "hallucination_rate": q.hallucination_rate,
                    "answer_relevancy": q.answer_relevancy,
                    "faithfulness": q.faithfulness,
                    "total_tests": q.total_tests,
                    "details": q.details,
                }
                for q in quality
            ],
            "performance": [
                {
                    "latency_p50": p.latency_p50,
                    "latency_p95": p.latency_p95,
                    "latency_p99": p.latency_p99,
                    "tokens_per_sec": p.tokens_per_sec,
                    "cost_per_1k": p.cost_per_1k,
                    "concurrent_users": p.concurrent_users,
                    "total_requests": p.total_requests,
                    "error_rate": p.error_rate,
                    "details": p.details,
                }
                for p in performance
            ],
        },
    }
