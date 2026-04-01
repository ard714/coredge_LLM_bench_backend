"""API routes for leaderboard, comparison, and Pareto chart data."""
from fastapi import APIRouter, Depends, Query, HTTPException
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from ..database import get_db
from ..models.model_endpoint import ModelEndpoint
from ..models.evaluation import Evaluation
from ..models.results import BenchmarkResult, ToolCallResult, QualityResult, PerformanceResult
from ..services.report_generator import get_report_generator

router = APIRouter(prefix="/api/results", tags=["results"])


async def _get_model_scores(db: AsyncSession, endpoint_id: int) -> dict | None:
    """Aggregate latest eval scores for a model."""
    # Get latest completed evaluation
    result = await db.execute(
        select(Evaluation)
        .where(Evaluation.model_endpoint_id == endpoint_id)
        .where(Evaluation.status == "completed")
        .order_by(Evaluation.completed_at.desc())
        .limit(1)
    )
    eval_obj = result.scalar_one_or_none()
    if not eval_obj:
        return None

    eval_id = eval_obj.id

    # Benchmarks — average score
    benchmarks = (await db.execute(
        select(BenchmarkResult).where(BenchmarkResult.evaluation_id == eval_id)
    )).scalars().all()
    bench_score = sum(b.score for b in benchmarks) / len(benchmarks) if benchmarks else 0

    # Tool call
    tc = (await db.execute(
        select(ToolCallResult).where(ToolCallResult.evaluation_id == eval_id)
    )).scalars().first()
    tc_score = tc.accuracy if tc else 0

    # Quality
    q = (await db.execute(
        select(QualityResult).where(QualityResult.evaluation_id == eval_id)
    )).scalars().first()
    quality_score = ((q.answer_relevancy + q.faithfulness + (1 - q.hallucination_rate)) / 3) if q else 0

    # Performance
    p = (await db.execute(
        select(PerformanceResult).where(PerformanceResult.evaluation_id == eval_id)
    )).scalars().first()

    return {
        "eval_id": eval_id,
        "capability_score": round(bench_score, 4),
        "tool_call_score": round(tc_score, 4),
        "quality_score": round(quality_score, 4),
        "latency_p50": p.latency_p50 if p else 0,
        "latency_p95": p.latency_p95 if p else 0,
        "tokens_per_sec": p.tokens_per_sec if p else 0,
        "cost_per_1k": p.cost_per_1k if p else 0,
        "error_rate": p.error_rate if p else 0,
        "benchmarks": {b.benchmark_name: round(b.score, 4) for b in benchmarks},
        "hallucination_rate": q.hallucination_rate if q else 0,
        "answer_relevancy": q.answer_relevancy if q else 0,
        "faithfulness": q.faithfulness if q else 0,
    }


@router.get("/leaderboard")
async def get_leaderboard(db: AsyncSession = Depends(get_db)):
    """Aggregated leaderboard across all models with completed evaluations."""
    endpoints = (await db.execute(
        select(ModelEndpoint).where(ModelEndpoint.is_active == True)
    )).scalars().all()

    leaderboard = []
    for ep in endpoints:
        scores = await _get_model_scores(db, ep.id)
        if scores:
            # Composite score: weighted average
            composite = (
                scores["capability_score"] * 0.3 +
                scores["tool_call_score"] * 0.2 +
                scores["quality_score"] * 0.3 +
                (1 - min(scores["latency_p50"], 5) / 5) * 0.1 +  # lower latency = better
                min(scores["tokens_per_sec"], 100) / 100 * 0.1   # higher throughput = better
            )
            leaderboard.append({
                "endpoint_id": ep.id,
                "model_name": ep.name,
                "model_id": ep.model_id,
                "provider": ep.provider,
                "composite_score": round(composite, 4),
                **scores,
            })

    leaderboard.sort(key=lambda x: x["composite_score"], reverse=True)
    for i, entry in enumerate(leaderboard):
        entry["rank"] = i + 1

    return leaderboard


@router.get("/compare")
async def compare_models(
    model_ids: str = Query(..., description="Comma-separated endpoint IDs"),
    db: AsyncSession = Depends(get_db),
):
    """Compare specific models side by side."""
    ids = [int(x.strip()) for x in model_ids.split(",") if x.strip()]
    comparison = []
    for eid in ids:
        ep = await db.get(ModelEndpoint, eid)
        if not ep:
            continue
        scores = await _get_model_scores(db, eid)
        if scores:
            comparison.append({
                "endpoint_id": eid,
                "model_name": ep.name,
                "model_id": ep.model_id,
                **scores,
            })
    return comparison


@router.get("/pareto")
async def get_pareto_data(db: AsyncSession = Depends(get_db)):
    """Cost vs Quality data for Pareto chart."""
    endpoints = (await db.execute(
        select(ModelEndpoint).where(ModelEndpoint.is_active == True)
    )).scalars().all()

    pareto = []
    for ep in endpoints:
        scores = await _get_model_scores(db, ep.id)
        if scores:
            quality_composite = (
                scores["capability_score"] * 0.4 +
                scores["tool_call_score"] * 0.2 +
                scores["quality_score"] * 0.4
            )
            pareto.append({
                "endpoint_id": ep.id,
                "model_name": ep.name,
                "cost_per_1k": scores["cost_per_1k"],
                "quality_score": round(quality_composite, 4),
                "capability_score": scores["capability_score"],
                "tokens_per_sec": scores["tokens_per_sec"],
            })

    return pareto


@router.get("/report/compare/pdf")
async def generate_comparison_report_pdf(
    model_ids: str = Query(..., description="Comma-separated model endpoint IDs"),
    db: AsyncSession = Depends(get_db),
):
    """Generate a PDF comparison report for multiple models."""
    ids = [int(x.strip()) for x in model_ids.split(",") if x.strip()]
    if not ids:
        raise HTTPException(status_code=400, detail="No valid model IDs provided")

    models_data = []
    for eid in ids:
        endpoint = await db.get(ModelEndpoint, eid)
        if not endpoint:
            continue

        scores = await _get_model_scores(db, eid)
        if not scores:
            continue

        # Calculate composite score
        composite = (
            scores["capability_score"] * 0.3 +
            scores["tool_call_score"] * 0.2 +
            scores["quality_score"] * 0.3 +
            (1 - min(scores["latency_p50"], 5) / 5) * 0.1 +
            min(scores["tokens_per_sec"], 100) / 100 * 0.1
        )

        models_data.append({
            "model_data": {
                "model_name": endpoint.name,
                "provider": endpoint.provider,
                "model_id": endpoint.model_id,
                "composite_score": round(composite, 4),
            },
            "scores": scores,
        })

    if not models_data:
        raise HTTPException(status_code=404, detail="No evaluation data found for specified models")

    # Generate PDF
    report_generator = get_report_generator()
    pdf_bytes = report_generator.generate_comparison_report(models_data)

    # Return as downloadable PDF
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=coredge-comparison-report.pdf"},
    )


@router.get("/report/{model_id}/pdf")
async def generate_model_report_pdf(
    model_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Generate a PDF performance report for a single model."""
    # Get model endpoint
    endpoint = await db.get(ModelEndpoint, model_id)
    if not endpoint:
        raise HTTPException(status_code=404, detail="Model endpoint not found")

    # Get scores
    scores = await _get_model_scores(db, model_id)
    if not scores:
        raise HTTPException(status_code=404, detail="No evaluation data found for this model")

    # Calculate composite score for display
    composite = (
        scores["capability_score"] * 0.3 +
        scores["tool_call_score"] * 0.2 +
        scores["quality_score"] * 0.3 +
        (1 - min(scores["latency_p50"], 5) / 5) * 0.1 +
        min(scores["tokens_per_sec"], 100) / 100 * 0.1
    )

    # Prepare model data
    model_data = {
        "model_name": endpoint.name,
        "provider": endpoint.provider,
        "model_id": endpoint.model_id,
        "composite_score": round(composite, 4),
    }

    # Generate PDF
    report_generator = get_report_generator()
    pdf_bytes = report_generator.generate_single_model_report(model_data, scores)

    # Return as downloadable PDF
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=coredge-report-{endpoint.name.replace(' ', '-').lower()}.pdf"},
    )
