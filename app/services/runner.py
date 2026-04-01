"""
Evaluation runner — orchestrates a full eval run across selected modules,
updates progress, and stores results.
"""
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.evaluation import Evaluation
from ..models.model_endpoint import ModelEndpoint
from ..models.results import BenchmarkResult, ToolCallResult, QualityResult, PerformanceResult
from ..services.llm_client import LLMClient
from ..eval.capability import run_capability_eval
from ..eval.tool_call import run_tool_call_eval
from ..eval.quality import run_quality_eval
from ..eval.performance import run_performance_eval


async def run_evaluation(db: AsyncSession, evaluation_id: int):
    """Execute a full evaluation run."""
    # Fetch evaluation and endpoint
    eval_obj = await db.get(Evaluation, evaluation_id)
    if not eval_obj:
        return
    endpoint = await db.get(ModelEndpoint, eval_obj.model_endpoint_id)
    if not endpoint:
        eval_obj.status = "failed"
        eval_obj.error = "Model endpoint not found"
        await db.commit()
        return

    # Create LLM client
    client = LLMClient(
        base_url=endpoint.base_url,
        api_key=endpoint.api_key,
        model_id=endpoint.model_id,
    )

    eval_obj.status = "running"
    eval_obj.started_at = datetime.now(timezone.utc)
    await db.commit()

    modules = eval_obj.modules or []
    total_modules = len(modules)
    completed = 0

    try:
        # 1. Capability benchmarks
        if "capability" in modules:
            results = await run_capability_eval(client)
            for r in results:
                db.add(BenchmarkResult(
                    evaluation_id=evaluation_id,
                    benchmark_name=r["benchmark_name"],
                    score=r["score"],
                    total=r["total"],
                    correct=r["correct"],
                    details=r.get("details"),
                ))
            completed += 1
            eval_obj.progress = int((completed / total_modules) * 100)
            await db.commit()

        # 2. Tool call accuracy
        if "tool_call" in modules:
            tc_result = await run_tool_call_eval(client)
            db.add(ToolCallResult(
                evaluation_id=evaluation_id,
                accuracy=tc_result["accuracy"],
                precision=tc_result["precision"],
                recall=tc_result["recall"],
                false_positive_rate=tc_result["false_positive_rate"],
                total_tests=tc_result["total_tests"],
                details=tc_result.get("details"),
            ))
            completed += 1
            eval_obj.progress = int((completed / total_modules) * 100)
            await db.commit()

        # 3. Quality metrics
        if "quality" in modules:
            q_result = await run_quality_eval(client)
            db.add(QualityResult(
                evaluation_id=evaluation_id,
                hallucination_rate=q_result["hallucination_rate"],
                answer_relevancy=q_result["answer_relevancy"],
                faithfulness=q_result["faithfulness"],
                total_tests=q_result["total_tests"],
                details=q_result.get("details"),
            ))
            completed += 1
            eval_obj.progress = int((completed / total_modules) * 100)
            await db.commit()

        # 4. Performance benchmarking
        if "performance" in modules:
            p_result = await run_performance_eval(
                client,
                cost_per_1k_input=endpoint.cost_per_1k_input,
                cost_per_1k_output=endpoint.cost_per_1k_output,
            )
            db.add(PerformanceResult(
                evaluation_id=evaluation_id,
                latency_p50=p_result["latency_p50"],
                latency_p95=p_result["latency_p95"],
                latency_p99=p_result["latency_p99"],
                tokens_per_sec=p_result["tokens_per_sec"],
                cost_per_1k=p_result["cost_per_1k"],
                concurrent_users=p_result["concurrent_users"],
                total_requests=p_result["total_requests"],
                error_rate=p_result["error_rate"],
                details=p_result.get("details"),
            ))
            completed += 1
            eval_obj.progress = int((completed / total_modules) * 100)
            await db.commit()

        eval_obj.status = "completed"
        eval_obj.progress = 100
        eval_obj.completed_at = datetime.now(timezone.utc)
        await db.commit()

    except Exception as e:
        eval_obj.status = "failed"
        eval_obj.error = str(e)[:2000]
        await db.commit()
        raise
