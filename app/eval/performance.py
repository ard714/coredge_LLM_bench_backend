"""
Performance benchmarking module — measures latency (p50/p95/p99),
tokens/sec throughput, and cost per 1K tokens under load.
"""
import asyncio
import time
import random
import numpy as np
from ..services.llm_client import LLMClient
from ..config import settings

# Prompts of varying length to simulate realistic load
LOAD_PROMPTS = [
    "Explain the concept of machine learning in one sentence.",
    "What are the main differences between Python and JavaScript?",
    "Summarize the key principles of object-oriented programming.",
    "Write a haiku about technology.",
    "Describe the water cycle in simple terms.",
    "What is the difference between a stack and a queue?",
    "Explain what an API is to a non-technical person.",
    "List three benefits of cloud computing.",
    "What is the time complexity of binary search?",
    "Explain the CAP theorem in distributed systems.",
]


async def _single_request(client: LLMClient, prompt: str) -> dict:
    """Make a single request and measure latency/tokens."""
    start = time.perf_counter()
    try:
        response = await client.chat([{"role": "user", "content": prompt}], max_tokens=256)
        elapsed = time.perf_counter() - start
        return {
            "success": True,
            "latency": response.get("latency", elapsed),
            "input_tokens": response.get("input_tokens", 0),
            "output_tokens": response.get("output_tokens", 0),
        }
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {
            "success": False,
            "latency": elapsed,
            "input_tokens": 0,
            "output_tokens": 0,
            "error": str(e),
        }


async def run_performance_eval(
    client: LLMClient,
    concurrent_users: int | None = None,
    cost_per_1k_input: float = 0.0,
    cost_per_1k_output: float = 0.0,
) -> dict:
    """Run performance benchmark with concurrent requests."""
    concurrent = concurrent_users or settings.DEFAULT_CONCURRENT_USERS
    total_requests = concurrent * 3  # 3 rounds per user

    # Create tasks
    tasks = []
    for i in range(total_requests):
        prompt = LOAD_PROMPTS[i % len(LOAD_PROMPTS)]
        tasks.append(_single_request(client, prompt))

    # Run in batches of `concurrent` size
    all_results = []
    for batch_start in range(0, len(tasks), concurrent):
        batch = tasks[batch_start:batch_start + concurrent]
        batch_results = await asyncio.gather(*batch)
        all_results.extend(batch_results)

    # Analyze results
    successful = [r for r in all_results if r["success"]]
    failed = [r for r in all_results if not r["success"]]

    if not successful:
        return {
            "latency_p50": 0,
            "latency_p95": 0,
            "latency_p99": 0,
            "tokens_per_sec": 0,
            "cost_per_1k": 0,
            "concurrent_users": concurrent,
            "total_requests": total_requests,
            "error_rate": 1.0,
            "details": {"error": "All requests failed"},
        }

    latencies = [r["latency"] for r in successful]
    total_output_tokens = sum(r["output_tokens"] for r in successful)
    total_input_tokens = sum(r["input_tokens"] for r in successful)
    total_time = sum(latencies)

    # Percentile calculations
    latency_arr = np.array(latencies)
    p50 = float(np.percentile(latency_arr, 50))
    p95 = float(np.percentile(latency_arr, 95))
    p99 = float(np.percentile(latency_arr, 99))

    # Throughput: tokens generated per second (wall-clock)
    tokens_per_sec = total_output_tokens / total_time if total_time > 0 else 0

    # Cost calculation
    input_cost = (total_input_tokens / 1000) * cost_per_1k_input
    output_cost = (total_output_tokens / 1000) * cost_per_1k_output
    total_tokens = total_input_tokens + total_output_tokens
    cost_per_1k = ((input_cost + output_cost) / total_tokens * 1000) if total_tokens > 0 else 0

    error_rate = len(failed) / len(all_results) if all_results else 0

    return {
        "latency_p50": round(p50, 4),
        "latency_p95": round(p95, 4),
        "latency_p99": round(p99, 4),
        "tokens_per_sec": round(tokens_per_sec, 2),
        "cost_per_1k": round(cost_per_1k, 6),
        "concurrent_users": concurrent,
        "total_requests": total_requests,
        "error_rate": round(error_rate, 4),
        "details": {
            "successful_requests": len(successful),
            "failed_requests": len(failed),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "avg_latency": round(sum(latencies) / len(latencies), 4),
            "min_latency": round(min(latencies), 4),
            "max_latency": round(max(latencies), 4),
        },
    }
