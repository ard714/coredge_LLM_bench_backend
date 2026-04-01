"""
OpenAI-compatible LLM client that works with vLLM, LiteLLM, and OpenAI endpoints.
"""
import asyncio
import time
import random
import json
from openai import AsyncOpenAI
from ..config import settings


class LLMClient:
    """Unified client for OpenAI-compatible LLM endpoints."""

    def __init__(self, base_url: str, api_key: str | None, model_id: str):
        self.model_id = model_id
        self.base_url = base_url
        self.mock = settings.MOCK_MODE and (not api_key or api_key == "mock")
        if not self.mock:
            self.client = AsyncOpenAI(base_url=base_url, api_key=api_key or "no-key")
        else:
            self.client = None

    async def chat(self, messages: list[dict], temperature: float = 0.0,
                   max_tokens: int = 1024, tools: list[dict] | None = None) -> dict:
        """Send a chat completion request. Returns the full response dict."""
        if self.mock:
            return await self._mock_chat(messages, tools)

        kwargs = dict(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if tools:
            kwargs["tools"] = tools

        start = time.perf_counter()
        response = await self.client.chat.completions.create(**kwargs)
        elapsed = time.perf_counter() - start

        choice = response.choices[0]
        result = {
            "content": choice.message.content or "",
            "tool_calls": None,
            "latency": elapsed,
            "input_tokens": response.usage.prompt_tokens if response.usage else 0,
            "output_tokens": response.usage.completion_tokens if response.usage else 0,
        }

        if choice.message.tool_calls:
            result["tool_calls"] = [
                {
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments),
                }
                for tc in choice.message.tool_calls
            ]

        return result

    async def _mock_chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """Generate a mock response for demo/development."""
        await asyncio.sleep(random.uniform(0.1, 0.5))
        latency = random.uniform(0.2, 1.5)

        user_msg = messages[-1]["content"] if messages else ""

        # Mock tool calls if tools are provided
        if tools and random.random() > 0.2:
            tool = random.choice(tools)
            func = tool["function"]
            mock_args = {}
            if "parameters" in func and "properties" in func["parameters"]:
                for key, prop in func["parameters"]["properties"].items():
                    if prop.get("type") == "string":
                        mock_args[key] = "mock_value"
                    elif prop.get("type") == "number":
                        mock_args[key] = random.randint(1, 100)
                    elif prop.get("type") == "boolean":
                        mock_args[key] = random.choice([True, False])
            return {
                "content": "",
                "tool_calls": [{"name": func["name"], "arguments": mock_args}],
                "latency": latency,
                "input_tokens": random.randint(50, 300),
                "output_tokens": random.randint(10, 50),
            }

        # Mock text response
        mock_answers = [
            "The answer is (A)",
            "The answer is (B)",
            "The answer is (C)",
            "The answer is (D)",
            "42",
            "def solution():\n    return sum(range(10))",
            "Based on the context, the answer is correct.",
        ]
        return {
            "content": random.choice(mock_answers),
            "tool_calls": None,
            "latency": latency,
            "input_tokens": random.randint(100, 500),
            "output_tokens": random.randint(20, 200),
        }
