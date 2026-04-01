"""
Quality metrics module — LLM-as-judge scoring for hallucination rate,
answer relevancy, and faithfulness.
"""
import re
import random
from ..services.llm_client import LLMClient
from ..config import settings

# ---------- test cases ----------

QA_PAIRS = [
    {
        "context": "The Eiffel Tower was constructed in 1889 for the World's Fair. It stands 330 meters tall and is located in Paris, France.",
        "question": "When was the Eiffel Tower built?",
        "reference_answer": "The Eiffel Tower was constructed in 1889.",
    },
    {
        "context": "Python was created by Guido van Rossum and first released in 1991. It emphasizes code readability and uses significant indentation.",
        "question": "Who created Python?",
        "reference_answer": "Python was created by Guido van Rossum.",
    },
    {
        "context": "The human heart has four chambers: two atria and two ventricles. It pumps about 5 liters of blood per minute.",
        "question": "How many chambers does the human heart have?",
        "reference_answer": "The human heart has four chambers.",
    },
    {
        "context": "The Great Wall of China is over 13,000 miles long. It was built over many centuries, starting from the 7th century BC.",
        "question": "How long is the Great Wall of China?",
        "reference_answer": "The Great Wall of China is over 13,000 miles long.",
    },
    {
        "context": "Photosynthesis converts carbon dioxide and water into glucose and oxygen, using sunlight as energy. This process occurs in chloroplasts.",
        "question": "What does photosynthesis produce?",
        "reference_answer": "Photosynthesis produces glucose and oxygen.",
    },
    {
        "context": "Albert Einstein published the theory of general relativity in 1915. It describes gravity as a curvature of spacetime.",
        "question": "What does general relativity describe?",
        "reference_answer": "General relativity describes gravity as a curvature of spacetime.",
    },
    {
        "context": "Water boils at 100 degrees Celsius at standard atmospheric pressure. Below 0 degrees Celsius, water freezes into ice.",
        "question": "At what temperature does water boil?",
        "reference_answer": "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    },
    {
        "context": "The Amazon Rainforest covers about 5.5 million square kilometers and produces approximately 20% of the world's oxygen.",
        "question": "How much oxygen does the Amazon Rainforest produce?",
        "reference_answer": "The Amazon Rainforest produces approximately 20% of the world's oxygen.",
    },
]


def _parse_judge_score(text: str) -> float:
    """Extract a numeric score from judge response (expected 1-5 scale, normalized to 0-1)."""
    numbers = re.findall(r"(\d+(?:\.\d+)?)\s*(?:/\s*5|out of 5)?", text)
    if numbers:
        val = float(numbers[0])
        if val > 1:
            return min(val / 5.0, 1.0)
        return val
    return 0.5  # default middle score


async def _judge_or_mock(client: LLMClient, prompt: str) -> float:
    """Use LLM-as-judge or return mock score."""
    if settings.MOCK_MODE:
        return round(random.uniform(0.5, 1.0), 3)

    response = await client.chat([{"role": "user", "content": prompt}], temperature=0.0)
    return _parse_judge_score(response["content"])


async def run_quality_eval(client: LLMClient) -> dict:
    """Run quality metrics evaluation using LLM-as-judge pattern."""
    hallucination_scores = []
    relevancy_scores = []
    faithfulness_scores = []
    details = []

    for qa in QA_PAIRS:
        # 1. Get model answer
        messages = [
            {"role": "system", "content": f"Answer based on this context:\n{qa['context']}"},
            {"role": "user", "content": qa["question"]},
        ]
        response = await client.chat(messages)
        model_answer = response["content"]

        # 2. Judge hallucination (lower is better, we'll invert)
        hallucination_prompt = f"""Rate how much of this answer contains information NOT supported by the context.
Context: {qa['context']}
Question: {qa['question']}
Answer: {model_answer}
Rate from 1 (no hallucination) to 5 (complete hallucination). Just give the number."""

        h_score = await _judge_or_mock(client, hallucination_prompt)

        # 3. Judge relevancy
        relevancy_prompt = f"""Rate how relevant this answer is to the question asked.
Question: {qa['question']}
Answer: {model_answer}
Rate from 1 (not relevant) to 5 (perfectly relevant). Just give the number."""

        r_score = await _judge_or_mock(client, relevancy_prompt)

        # 4. Judge faithfulness
        faithfulness_prompt = f"""Rate how faithful this answer is to the provided context.
Context: {qa['context']}
Answer: {model_answer}
Reference: {qa['reference_answer']}
Rate from 1 (unfaithful) to 5 (completely faithful). Just give the number."""

        f_score = await _judge_or_mock(client, faithfulness_prompt)

        hallucination_scores.append(h_score)
        relevancy_scores.append(r_score)
        faithfulness_scores.append(f_score)

        details.append({
            "question": qa["question"],
            "model_answer": model_answer[:300],
            "reference": qa["reference_answer"],
            "hallucination_score": h_score,
            "relevancy_score": r_score,
            "faithfulness_score": f_score,
        })

    total = len(QA_PAIRS)
    avg_hallucination = sum(hallucination_scores) / total if total > 0 else 0
    avg_relevancy = sum(relevancy_scores) / total if total > 0 else 0
    avg_faithfulness = sum(faithfulness_scores) / total if total > 0 else 0

    return {
        "hallucination_rate": round(avg_hallucination, 4),
        "answer_relevancy": round(avg_relevancy, 4),
        "faithfulness": round(avg_faithfulness, 4),
        "total_tests": total,
        "details": details,
    }
