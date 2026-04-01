"""
Capability testing module — runs standard benchmarks (MMLU, GSM8K, HumanEval)
against an LLM endpoint and scores correctness.

Uses Hugging Face datasets for real benchmark data with fallback to sample data.
"""
import re
import asyncio
from ..config import settings
from ..services.llm_client import LLMClient
from .hf_datasets import (
    load_mmlu_questions,
    load_gsm8k_questions,
    load_humaneval_problems,
)
from .code_executor import run_humaneval_test

# ---------- fallback sample datasets (used if HF datasets fail) ----------

MMLU_FALLBACK = [
    {
        "question": "What is the capital of France?",
        "choices": ["(A) London", "(B) Paris", "(C) Berlin", "(D) Madrid"],
        "answer": "B",
    },
    {
        "question": "Which planet is known as the Red Planet?",
        "choices": ["(A) Venus", "(B) Jupiter", "(C) Mars", "(D) Saturn"],
        "answer": "C",
    },
    {
        "question": "What is the speed of light in vacuum approximately?",
        "choices": ["(A) 3×10^6 m/s", "(B) 3×10^8 m/s", "(C) 3×10^10 m/s", "(D) 3×10^4 m/s"],
        "answer": "B",
    },
    {
        "question": "Who wrote 'Romeo and Juliet'?",
        "choices": ["(A) Charles Dickens", "(B) Mark Twain", "(C) William Shakespeare", "(D) Jane Austen"],
        "answer": "C",
    },
    {
        "question": "What is the chemical symbol for gold?",
        "choices": ["(A) Ag", "(B) Fe", "(C) Au", "(D) Cu"],
        "answer": "C",
    },
]

GSM8K_FALLBACK = [
    {"question": "If a train travels 60 km/h for 2.5 hours, how far does it go?", "answer": "150"},
    {"question": "A store sells apples for $2 each. If you buy 7 apples, how much do you pay?", "answer": "14"},
    {"question": "If 3x + 5 = 20, what is x?", "answer": "5"},
    {"question": "A rectangle has length 8 and width 5. What is its area?", "answer": "40"},
    {"question": "If you have 48 cookies and share equally among 6 friends, how many does each get?", "answer": "8"},
]

HUMANEVAL_FALLBACK = [
    {
        "prompt": "Write a Python function that returns the sum of a list of numbers.\ndef sum_list(nums):",
        "test_code": "assert sum_list([1,2,3]) == 6",
        "entry_point": "sum_list",
    },
    {
        "prompt": "Write a Python function that returns the factorial of n.\ndef factorial(n):",
        "test_code": "assert factorial(5) == 120",
        "entry_point": "factorial",
    },
]


# ---------- answer extraction utilities ----------

def _extract_answer_letter(text: str) -> str | None:
    """Extract the answer letter from model output like 'The answer is (B)'."""
    patterns = [
        r"\(([A-D])\)",
        r"answer is ([A-D])",
        r"^([A-D])[\.\)\s]",
        r"([A-D])$",
    ]
    for pat in patterns:
        m = re.search(pat, text.strip(), re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return None


def _extract_number(text: str) -> str | None:
    """Extract a numeric answer from model output."""
    numbers = re.findall(r"[-+]?\d*\.?\d+", text.strip())
    if numbers:
        # Return the last number found (models often explain first, give answer last)
        n = numbers[-1]
        if "." in n:
            return str(int(float(n))) if float(n) == int(float(n)) else n
        return n
    return None


# ---------- benchmark runners ----------

async def run_mmlu(client: LLMClient, use_hf: bool = True) -> dict:
    """Run MMLU-style multiple-choice benchmark."""
    # Load questions
    questions = []
    if use_hf:
        try:
            questions = load_mmlu_questions(
                subjects=settings.MMLU_SUBJECTS,
                samples_per_subject=settings.MMLU_SAMPLES_PER_SUBJECT,
            )
        except Exception as e:
            print(f"Warning: Failed to load HF MMLU dataset: {e}, using fallback")
            questions = []

    if not questions:
        questions = MMLU_FALLBACK

    correct = 0
    total = len(questions)
    details = []

    for q in questions:
        prompt = f"{q['question']}\n" + "\n".join(f"({chr(65+i)}) {c}" for i, c in enumerate(q["choices"]))
        prompt += "\n\nAnswer with the letter only."
        response = await client.chat([{"role": "user", "content": prompt}])
        predicted = _extract_answer_letter(response["content"])
        is_correct = predicted == q["answer"]
        if is_correct:
            correct += 1
        details.append({
            "question": q["question"][:200],
            "expected": q["answer"],
            "predicted": predicted,
            "correct": is_correct,
            "raw_output": response["content"][:200],
        })

    return {
        "benchmark_name": "mmlu",
        "score": correct / total if total > 0 else 0,
        "total": total,
        "correct": correct,
        "details": details,
    }


async def run_gsm8k(client: LLMClient, use_hf: bool = True) -> dict:
    """Run GSM8K math reasoning benchmark."""
    # Load questions
    questions = []
    if use_hf:
        try:
            questions = load_gsm8k_questions(samples=settings.GSM8K_SAMPLES)
        except Exception as e:
            print(f"Warning: Failed to load HF GSM8K dataset: {e}, using fallback")
            questions = []

    if not questions:
        questions = GSM8K_FALLBACK

    correct = 0
    total = len(questions)
    details = []

    for q in questions:
        prompt = f"Solve this math problem and give just the numeric answer:\n{q['question']}"
        response = await client.chat([{"role": "user", "content": prompt}])
        predicted = _extract_number(response["content"])
        is_correct = predicted == q["answer"]
        if is_correct:
            correct += 1
        details.append({
            "question": q["question"][:200],
            "expected": q["answer"],
            "predicted": predicted,
            "correct": is_correct,
            "raw_output": response["content"][:200],
        })

    return {
        "benchmark_name": "gsm8k",
        "score": correct / total if total > 0 else 0,
        "total": total,
        "correct": correct,
        "details": details,
    }


async def run_humaneval(client: LLMClient, use_hf: bool = True) -> dict:
    """Run HumanEval code generation benchmark with sandboxed execution."""
    # Load problems
    problems = []
    if use_hf:
        try:
            problems = load_humaneval_problems(samples=settings.HUMANEVAL_SAMPLES)
        except Exception as e:
            print(f"Warning: Failed to load HF HumanEval dataset: {e}, using fallback")
            problems = []

    if not problems:
        problems = HUMANEVAL_FALLBACK

    correct = 0
    total = len(problems)
    details = []

    for p in problems:
        prompt = f"Complete this Python function. Return ONLY the code:\n\n{p['prompt']}"
        response = await client.chat([{"role": "user", "content": prompt}], max_tokens=512)

        # Run code against test cases
        result = await run_humaneval_test(
            prompt=p["prompt"],
            response=response["content"],
            test_code=p["test_code"],
            entry_point=p["entry_point"],
            timeout=settings.HUMANEVAL_TIMEOUT,
        )

        is_correct = result["passed"]
        if is_correct:
            correct += 1

        details.append({
            "entry_point": p["entry_point"],
            "correct": is_correct,
            "error": result.get("error"),
            "raw_output": response["content"][:300],
        })

    return {
        "benchmark_name": "humaneval",
        "score": correct / total if total > 0 else 0,
        "total": total,
        "correct": correct,
        "details": details,
    }


async def run_capability_eval(client: LLMClient, use_hf: bool = True) -> list[dict]:
    """Run all capability benchmarks and return list of results."""
    results = []
    results.append(await run_mmlu(client, use_hf=use_hf))
    results.append(await run_gsm8k(client, use_hf=use_hf))
    results.append(await run_humaneval(client, use_hf=use_hf))
    return results