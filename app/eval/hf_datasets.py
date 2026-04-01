"""
Hugging Face dataset loaders for standard benchmarks.
Caches datasets in memory to avoid re-downloading.
"""
from datasets import load_dataset
from typing import Generator

# In-memory cache for loaded datasets
_DATASET_CACHE: dict = {}


def _get_cache_key(name: str, subset: str | None = None) -> str:
    """Generate cache key for dataset."""
    return f"{name}:{subset}" if subset else name


def load_mmlu_questions(
    subjects: list[str],
    samples_per_subject: int = 50,
) -> list[dict]:
    """
    Load MMLU multiple-choice questions from Hugging Face.

    Args:
        subjects: List of MMLU subjects to include
        samples_per_subject: Max questions per subject

    Returns:
        List of questions with format:
        {"question": str, "choices": list[str], "answer": str}
    """
    all_questions = []

    for subject in subjects:
        cache_key = _get_cache_key("cais/mmlu", subject)

        if cache_key not in _DATASET_CACHE:
            try:
                _DATASET_CACHE[cache_key] = load_dataset(
                    "cais/mmlu", subject, split="test", trust_remote_code=True
                )
            except Exception as e:
                print(f"Warning: Failed to load MMLU subject '{subject}': {e}")
                continue

        dataset = _DATASET_CACHE[cache_key]
        samples = list(dataset.select(range(min(samples_per_subject, len(dataset)))))

        for sample in samples:
            all_questions.append({
                "question": sample["question"],
                "choices": [sample["choices"][i] for i in range(4)],
                "answer": ["A", "B", "C", "D"][sample["answer"]],
            })

    return all_questions


def load_gsm8k_questions(samples: int = 100) -> list[dict]:
    """
    Load GSM8K math problems from Hugging Face.

    Args:
        samples: Max number of questions to load

    Returns:
        List of questions with format:
        {"question": str, "answer": str}
    """
    cache_key = _get_cache_key("gsm8k", "main")

    if cache_key not in _DATASET_CACHE:
        try:
            _DATASET_CACHE[cache_key] = load_dataset("gsm8k", "main", split="test")
        except Exception as e:
            print(f"Warning: Failed to load GSM8K: {e}")
            return []

    dataset = _DATASET_CACHE[cache_key]
    samples_list = list(dataset.select(range(min(samples, len(dataset)))))

    questions = []
    for sample in samples_list:
        # GSM8K answer format: "#### <number>" at the end
        answer_text = sample["answer"]
        answer = answer_text.split("#### ")[-1].strip().replace(",", "")
        questions.append({
            "question": sample["question"],
            "answer": answer,
        })

    return questions


def load_humaneval_problems(samples: int = 50) -> list[dict]:
    """
    Load HumanEval code generation problems from Hugging Face.

    Args:
        samples: Max number of problems to load

    Returns:
        List of problems with format:
        {"prompt": str, "test_code": str, "entry_point": str}
    """
    cache_key = _get_cache_key("openai_humaneval")

    if cache_key not in _DATASET_CACHE:
        try:
            _DATASET_CACHE[cache_key] = load_dataset("openai_humaneval", split="test")
        except Exception as e:
            print(f"Warning: Failed to load HumanEval: {e}")
            return []

    dataset = _DATASET_CACHE[cache_key]
    samples_list = list(dataset.select(range(min(samples, len(dataset)))))

    problems = []
    for sample in samples_list:
        problems.append({
            "prompt": sample["prompt"],
            "test_code": sample["test"],
            "entry_point": sample["entry_point"],
        })

    return problems


def clear_cache():
    """Clear the dataset cache."""
    global _DATASET_CACHE
    _DATASET_CACHE = {}