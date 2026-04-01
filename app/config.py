from pydantic_settings import BaseSettings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    APP_NAME: str = "Coredge LLM Benchmark"
    DATABASE_URL: str = f"sqlite+aiosqlite:///{BASE_DIR / 'benchmark.db'}"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"

    # Default eval settings
    DEFAULT_CONCURRENT_USERS: int = 10
    DEFAULT_PERF_DURATION_SEC: int = 30
    MOCK_MODE: bool = False  # Use simulated responses when no real endpoint

    # Benchmark dataset settings (core STEM subjects for faster evaluation)
    MMLU_SUBJECTS: list[str] = [
        "abstract_algebra", "anatomy", "astronomy", "business_ethics",
        "college_chemistry", "college_computer_science", "college_mathematics",
        "college_physics", "computer_security", "conceptual_physics",
        "electrical_engineering", "elementary_mathematics", "formal_logic",
        "machine_learning"
    ]
    MMLU_SAMPLES_PER_SUBJECT: int = 50
    GSM8K_SAMPLES: int = 100
    HUMANEVAL_SAMPLES: int = 50
    HUMANEVAL_TIMEOUT: int = 10  # seconds for code execution

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
