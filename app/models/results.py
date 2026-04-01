from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey, func
from ..database import Base


class BenchmarkResult(Base):
    __tablename__ = "benchmark_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    evaluation_id = Column(Integer, ForeignKey("evaluations.id"), nullable=False)
    benchmark_name = Column(String(100), nullable=False)  # mmlu, gsm8k, humaneval
    score = Column(Float, nullable=False)
    total = Column(Integer, nullable=False)
    correct = Column(Integer, nullable=False)
    details = Column(JSON, nullable=True)
    created_at = Column(DateTime, server_default=func.now())


class ToolCallResult(Base):
    __tablename__ = "tool_call_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    evaluation_id = Column(Integer, ForeignKey("evaluations.id"), nullable=False)
    accuracy = Column(Float, nullable=False)
    precision = Column(Float, nullable=False)
    recall = Column(Float, nullable=False)
    false_positive_rate = Column(Float, nullable=False)
    total_tests = Column(Integer, nullable=False)
    details = Column(JSON, nullable=True)
    created_at = Column(DateTime, server_default=func.now())


class QualityResult(Base):
    __tablename__ = "quality_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    evaluation_id = Column(Integer, ForeignKey("evaluations.id"), nullable=False)
    hallucination_rate = Column(Float, nullable=False)
    answer_relevancy = Column(Float, nullable=False)
    faithfulness = Column(Float, nullable=False)
    total_tests = Column(Integer, nullable=False)
    details = Column(JSON, nullable=True)
    created_at = Column(DateTime, server_default=func.now())


class PerformanceResult(Base):
    __tablename__ = "performance_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    evaluation_id = Column(Integer, ForeignKey("evaluations.id"), nullable=False)
    latency_p50 = Column(Float, nullable=False)
    latency_p95 = Column(Float, nullable=False)
    latency_p99 = Column(Float, nullable=False)
    tokens_per_sec = Column(Float, nullable=False)
    cost_per_1k = Column(Float, nullable=False)
    concurrent_users = Column(Integer, nullable=False)
    total_requests = Column(Integer, nullable=False)
    error_rate = Column(Float, nullable=False, default=0.0)
    details = Column(JSON, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
