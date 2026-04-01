from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, func
from ..database import Base


class ModelEndpoint(Base):
    __tablename__ = "model_endpoints"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), nullable=False)
    provider = Column(String(50), nullable=False, default="openai")  # openai, vllm, litellm
    base_url = Column(String(500), nullable=False)
    api_key = Column(String(500), nullable=True)
    model_id = Column(String(200), nullable=False)
    cost_per_1k_input = Column(Float, nullable=False, default=0.0)
    cost_per_1k_output = Column(Float, nullable=False, default=0.0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
