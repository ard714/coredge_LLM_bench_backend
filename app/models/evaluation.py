from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey, func
from ..database import Base


class Evaluation(Base):
    __tablename__ = "evaluations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_endpoint_id = Column(Integer, ForeignKey("model_endpoints.id"), nullable=False)
    status = Column(String(20), default="pending")  # pending, running, completed, failed
    modules = Column(JSON, nullable=False)  # ["capability","tool_call","quality","performance"]
    progress = Column(Integer, default=0)  # 0-100
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    error = Column(String(2000), nullable=True)
    created_at = Column(DateTime, server_default=func.now())
