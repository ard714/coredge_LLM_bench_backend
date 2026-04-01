"""API routes for model endpoint CRUD."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from ..database import get_db
from ..models.model_endpoint import ModelEndpoint

router = APIRouter(prefix="/api/endpoints", tags=["endpoints"])


class EndpointCreate(BaseModel):
    name: str
    provider: str = "openai"
    base_url: str
    api_key: str | None = None
    model_id: str
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0


class EndpointUpdate(BaseModel):
    name: str | None = None
    provider: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    model_id: str | None = None
    cost_per_1k_input: float | None = None
    cost_per_1k_output: float | None = None
    is_active: bool | None = None


@router.get("")
async def list_endpoints(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(ModelEndpoint).order_by(ModelEndpoint.created_at.desc()))
    endpoints = result.scalars().all()
    return [
        {
            "id": e.id,
            "name": e.name,
            "provider": e.provider,
            "base_url": e.base_url,
            "model_id": e.model_id,
            "cost_per_1k_input": e.cost_per_1k_input,
            "cost_per_1k_output": e.cost_per_1k_output,
            "is_active": e.is_active,
            "created_at": str(e.created_at) if e.created_at else None,
        }
        for e in endpoints
    ]


@router.post("", status_code=201)
async def create_endpoint(data: EndpointCreate, db: AsyncSession = Depends(get_db)):
    endpoint = ModelEndpoint(
        name=data.name,
        provider=data.provider,
        base_url=data.base_url,
        api_key=data.api_key,
        model_id=data.model_id,
        cost_per_1k_input=data.cost_per_1k_input,
        cost_per_1k_output=data.cost_per_1k_output,
    )
    db.add(endpoint)
    await db.commit()
    await db.refresh(endpoint)
    return {"id": endpoint.id, "name": endpoint.name}


@router.get("/{endpoint_id}")
async def get_endpoint(endpoint_id: int, db: AsyncSession = Depends(get_db)):
    endpoint = await db.get(ModelEndpoint, endpoint_id)
    if not endpoint:
        raise HTTPException(status_code=404, detail="Endpoint not found")
    return {
        "id": endpoint.id,
        "name": endpoint.name,
        "provider": endpoint.provider,
        "base_url": endpoint.base_url,
        "model_id": endpoint.model_id,
        "cost_per_1k_input": endpoint.cost_per_1k_input,
        "cost_per_1k_output": endpoint.cost_per_1k_output,
        "is_active": endpoint.is_active,
        "created_at": str(endpoint.created_at) if endpoint.created_at else None,
    }


@router.put("/{endpoint_id}")
async def update_endpoint(endpoint_id: int, data: EndpointUpdate, db: AsyncSession = Depends(get_db)):
    endpoint = await db.get(ModelEndpoint, endpoint_id)
    if not endpoint:
        raise HTTPException(status_code=404, detail="Endpoint not found")
    for field, value in data.model_dump(exclude_unset=True).items():
        setattr(endpoint, field, value)
    await db.commit()
    return {"id": endpoint.id, "name": endpoint.name}


@router.delete("/{endpoint_id}")
async def delete_endpoint(endpoint_id: int, db: AsyncSession = Depends(get_db)):
    endpoint = await db.get(ModelEndpoint, endpoint_id)
    if not endpoint:
        raise HTTPException(status_code=404, detail="Endpoint not found")
    await db.delete(endpoint)
    await db.commit()
    return {"deleted": True}


@router.post("/{endpoint_id}/test")
async def test_endpoint(endpoint_id: int, db: AsyncSession = Depends(get_db)):
    """Test if an endpoint is reachable and credentials work."""
    endpoint = await db.get(ModelEndpoint, endpoint_id)
    if not endpoint:
        raise HTTPException(status_code=404, detail="Endpoint not found")

    from ..services.llm_client import LLMClient

    client = LLMClient(
        base_url=endpoint.base_url,
        api_key=endpoint.api_key,
        model_id=endpoint.model_id,
    )

    try:
        response = await client.chat(
            [{"role": "user", "content": "Say 'ok'"}],
            max_tokens=10,
        )
        return {
            "success": True,
            "message": "Connection successful",
            "model": endpoint.model_id,
            "latency_ms": round(response.get("latency", 0) * 1000, 2),
        }
    except Exception as e:
        return {
            "success": False,
            "message": str(e)[:200],
            "model": endpoint.model_id,
        }
