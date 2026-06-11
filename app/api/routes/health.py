"""Health check route.

A lightweight liveness probe with no external dependencies, suitable for
container orchestrators and uptime checks.
"""

from fastapi import APIRouter

from app import __version__
from app.schemas.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Report service liveness."""
    return HealthResponse(status="ok", version=__version__)
