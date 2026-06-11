"""Schemas for the health check endpoint."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Liveness response returned by `GET /health`."""

    status: str = Field(description="Service status indicator, 'ok' when live.")
    version: str = Field(description="Running application version.")
