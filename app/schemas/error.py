"""Error envelope returned for failed requests."""

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Consistent error body for non-2xx responses."""

    error_code: str = Field(description="Stable machine-readable error identifier.")
    detail: str = Field(description="Human-readable description of the failure.")
