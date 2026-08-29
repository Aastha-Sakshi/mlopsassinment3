"""API response models."""

from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str | None = None
    model_version: str | None = None


class PredictionResponse(BaseModel):
    label: str
    probabilities: dict[str, float]
    model_version: str


class MetricsResponse(BaseModel):
    prediction_request_count: int
    prediction_error_count: int
    average_prediction_latency_ms: float
