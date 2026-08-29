"""FastAPI inference app backed by a versioned model bundle."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from src.api.schemas import HealthResponse, MetricsResponse, PredictionResponse
from src.config import load_config
from src.inference.predictor import ModelPredictor


logger = logging.getLogger("cats_dogs_api")
logger.setLevel(logging.INFO)


def create_app(config_path: Path = Path("configs/base.yaml")) -> FastAPI:
    config = load_config(config_path)
    bundle_path = Path(os.getenv("MODEL_BUNDLE_PATH", str(config.runtime.model_bundle_path)))
    predictor = ModelPredictor(config, bundle_path)
    app = FastAPI(title="Cats vs Dogs Inference Service", version="1.0.0")
    app.state.predictor = predictor
    app.state.model_error = None
    app.state.prediction_request_count = 0
    app.state.prediction_error_count = 0
    app.state.total_prediction_latency_ms = 0.0

    @app.on_event("startup")
    def load_model_bundle() -> None:
        if not predictor.bundle_dir.exists():
            return
        try:
            predictor.load()
        except Exception as error:  # noqa: BLE001
            app.state.model_error = str(error)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        if predictor.is_ready():
            metadata = predictor.loaded_bundle.metadata
            return HealthResponse(
                status="ok",
                model_loaded=True,
                model_type=metadata.model_type,
                model_version=metadata.model_version,
            )
        return HealthResponse(status="ok", model_loaded=False)

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(file: UploadFile = File(...)) -> PredictionResponse:
        if not predictor.is_ready():
            raise HTTPException(status_code=503, detail="Model bundle is not loaded.")
        image_bytes = await file.read()
        started = time.perf_counter()
        app.state.prediction_request_count += 1
        try:
            prediction = predictor.predict_bytes(image_bytes)
        except ValueError as error:
            app.state.prediction_error_count += 1
            raise HTTPException(status_code=400, detail=str(error)) from error
        finally:
            latency_ms = (time.perf_counter() - started) * 1000
            app.state.total_prediction_latency_ms += latency_ms
            logger.info(
                "prediction_request filename=%s content_type=%s latency_ms=%.2f",
                file.filename,
                file.content_type,
                latency_ms,
            )
        return PredictionResponse(**prediction.__dict__)

    @app.get("/metrics", response_model=MetricsResponse)
    def metrics() -> MetricsResponse:
        request_count = app.state.prediction_request_count
        average_latency = (
            app.state.total_prediction_latency_ms / request_count if request_count else 0.0
        )
        return MetricsResponse(
            prediction_request_count=request_count,
            prediction_error_count=app.state.prediction_error_count,
            average_prediction_latency_ms=average_latency,
        )

    return app


app = create_app()
