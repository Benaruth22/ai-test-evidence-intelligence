
import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import predict_failure
from src.database import save_prediction, get_monitoring_summary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Test Evidence Intelligence API",
    description="Classify failed software test executions and store predictions.",
    version="1.3.0",
)


class TestEvidence(BaseModel):
    test_name: str = Field(..., min_length=1, max_length=255)
    execution_time: float = Field(..., ge=0)
    status: str
    error_message: str = ""
    environment: str = "qa"
    retry_count: int = Field(default=0, ge=0)
    timestamp: Optional[str] = None


@app.get("/")
def home():
    return {
        "message": "AI Test Evidence Intelligence API",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/monitoring/summary")
def monitoring_summary():
    try:
        return get_monitoring_summary()
    except Exception as error:
        logger.exception("Failed to retrieve monitoring summary")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve monitoring summary.",
        ) from error


@app.post("/predict")
def predict(evidence: TestEvidence):
    status = evidence.status.strip().upper()

    if status not in {"PASS", "FAIL"}:
        raise HTTPException(
            status_code=422,
            detail="status must be PASS or FAIL",
        )

    if status == "PASS":
        raise HTTPException(
            status_code=422,
            detail="Root-cause prediction is available only for failed test executions.",
        )

    data = evidence.model_dump()
    data["status"] = status
    data["environment"] = evidence.environment.strip().lower()
    data["timestamp"] = (
        evidence.timestamp
        or datetime.now().isoformat(timespec="seconds")
    )

    try:
        # The predictor expects one dictionary argument.
        result = predict_failure(data)

        category = (
            result.get("predicted_category")
            or result.get("predicted_failure_category")
        )
        confidence = result.get("confidence")

        if not category:
            raise ValueError("Prediction did not return a category.")

        prediction_for_db = {
            "predicted_failure_category": category,
            "confidence": confidence,
        }

        record_id = save_prediction(data, prediction_for_db)

        return {
            **result,
            "predicted_failure_category": category,
            "confidence": confidence,
            "database_record_id": record_id,
            "saved_to_database": True,
        }

    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    except Exception as error:
        logger.exception("Prediction or database save failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction or database save failed.",
        ) from error
