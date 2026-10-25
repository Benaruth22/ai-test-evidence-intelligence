
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_monitoring_summary_endpoint(monkeypatch):
    def mock_get_monitoring_summary():
        return {
            "total_predictions": 2,
            "average_confidence": 0.89,
            "first_prediction": "2026-10-25T16:56:17",
            "latest_prediction": "2026-10-25T17:20:10",
            "categories": [
                {
                    "category": "FLAKY",
                    "count": 2,
                    "average_confidence": 0.89,
                }
            ],
        }

    monkeypatch.setattr(
        "src.api.get_monitoring_summary",
        mock_get_monitoring_summary,
    )

    response = client.get("/monitoring/summary")

    assert response.status_code == 200
    result = response.json()
    assert result["total_predictions"] == 2
    assert result["average_confidence"] == 0.89
    assert result["categories"][0]["category"] == "FLAKY"


def test_monitoring_summary_database_error_returns_server_error(monkeypatch):
    def mock_get_monitoring_summary():
        raise RuntimeError("Database unavailable")

    monkeypatch.setattr(
        "src.api.get_monitoring_summary",
        mock_get_monitoring_summary,
    )

    response = client.get("/monitoring/summary")

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "Failed to retrieve monitoring summary."
    )


def test_successful_prediction(monkeypatch):
    def mock_predict_failure(data):
        return {
            "test_name": data["test_name"],
            "predicted_category": "REAL_BUG",
            "confidence": 0.95,
        }

    def mock_save_prediction(evidence, prediction):
        assert prediction["predicted_failure_category"] == "REAL_BUG"
        return 1234

    monkeypatch.setattr("src.api.predict_failure", mock_predict_failure)
    monkeypatch.setattr("src.api.save_prediction", mock_save_prediction)

    response = client.post(
        "/predict",
        json={
            "test_name": "CheckoutTest",
            "execution_time": 5.2,
            "status": "FAIL",
            "error_message": "AssertionError: total mismatch",
            "environment": "qa",
            "retry_count": 1,
        },
    )

    assert response.status_code == 200
    result = response.json()
    assert result["predicted_failure_category"] == "REAL_BUG"
    assert result["confidence"] == 0.95
    assert result["database_record_id"] == 1234
    assert result["saved_to_database"] is True


def test_invalid_status_is_rejected():
    response = client.post(
        "/predict",
        json={
            "test_name": "CheckoutTest",
            "execution_time": 5.2,
            "status": "UNKNOWN",
        },
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "status must be PASS or FAIL"


def test_negative_execution_time_is_rejected():
    response = client.post(
        "/predict",
        json={
            "test_name": "CheckoutTest",
            "execution_time": -1,
            "status": "FAIL",
        },
    )
    assert response.status_code == 422


def test_pass_execution_is_rejected():
    response = client.post(
        "/predict",
        json={
            "test_name": "CheckoutTest",
            "execution_time": 5.2,
            "status": "PASS",
        },
    )
    assert response.status_code == 422
    assert "only for failed test executions" in response.json()["detail"]


def test_prediction_failure_returns_server_error(monkeypatch):
    def mock_predict_failure(data):
        raise RuntimeError("Model unavailable")

    monkeypatch.setattr("src.api.predict_failure", mock_predict_failure)

    response = client.post(
        "/predict",
        json={
            "test_name": "CheckoutTest",
            "execution_time": 5.2,
            "status": "FAIL",
        },
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Prediction or database save failed."
