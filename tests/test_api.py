import os
os.environ["BANKAI_API_KEY"] = "test-key"
os.environ["BANKAI_DATABASE_URL"] = "sqlite:///./test_bankai.db"

from fastapi.testclient import TestClient
from app.main import app


def test_health_is_public() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_prediction_requires_api_key() -> None:
    with TestClient(app) as client:
        response = client.post("/api/v1/predictions/sms", json={"message": "hello"})
    assert response.status_code == 401


def test_sms_request_validates_content() -> None:
    with TestClient(app) as client:
        response = client.post("/api/v1/predictions/sms", headers={"X-API-Key": "test-key"}, json={"message": ""})
    assert response.status_code == 422

