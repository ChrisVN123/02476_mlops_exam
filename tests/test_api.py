from fastapi.testclient import TestClient

from src.exam_project.api import app

client = TestClient(app)


def test_valid_initials():
    response = client.get("/predict/AAPL")
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "correct" in response.json()


def test_invalid_initials():
    response = client.get("/predict/INVALID")
    assert response.status_code == 400
    json_response = response.json()
    assert json_response["error"] == "Invalid initials provided."
    assert "available_initials" in json_response


def test_error():
    response = client.get("/predict/AAPL?model_path=nonexistent_model.onnx")
    assert response.status_code == 500
    assert response.json() == {"detail": "An unexpected error occurred."}
