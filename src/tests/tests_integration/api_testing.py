from fastapi.testclient import TestClient
from src.app.service import app

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Food Segmentation API is running"}


def test_segment_image():
    with TestClient(app) as client:
        with open("src/tests/tests_integration/burger.jpg", "rb") as image_file:
            response = client.post("/segment", files={"file": image_file})
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert len(response.content) > 0
