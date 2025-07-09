from fastapi.testclient import TestClient
from src.segmentation.fast_api import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Food Segmentation API!"}
    
def test_segment_image():  
    with open("src/tests/integration_tests/images.jpeg", "rb") as image_file:
        response = client.post("/segment", files={"file": image_file})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert len(response.content) > 0
 