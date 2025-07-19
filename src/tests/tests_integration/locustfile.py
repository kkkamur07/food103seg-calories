from locust import HttpUser, task, between
import os


class FoodSegmentationUser(HttpUser):
    """Locust user for testing the Food Segmentation API"""

    wait_time = between(2, 4)

    @task(1)
    def read_root(self):
        response = self.client.get("/")
        if response.status_code != 200:
            print(f"Root endpoint failed: {response.status_code}")

    @task(3)
    def segment_image(self):
        path = "src/tests/tests_integration/burger.jpg"
        if not os.path.exists(path):
            print(f"Image file not found: {path}")
            return

        with open(path, "rb") as image_file:
            response = self.client.post("/segment", files={"file": image_file})
            if response.status_code != 200:
                print(f"Food Segmentation failed: {response.status_code}")
