import os
import pandas as pd
import requests
import streamlit as st
# from google.cloud import run_v2 # Commented out as it's not needed in mock mode

# --- Configuration for Mock Backend ---
# Set this to True to run the frontend with dummy data, without a real backend.
# Set to False when you have a backend running (locally or in the cloud).
MOCK_BACKEND = True

# Define a local backend URL for testing when MOCK_BACKEND is False
# If your backend runs locally, it's typically http://localhost:8000
LOCAL_BACKEND_URL = "http://localhost:8000"
# --- End Configuration ---


def get_backend_url():
    """
    Get the URL of the backend service.
    In mock mode, returns a dummy URL.
    """
    if MOCK_BACKEND:
        return "http://mock-backend-url.com" # Just a placeholder URL
    else:
        # Original logic to get Cloud Run URL
        # from google.cloud import run_v2 # Import here to avoid error if not installed
        # client = run_v2.ServicesClient()
        # parent = "projects/my-personal-mlops-project/locations/europe-west1"
        # services = client.list_services(parent=parent)
        # for service in services:
        #     if service.name.split("/")[-1] == "production-model":
        #         return service.uri
        # return os.environ.get("BACKEND", None) # Fallback to environment variable

        # For local testing, you might return a local URL here
        return LOCAL_BACKEND_URL


def classify_image(image, backend):
    """
    Send the image to the backend for classification.
    Returns dummy data if MOCK_BACKEND is True.
    """
    if MOCK_BACKEND:
        # --- MOCKING BACKEND RESPONSE ---
        st.info("Running in MOCK BACKEND mode. Returning dummy data.")
        # You can add more sophisticated mocking logic here based on image content
        # For example, if "dog" is in the filename, return "dog" prediction
        dummy_prediction = "Dummy Class"
        dummy_probabilities = [0.1, 0.2, 0.7] # Example probabilities (sum to 1)

        # To make it slightly dynamic based on image size or type
        if image:
            try:
                from PIL import Image # Import PIL here to avoid error if not used
                import io
                img_bytes = io.BytesIO(image)
                img = Image.open(img_bytes)
                if img.width > 500:
                    dummy_prediction = "Large Image Class"
                else:
                    dummy_prediction = "Small Image Class"
                dummy_probabilities = [0.5, 0.3, 0.2] # Different dummy probabilities
            except Exception:
                pass # Fallback if image parsing fails

        return {
            "prediction": dummy_prediction,
            "probabilities": dummy_probabilities
        }
    else:
        # --- REAL BACKEND CALL ---
        predict_url = f"{backend}/classify/" # Assuming backend has /classify endpoint
        try:
            # requests.post expects a file-like object or bytes for 'files'
            # For Streamlit's uploaded_file.read(), it's already bytes.
            response = requests.post(predict_url, files={"file": image}, timeout=30) # Increased timeout
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            return response.json()
        except requests.exceptions.ConnectionError:
            st.error(f"Could not connect to the backend at {predict_url}. Is it running?")
            return None
        except requests.exceptions.Timeout:
            st.error(f"Backend request timed out after 30 seconds to {predict_url}.")
            return None
        except requests.exceptions.RequestException as e:
            st.error(f"Error from backend: {e}")
            return None


def main() -> None:
    """Main function of the Streamlit frontend."""
    st.set_page_config(
        page_title="Image Classification Frontend",
        page_icon="🖼️",
        layout="centered", # or "wide"
        initial_sidebar_state="auto"
    )

    st.title("Image Classification")

    # --- Toggle for Mock Backend in Sidebar ---
    global MOCK_BACKEND # Allow changing the global variable
    MOCK_BACKEND = st.sidebar.checkbox("Use Mock Backend (for UI testing)", value=MOCK_BACKEND)
    st.sidebar.markdown("---")
    st.sidebar.info(
        "Toggle 'Use Mock Backend' to switch between dummy data "
        "and attempting to connect to a real backend. "
        "When Mock is OFF, ensure your backend is running locally or deployed."
    )
    # --- End Toggle ---

    backend = get_backend_url()
    if backend is None:
        st.error("Backend service URL not found. Please ensure it's deployed or set up correctly.")
        st.stop() # Stop Streamlit execution if no backend URL

    st.write(f"Backend URL: {backend} (Mode: {'Mock' if MOCK_BACKEND else 'Real'})")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        st.write("") # Add a little space

        # Read the image bytes once
        image_bytes = uploaded_file.read()

        # Classify the image
        with st.spinner("Classifying image..."):
            result = classify_image(image_bytes, backend=backend)

        if result is not None:
            prediction = result["prediction"]
            probabilities = result["probabilities"]

            st.subheader("Classification Result:")
            st.success(f"Prediction: **{prediction}**")

            # Make a nice bar chart for probabilities
            # Ensure probabilities list matches the number of classes for the chart
            num_classes = len(probabilities)
            if num_classes > 0:
                # Create class labels dynamically if they are not part of the result
                class_labels = [f"Class {i}" for i in range(num_classes)]
                # If your backend returns actual class names, use them:
                # class_labels = result.get("class_names", [f"Class {i}" for i in range(num_classes)])

                data = {"Class": class_labels, "Probability": probabilities}
                df = pd.DataFrame(data)
                df.set_index("Class", inplace=True)
                st.bar_chart(df, y="Probability")
            else:
                st.warning("No probabilities returned or probabilities list is empty.")
        else:
            st.error("Failed to get prediction from backend.")
    else:
        st.info("Upload an image to see classification results.")


if __name__ == "__main__":
    main()
