import streamlit as st
from PIL import Image as PILImage
import io
import requests
import time
import os

# Configuration
st.set_page_config(
    page_title="Food Segmentation App",
    page_icon="🍽️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    div.stButton > button {
        height: 4.95em;
        margin-top: 0.7em;
        font-size: 1em;
        width: 100%;
    }
</style>
""",
    unsafe_allow_html=True,
)

# FastAPI service configuration
import os

# For cloud compatibility.
FASTAPI_SERVICE_URL = os.getenv("FASTAPI_SERVICE_URL", "http://localhost:8080")
SEGMENT_ENDPOINT = f"{FASTAPI_SERVICE_URL}/segment"
HEALTH_ENDPOINT = f"{FASTAPI_SERVICE_URL}/healthz"


@st.cache_data(show_spinner="Connecting to FastAPI service...")
def check_fastapi_connection():
    """Check if FastAPI service is running"""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=10)
        if response.status_code == 200:
            return True, response.json()
        else:
            st.error(
                f"FastAPI service connection failed: Status {response.status_code}"
            )
            return False, None
    except requests.exceptions.ConnectionError:
        st.error(f"Could not connect to FastAPI service at {FASTAPI_SERVICE_URL}")
        return False, None
    except requests.exceptions.Timeout:
        st.error("FastAPI service connection timed out")
        return False, None
    except Exception as e:
        st.error(f"Connection error: {e}")
        return False, None


def call_fastapi_segmentation(image_bytes: bytes, image_format: str):
    """Send image to FastAPI for segmentation"""
    try:
        files = {
            "file": (f"image.{image_format}", image_bytes, f"image/{image_format}")
        }

        response = requests.post(SEGMENT_ENDPOINT, files=files, timeout=120)

        if response.status_code == 200:
            return PILImage.open(io.BytesIO(response.content))
        else:
            st.error(f"Segmentation failed: Status {response.status_code}")
            try:
                st.json(response.json())
            except:
                st.write(f"Response: {response.text}")
            return None

    except requests.exceptions.ConnectionError:
        st.error("Could not connect to FastAPI service")
        return None
    except requests.exceptions.Timeout:
        st.error("Request timed out")
        return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None


# Main app
st.title("🍽️ Food Segmentation")
st.markdown(
    "Upload an image of food to segment different food items using FastAPI + PyTorch"
)

# Check FastAPI connection
is_connected, health_info = check_fastapi_connection()

if is_connected:
    st.success("✅ Connected to FastAPI service!")
    if health_info:
        st.info(
            f"Model loaded: {health_info.get('model_loaded', 'Unknown')} | Device: {health_info.get('device', 'Unknown')}"
        )

    st.header("Upload Your Food Image")

    col_uploader, col_button = st.columns([0.7, 0.3])

    with col_uploader:
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"],
            help="Upload JPG, JPEG, or PNG image",
        )

    if uploaded_file is not None:
        with col_button:
            st.write("")
            segment_button = st.button("Segment Food Items", use_container_width=True)

        original_image = PILImage.open(uploaded_file)
        image_bytes = uploaded_file.getvalue()
        image_format = uploaded_file.type.split("/")[-1]

        st.divider()

        if segment_button:
            st.subheader("Segmentation Results")

            with st.spinner("Processing image via FastAPI..."):
                segmented_image = call_fastapi_segmentation(image_bytes, image_format)

            if segmented_image:
                segmented_image=segmented_image.resize(original_image.size)
                st.success("✅ Segmentation complete!")

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Original Image")
                    st.image(
                        original_image,
                        caption="Uploaded Image",
                        use_container_width=True,
                    )

                with col2:
                    st.subheader("Segmented Image")
                    st.image(
                        segmented_image,
                        caption="Segmented Output",
                        use_container_width=True,
                    )
            else:
                st.error("❌ Segmentation failed")
    else:
        st.info("Please upload an image to get started")
else:
    st.error(
        "❌ Cannot connect to FastAPI service. Please ensure it's running on port 3000"
    )
