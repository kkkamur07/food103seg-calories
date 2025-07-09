# streamlit_bentoml_frontend.py
import streamlit as st
from PIL import Image as PILImage 
import io
import requests
import time

# --- Configuration ---
st.set_page_config(
    page_title="Food Segmentation (BentoML Backend)",
    page_icon="🍽️",
    layout="centered",
    initial_sidebar_state="expanded"
)
#CSS for Button size management

st.markdown("""
<style>
    /* Adjust button height to roughly match file uploader's browse button */
    div.stButton > button {
        height: 4.95em; 
        margin-top: 0.7em; 
        font-size: 1em; 
        width: 100%; 
    }
    
</style>
""", unsafe_allow_html=True)

# Default BentoML service address and port
BENTOML_SERVICE_URL = "http://localhost:3000/"  #! Beware of this URL -> Make or break things. 
#As the bentoml service is defined as "segment" in service.py file
BENTOML_SEGMENT_ENDPOINT = BENTOML_SERVICE_URL + "segment" #! Make sure the naming is correct. 

#To check if the BentoMl service is running  

@st.cache_data(show_spinner="Connecting to BentoML service...")
def check_bentoml_connection():
    try:
        # BentoML services has /healthz endpoint
        response = requests.get(BENTOML_SERVICE_URL + "healthz", timeout=10) 
        if response.status_code == 200:
            return True
        else:
            st.error(f"BentoML service connection failed: Status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        st.error(
            f"Could not connect to the BentoML service at {BENTOML_SERVICE_URL}. "
            "Please ensure your BentoML service is running."
        )
        return False
    except requests.exceptions.Timeout:
        st.error("BentoML service connection timed out. The backend might be slow to respond.")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred during connection check: {e}")
        return False


def call_bentoml_segmentation(image_bytes: bytes, image_format: str) -> PILImage.Image | None:
    #Sends image for segmentation only if the connection to service was successful
    try:
        files = {
            'image': (f"uploaded_image.{image_format}", image_bytes, f'image/{image_format}')
        }

        #Sends image to BentoMl for segmentation
        response = requests.post(BENTOML_SEGMENT_ENDPOINT, files=files, timeout=120)
        
        if response.status_code == 200:
            segmented_image_bytes = response.content
            return PILImage.open(io.BytesIO(segmented_image_bytes))

        else:
            st.error(f"Failed to get segmentation from BentoML. Status Code: {response.status_code}")
            try:
                # If json error is returned by BentoML
                st.json(response.json()) 
            except requests.exceptions.JSONDecodeError:
                # Otherwise,it displays the raw text content
                st.write(f"Raw BentoML response content: {response.text}") 
            return None
        
    #If connection to BentoML service was not sucessful
    except requests.exceptions.ConnectionError:
        st.error(
            f"Could not connect to the BentoML service at {BENTOML_SEGMENT_ENDPOINT}. "
            "Please ensure your BentoML service is running."
        )
        return None
    
    except requests.exceptions.Timeout:

        st.error("BentoML service request timed out. The model might be taking too long to process the image.")
        return None
    
    except Exception as e:
        st.error(f"An unexpected error occurred during BentoML API call: {e}")
        return None


#Layout of the StreamLit app

st.title("🍽️ Food Segmentation")
st.markdown(
    """
    Upload an image of food,our model will attempt to segment different food items within it.
    """
)

# Check BentoML service connection
bentoml_is_connected = check_bentoml_connection()

if bentoml_is_connected:
    st.success("Connected to BentoML service!")
    st.header("Upload Your Food Image")
    #Columns to get upload image and segment button side by side
    col_uploader, col_button = st.columns([0.7, 0.3])

    with col_uploader:
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"],
            help="Upload an image file (JPG, JPEG, PNG) to perform food segmentation."
        )

    if uploaded_file is not None:
        with col_button:
            st.write("")
            segment_button=st.button("Segment Food Items",use_container_width=True)#button is filled to the column
        
        original_image = PILImage.open(uploaded_file)
        image_bytes = uploaded_file.getvalue()
        
        # Determine image format for Content-Type header
        image_format = uploaded_file.type.split('/')[-1] # e.g., 'png', 'jpeg'

        st.divider()
        if segment_button:
            st.subheader("Segmentation Results")
            
            with st.spinner("Processing image and performing segmentation via BentoML..."):
                #Send the uploadedimage to bentoml for segmentation
                segmented_image = call_bentoml_segmentation(image_bytes, image_format)

            if segmented_image:
                segmented_image=segmented_image.resize(original_image.size)
                st.success("Segmentation complete!")
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Original Image")
                    st.image(original_image, caption="Uploaded Image",use_container_width=True)

                with col2:
                    st.subheader("Segmented Image")
                    st.image(segmented_image, caption="Segmented Output",use_container_width=True)
                    
            else:
                st.warning("Could not display segmented image due to an error during processing by BentoML.")

    else:
        st.info("Please upload an image to get started.")
else:
    st.warning("Cannot proceed. Please ensure the BentoML service is running and accessible.")

