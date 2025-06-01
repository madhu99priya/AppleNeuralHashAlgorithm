import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from numpy.linalg import norm
from PIL import Image

# Load the feature extractor model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mobilenetv2_feature_extractor.h5")

model = load_model()

# Preprocess image for model
def preprocess(img: Image.Image):
    img = img.resize((224, 224)).convert("RGB")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Get feature vector
def get_feature_vector(img: Image.Image):
    preprocessed = preprocess(img)
    feature_vector = model.predict(preprocessed)[0]  # shape: (1280,)
    return feature_vector

# Cosine similarity
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

# Streamlit UI
st.title("Face Similarity Checker")
st.markdown("Upload two face images to compare their similarity using MobileNetV2.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Image 1")
    uploaded_img1 = st.file_uploader("Upload first image", type=["jpg", "jpeg", "png"], key="img1")

with col2:
    st.subheader("Image 2")
    uploaded_img2 = st.file_uploader("Upload second image", type=["jpg", "jpeg", "png"], key="img2")

if uploaded_img1 and uploaded_img2:
    # Load images
    img1 = Image.open(uploaded_img1)
    img2 = Image.open(uploaded_img2)

    # Show images
    col1.image(img1, caption="Image 1",  use_container_width=True)
    col2.image(img2, caption="Image 2",  use_container_width=True)

    # Compute feature vectors and similarity
    vec1 = get_feature_vector(img1)
    vec2 = get_feature_vector(img2)
    similarity = cosine_similarity(vec1, vec2)

    # Display result
    st.markdown("---")
    st.subheader("Cosine Similarity Score:")
    st.success(f"{similarity:.4f}")

    # Optional interpretation
    if similarity > 0.85:
        st.info("✅ Likely the same or very similar person.")
    elif similarity > 0.6:
        st.warning("⚠️ Possibly similar, but not certain.")
    else:
        st.error("❌ Likely different individuals.")
