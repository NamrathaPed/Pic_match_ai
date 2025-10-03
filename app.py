import streamlit as st
from PIL import Image
import faiss
import json
import numpy as np
from model import get_image_embedding
import os

EMBEDDINGS_DIR = "embeddings"

# Load FAISS index and image paths
index = faiss.read_index(os.path.join(EMBEDDINGS_DIR, "index.faiss"))
with open(os.path.join(EMBEDDINGS_DIR, "image_paths.json")) as f:
    image_data = json.load(f)

st.title("Pic Match AI 🚀")
st.write("Upload an image to find similar items and identify it!")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Compute embedding
    import tempfile
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    image.save(temp_file.name)
    query_embedding = get_image_embedding(temp_file.name)

    # Search FAISS
    k = 5  # number of similar images
    D, I = index.search(query_embedding, k)

    st.write("### Top Similar Images")
    for i in range(k):
        matched = image_data[I[0][i]]
        matched_img = Image.open(matched["path"])
        st.image(matched_img, caption=f"Label: {matched['label']} (Score: {D[0][i]:.2f})", use_column_width=True)
