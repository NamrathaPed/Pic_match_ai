import streamlit as st
from PIL import Image
import hnswlib
import json
import numpy as np
from model import get_image_embedding
import os
import tempfile

EMBEDDINGS_DIR = "embeddings"
IMAGE_DIR = "data"  # folder where your images are stored

# Ensure image_paths.json exists
image_json_path = os.path.join(EMBEDDINGS_DIR, "image_paths.json")
if not os.path.exists(image_json_path):
    # Build JSON with path + label if missing
    image_data = []
    for root, dirs, files in os.walk(IMAGE_DIR):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                label = os.path.basename(root)  # folder name as label
                image_data.append({"path": os.path.join(root, file), "label": label})
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    with open(image_json_path, "w") as f:
        json.dump(image_data, f, indent=4)
else:
    with open(image_json_path, "r") as f:
        image_data = json.load(f)

# Load Hnswlib index
dim = 512  # update this if your model’s embedding dimension is different
index = hnswlib.Index(space='l2', dim=dim)
index.load_index(os.path.join(EMBEDDINGS_DIR, "index.bin"))

# Streamlit UI
st.title("Pic Match AI 🚀")
st.write("Upload an image to find similar items and identify it!")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Compute embedding
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    image.save(temp_file.name)
    query_embedding = get_image_embedding(temp_file.name)
    query_embedding = np.array(query_embedding, dtype="float32")

    # Search Hnswlib index
    k = 5  # number of similar images
    labels, distances = index.knn_query(query_embedding, k=k)

    st.write("### Top Similar Images")
    for i in range(k):
        matched_path = image_data[labels[0][i]]
        matched_img = Image.open(matched_path["path"])
        st.image(
            matched_img,
            caption=f"Label: {matched_path['label']} (Distance: {distances[0][i]:.2f})",
            use_column_width=True
        )
