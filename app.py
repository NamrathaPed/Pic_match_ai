import streamlit as st
from PIL import Image
import hnswlib
import json
import numpy as np
from model import get_image_embedding
import os
import tempfile
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
import torch

# Define paths
EMBEDDINGS_DIR = "embeddings"
IMAGE_DIR = "data"

# Streamlit page setup
st.set_page_config(page_title="Pic Match AI", page_icon="🪄", layout="wide")
st.title("🪄 Pic Match AI")
st.caption("Upload an image to find similar items, get AI description, and ask questions about it!")

# Load image metadata
image_json_path = os.path.join(EMBEDDINGS_DIR, "image_paths.json")
if not os.path.exists(image_json_path):
    st.error("❌ Image metadata not found. Please run index.py first.")
    st.stop()

with open(image_json_path, "r") as f:
    image_data = json.load(f)

# Load HNSWlib index
dim = 512  # CLIP ViT-B/32 embedding size
index = hnswlib.Index(space='cosine', dim=dim)
index.load_index(os.path.join(EMBEDDINGS_DIR, "index.bin"))

# Load BLIP models
@st.cache_resource
def load_blip_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Captioning model for rich descriptions
    caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    caption_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)

    # VQA model for Q&A about images
    vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)

    return caption_processor, caption_model, vqa_processor, vqa_model, device

caption_processor, caption_model, vqa_processor, vqa_model, device = load_blip_models()

# Upload image
uploaded_file = st.file_uploader(
    "📸 Upload or drag an image here",
    type=["jpg", "jpeg", "png", "webp", "bmp", "tiff"]
)

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # AI-generated caption for uploaded image
    st.subheader("🤖 AI Image Description")
    with st.spinner("Analyzing image..."):
        caption_inputs = caption_processor(image, return_tensors="pt").to(device)
        caption_output = caption_model.generate(**caption_inputs, max_new_tokens=50)
        description = caption_processor.decode(caption_output[0], skip_special_tokens=True)
    st.success(description)

    # Save temporarily for embedding
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    image.save(temp_file.name, format="JPEG")

    # Compute embedding
    query_embedding = get_image_embedding(temp_file.name)
    query_embedding = np.array(query_embedding, dtype="float32")
    query_embedding /= np.linalg.norm(query_embedding, axis=1, keepdims=True)

    # Search top-k similar images
    k = 5
    labels, distances = index.knn_query(query_embedding, k=k)

    # Display top-k similar images
    st.subheader("🔍 Top Similar Images")
    cols = st.columns(k)
    for i in range(k):
        matched_info = image_data[labels[0][i]]
        matched_img = Image.open(matched_info["path"])
        with cols[i]:
            st.image(
                matched_img,
                caption=f"{matched_info['label']}  \n(Similarity: {(1 - distances[0][i]):.2f})",
                use_container_width=True
            )

    # --- Chat-style AI Q&A about uploaded image + retrieved items ---
    st.subheader("💬 Ask AI About the Uploaded Image")
    user_question = st.text_input("Type your question here:")

    if user_question:
        with st.spinner("🤔 Thinking..."):
            # Build a textual context of top-k retrieved images
            context_text = "The system found these similar items: " + ", ".join(
                [image_data[labels[0][i]]['label'] for i in range(k)]
            )
            full_question = f"{context_text}. Question: {user_question}"

            # Feed the uploaded image + context question to BLIP VQA
            vqa_inputs = vqa_processor(image, full_question, return_tensors="pt").to(device)
            vqa_output = vqa_model.generate(**vqa_inputs, max_new_tokens=50)
            answer = vqa_processor.decode(vqa_output[0], skip_special_tokens=True).strip()

            # Fallback: use caption model if answer is uninformative
            if not answer or answer.lower() in ["none", "no", "n/a", "nothing"]:
                caption_inputs = caption_processor(image, return_tensors="pt").to(device)
                caption_output = caption_model.generate(**caption_inputs, max_new_tokens=50)
                caption = caption_processor.decode(caption_output[0], skip_special_tokens=True)
                answer = f"I found {caption.lower()}."

        st.markdown(f"**🤖 AI Answer:** {answer}")
