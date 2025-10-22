import os
import json
import torch
import faiss
import numpy as np
from PIL import Image
from model import model, preprocess, device  # we already built this in model.py

# Empty lists to hold embeddings and paths
embeddings = []
image_paths = []

# Walk through dataset folder
for root, dirs, files in os.walk("data/"):
    for file in files:
        if file.endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(root, file)
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

            # Get embedding
            with torch.no_grad():
                embedding = model.encode_image(image)
            embedding = embedding.cpu().numpy()

            embeddings.append(embedding)
            image_paths.append(image_path)

# Stack into numpy array
embeddings = np.vstack(embeddings).astype("float32")

# Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Make sure folder exists
os.makedirs("embeddings", exist_ok=True)

# Save index
faiss.write_index(index, "embeddings/index.faiss")

# Save image paths
with open("embeddings/image_paths.json", "w") as f:
    json.dump(image_paths, f)

print(f"Indexed {len(image_paths)} images.")
