import os
import json
import torch
import hnswlib
import numpy as np
from PIL import Image
from model import model, preprocess, device  # existing model setup

# Lists to store embeddings and image metadata
embeddings = []
image_data = []  # stores both image path and label (folder name)

# Walk through dataset folder
for root, dirs, files in os.walk("data/"):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff")):
            image_path = os.path.join(root, file)
            label = os.path.basename(root)
            print(f"Processing: {image_path}")

            # Preprocess and generate normalized embedding
            image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model.encode_image(image)
                embedding /= embedding.norm(dim=-1, keepdim=True)  # normalize for cosine similarity
            embedding = embedding.cpu().numpy()

            embeddings.append(embedding)
            image_data.append({"path": image_path, "label": label})

# Convert list of embeddings into a single numpy array
embeddings = np.vstack(embeddings).astype("float32")

# Initialize HNSWlib index (cosine distance)
dim = embeddings.shape[1]
index = hnswlib.Index(space='cosine', dim=dim)
index.init_index(max_elements=len(embeddings), ef_construction=200, M=16)
index.add_items(embeddings)
index.set_ef(50)  # tradeoff between speed and accuracy

# Ensure output folder exists
os.makedirs("embeddings", exist_ok=True)

# Save index and image metadata
index.save_index("embeddings/index.bin")
with open("embeddings/image_paths.json", "w") as f:
    json.dump(image_data, f, indent=4)

print(f"✅ Indexed {len(image_data)} images successfully using Hnswlib with cosine similarity.")
