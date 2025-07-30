import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from torchvision import transforms
from pathlib import Path
import json
from tqdm import tqdm

# Preprocess transform (FaceNet expects 160x160 RGB)
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Load pre-trained FaceNet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def get_embedding(image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(img_tensor)
    return embedding.squeeze(0).cpu().numpy()

def build_embedding_db(cropped_root, output_path):
    cropped_root = Path(cropped_root)
    embedding_db = []

    for person_dir in tqdm(list(cropped_root.iterdir()), desc="Generating embeddings"):
        if not person_dir.is_dir():
            continue

        person_id = person_dir.name
        person_embeddings = []

        for img_file in person_dir.glob("*.jpg"):
            try:
                emb = get_embedding(img_file)
                person_embeddings.append(emb.tolist())
            except Exception as e:
                print(f"Error on {img_file}: {e}")

        if person_embeddings:
            embedding_db.append({
                "person_id": person_id,
                "embeddings": person_embeddings
            })

    with open(output_path, "w") as f:
        json.dump(embedding_db, f)

    print(f"\n✅ Embedding DB saved to {output_path}")

if __name__ == "__main__":
    build_embedding_db("images/cropped_criminals", "embeddings/criminals.json")