import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

# === Step 1: Load Backbone Model ===
class FaceRecognitionModel(torch.nn.Module):
    def __init__(self, embedding_size=128):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])  # remove FC
        self.embedding = nn.Linear(512, embedding_size)

        

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        return F.normalize(x)

# Load model weights
model = FaceRecognitionModel(embedding_size=128)
model.load_state_dict(torch.load("arcface_resnet18_backbone_epoch6.pth", map_location='cpu'))
model.eval()

# === Step 2: Define Preprocessing ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

def load_and_preprocess(image_path):
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0)  # shape: (1, 3, 224, 224)

# === Step 3: Load and Process Two Images ===
img1 = load_and_preprocess("D:\\FYP\\NeuralHASH\\AppleNeuralHashAlgorithm\\dataset\\cropped\\n000006\\0010_01.jpg")
img2 = load_and_preprocess("D:\\FYP\\NeuralHASH\\AppleNeuralHashAlgorithm\\dataset\\cropped\\n000006\\0002_02.jpg")

# === Step 4: Get Embeddings ===
with torch.no_grad():
    emb1 = model(img1)
    emb2 = model(img2)

# === Step 5: Compute Cosine Similarity ===
cosine_sim = F.cosine_similarity(emb1, emb2).item()
print(f"Cosine Similarity: {cosine_sim:.4f}")

print('feature_Vector',emb1)
print('feature_Vector',emb2)

print(len(emb1))

# Optional interpretation
if cosine_sim > 0.8:
    print("✅ Likely same person")
else:
    print("❌ Likely different persons")
