# train_arcface.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from neuralhash_face_model import NeuralHashFaceNet
from arc_margin import ArcMarginProduct
from tqdm import tqdm

# Dataset setup
transform = transforms.Compose([
    transforms.Resize((360, 360)),
    transforms.ToTensor()
])
dataset = datasets.ImageFolder('E:/Accedemic/FYP/AppleNeuralHashAlgorithm-main/AppleNeuralHashAlgorithm/dataset1/cropped/', transform=transform)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralHashFaceNet().to(device)
arc_head = ArcMarginProduct(in_features=128, out_features=len(dataset.classes)).to(device)

# Loss + Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(model.parameters()) + list(arc_head.parameters()), lr=1e-3)

# Training loop
for epoch in range(10):
    model.train()
    total_loss = 0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
        imgs, labels = imgs.to(device), labels.to(device)

        features = model(imgs)                # (B, 128)
        logits = arc_head(features, labels)   # (B, num_classes)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"✅ Epoch {epoch} - Loss: {total_loss:.4f}")
