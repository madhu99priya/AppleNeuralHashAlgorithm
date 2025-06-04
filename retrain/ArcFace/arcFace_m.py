import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
import math
from tqdm import tqdm

# === Step 1: ArcFace Loss Layer ===
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.clamp(cosine ** 2, 0.0, 1.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return logits * self.s

# === Step 2: Backbone + Embedding ===
class FaceRecognitionModel(nn.Module):
    def __init__(self, embedding_size=128):
        super().__init__()
        # Updated pretrained loading to use weights enum
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])  # remove final fc
        self.embedding = nn.Linear(2048, embedding_size)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        return F.normalize(x)

# === Main training code ===
if __name__ == "__main__":
    # === Step 3: Data Loading ===
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])

    train_dataset = datasets.ImageFolder(root='D:\\FYP\\NeuralHASH\\AppleNeuralHashAlgorithm\\dataset\\cropped', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    num_classes = len(train_dataset.classes)

    # === Step 4: Setup Model, Optimizer, Loss ===
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = FaceRecognitionModel(embedding_size=128).to(device)
    metric_fc = ArcMarginProduct(128, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(model.parameters()) + list(metric_fc.parameters()), lr=1e-3)

    # === Step 5: Training Loop with tqdm ===
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            embeddings = model(images)                     # [B, 128]
            logits = metric_fc(embeddings, labels)         # [B, num_classes]
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            avg_loss = total_loss / total
            acc = correct / total * 100

            loop.set_postfix(loss=f"{avg_loss:.4f}", accuracy=f"{acc:.2f}%")

    # === Step 6: Save Model ===
    torch.save(model.state_dict(), 'arcface_backbone.pth')
    torch.save(metric_fc.state_dict(), 'arcface_head.pth')
