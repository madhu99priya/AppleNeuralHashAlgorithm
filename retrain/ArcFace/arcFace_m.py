import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
import math
from tqdm import tqdm
import os

# === ArcMarginProduct Layer ===
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

# === Face Recognition Model ===
class FaceRecognitionModel(nn.Module):
    def __init__(self, embedding_size=128):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.embedding = nn.Linear(2048, embedding_size)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        return F.normalize(x)

# === Main Training Script ===
if __name__ == "__main__":
    # --- Config ---
    checkpoint_path = 'checkpoint_arcface_mid.pth'
    batch_size = 32
    num_epochs = 5
    save_every_n_batches = 100

    # --- Transforms and Dataset ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])

    train_dataset = datasets.ImageFolder(
        root='D:\\FYP\\NeuralHASH\\AppleNeuralHashAlgorithm\\dataset\\test',
        transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    num_classes = len(train_dataset.classes)

    # --- Device ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Model, Loss, Optimizer ---
    model = FaceRecognitionModel(embedding_size=128).to(device)
    metric_fc = ArcMarginProduct(128, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(model.parameters()) + list(metric_fc.parameters()), lr=1e-3)

    # --- Checkpoint Resume ---
    start_epoch = 0
    resume_batch = 0
    if os.path.exists(checkpoint_path):
        print("🔁 Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        metric_fc.load_state_dict(checkpoint['metric_fc_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        resume_batch = checkpoint.get('batch_idx', 0)
        print(f"✅ Resumed from epoch {start_epoch}, batch {resume_batch}")
    else:
        print("🆕 Starting new training")

    # --- Training Loop ---
    batch_count = 0
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch [{epoch+1}/{num_epochs}]")
        for batch_idx, (images, labels) in loop:
            # Skip previously processed batches if resuming
            if epoch == start_epoch and batch_idx < resume_batch:
                continue

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            embeddings = model(images)
            logits = metric_fc(embeddings, labels)
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

            batch_count += 1
            if batch_count % save_every_n_batches == 0:
                torch.save({
                    'epoch': epoch,
                    'batch_idx': batch_idx,
                    'model_state_dict': model.state_dict(),
                    'metric_fc_state_dict': metric_fc.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, checkpoint_path)
                print(f"💾 Saved checkpoint at epoch {epoch}, batch {batch_idx}")

        # Save full model after each epoch
        torch.save(model.state_dict(), f'arcface_backbone_epoch{epoch+1}.pth')
        torch.save(metric_fc.state_dict(), f'arcface_head_epoch{epoch+1}.pth')
