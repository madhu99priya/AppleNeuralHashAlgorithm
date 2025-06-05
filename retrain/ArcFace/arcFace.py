import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import random_split, DataLoader
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

# === Face Recognition Model (ResNet18) ===
class FaceRecognitionModel(nn.Module):
    def __init__(self, embedding_size=128):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])  # Remove final FC
        self.embedding = nn.Linear(512, embedding_size)  # ResNet18 has 512-dim output

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        return F.normalize(x)

# === Main Training Script ===
if __name__ == "__main__":
    # --- Config ---
    checkpoint_path = 'checkpoint_arcface_resnet18.pth'
    batch_size = 32
    num_epochs = 10
    save_every_n_batches = 100

    # --- Transforms and Dataset ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])

    dataset = datasets.ImageFolder(
        root='D:\\FYP\\NeuralHASH\\AppleNeuralHashAlgorithm\\dataset\\cropped',
        transform=transform
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    num_classes = len(dataset.classes)

    print("No of training images:", train_size)
    print("No of validation images:", val_size)
    print("No of classes:", num_classes)

    # --- Device ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Model, Loss, Optimizer ---
    model = FaceRecognitionModel(embedding_size=128).to(device)
    metric_fc = ArcMarginProduct(128, num_classes, s=20.0, m=0.3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': 1e-4},
        {'params': metric_fc.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-4)

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
            loop.set_postfix(train_loss=f"{avg_loss:.4f}", train_acc=f"{acc:.2f}%")

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

        # Save model at the end of each epoch
        torch.save(model.state_dict(), f'arcface_resnet18_backbone_epoch{epoch+1}.pth')
        torch.save(metric_fc.state_dict(), f'arcface_resnet18_head_epoch{epoch+1}.pth')

        # === Validation Loop ===
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                embeddings = model(images)
                cosine_sim = F.linear(F.normalize(embeddings), F.normalize(metric_fc.weight))
                _, preds = torch.max(cosine_sim, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total * 100
        print(f"🔍 Validation Accuracy: {val_acc:.2f}%")

        loop.set_description(f"Epoch [{epoch+1}/{num_epochs}] (Val Acc: {val_acc:.2f}%)")
