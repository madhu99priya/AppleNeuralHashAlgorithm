import os
import time 
import random
from PIL import Image
import torch
from torch import nn, optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader

# === Configuration ===
DATA_DIR = "./dataset/cropped"
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Dataset Definition ===
class ImagePairDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = os.listdir(data_dir)
        self.image_pairs = []
        self.labels = []

        for cls in self.classes:
            img_list = os.listdir(os.path.join(data_dir, cls))
            if len(img_list) < 2:
                continue
            for i in range(len(img_list) - 1):
                self.image_pairs.append(((cls, img_list[i]), (cls, img_list[i + 1])))
                self.labels.append(1)  # Positive pair

        # Add negative pairs
        for _ in range(len(self.image_pairs)):
            cls1, cls2 = random.sample(self.classes, 2)
            img1_list = os.listdir(os.path.join(data_dir, cls1))
            img2_list = os.listdir(os.path.join(data_dir, cls2))
            if img1_list and img2_list:
                self.image_pairs.append(((cls1, random.choice(img1_list)), (cls2, random.choice(img2_list))))
                self.labels.append(0)  # Negative pair

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        try:
            (cls1, img1_name), (cls2, img2_name) = self.image_pairs[idx]
            img1_path = os.path.join(self.data_dir, cls1, img1_name)
            img2_path = os.path.join(self.data_dir, cls2, img2_name)

            img1 = Image.open(img1_path).convert("RGB")
            img2 = Image.open(img2_path).convert("RGB")

            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            return img1, img2, torch.tensor(self.labels[idx], dtype=torch.float32)

        except Exception as e:
            print(f"[WARNING] Skipping {idx}: {e}")
            return self.__getitem__((idx + 1) % len(self))

# === NeuralHash-like Head ===
class NeuralHashNet(nn.Module):
    def __init__(self):
        super(NeuralHashNet, self).__init__()
        base = models.mobilenet_v3_large(pretrained=True)

        # Freeze backbone layers
        for param in base.features.parameters():
            param.requires_grad = False

        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.hash_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(960, 128),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.hash_head(x)
        return x


# === Contrastive Loss ===
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
        loss = torch.mean(
            label * torch.pow(euclidean_distance, 2) +
            (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss

# === Training Utilities ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = ImagePairDataset(DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = NeuralHashNet().to(DEVICE)
criterion = ContrastiveLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# === Training Loop ===

start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for img1, img2, label in loader:
        img1, img2, label = img1.to(DEVICE), img2.to(DEVICE), label.to(DEVICE)

        optimizer.zero_grad()
        out1 = model(img1)
        out2 = model(img2)
        loss = criterion(out1, out2, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(loader):.4f}")

end_time = time.time()
total_time = end_time - start_time
print(f"\n⏱️ Total training time: {total_time:.2f} seconds")


# === Save Model ===
torch.save(model.state_dict(), "mobilenetv3_neuralhash.pth")
