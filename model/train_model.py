import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import onnx
from onnx2torch import convert


# ✅ Custom wrapper to intercept addition
class SafeAddModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Run forward pass, intercept addition failures
        def safe_add(a, b):
            if a.shape != b.shape:
                print(f"⚠️ Shape mismatch detected: {a.shape} vs {b.shape} — interpolating.")
                b = F.interpolate(b, size=a.shape[2:], mode="bilinear", align_corners=False)
            return a + b

        # Patch torch.add to our safe_add during forward
        orig_add = torch.add
        torch.add = safe_add
        try:
            out = self.model(x)
        finally:
            torch.add = orig_add  # restore original
        return out


# ✅ Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
onnx_model = onnx.load("./model/model.onnx")
base_model = convert(onnx_model)
base_model.load_state_dict(torch.load("./model/model_weights.pth", map_location=device))
base_model.to(device)
base_model.eval()

# Wrap it with our safe add fix
model = SafeAddModel(base_model)
model.train()

# ✅ Dataset
transform = transforms.Compose([
    transforms.Resize(400),
    transforms.CenterCrop((384, 384)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder("./dataset1/cropped", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# ✅ Training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(1):  # Try 1 epoch to confirm
    total_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        try:
            outputs = model(inputs)
        except Exception as e:
            print(f"❌ Still failed: {e}")
            exit()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"✅ Epoch {epoch+1} completed. Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "fine_tuned_model_safeadd.pth")
print("✅ Fine-tuned model saved as fine_tuned_model_safeadd.pth")
