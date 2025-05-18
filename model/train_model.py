import torch
from torch import nn, optim
from onnx2torch import convert
import onnx
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Convert ONNX to PyTorch
onnx_model = onnx.load("model.onnx")
model = convert(onnx_model)

# Load pretrained weights
model.load_state_dict(torch.load("model_weights.pth", map_location=device))
model.to(device)
model.train()

# Use existing preprocessed dataset
transform = transforms.Compose([
    transforms.Resize((360, 360)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder("../dataset1/cropped", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(5):  # Change epochs as needed
    total_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"✅ Epoch {epoch+1} completed. Loss: {total_loss:.4f}")

# Save fine-tuned model
torch.save(model.state_dict(), "fine_tuned_model.pth")
print("✅ Fine-tuned model saved as fine_tuned_model.pth")
