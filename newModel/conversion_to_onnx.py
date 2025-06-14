import torch
import torch.nn as nn
from torchvision import models
import os

# === NeuralHashNet (must match the one in training script) ===
class NeuralHashNet(nn.Module):
    def __init__(self):
        super(NeuralHashNet, self).__init__()
        base = models.mobilenet_v3_large(pretrained=False)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.hash_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(960, 128),  # Hash size
            nn.Tanh()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.hash_head(x)
        return x

# === Load trained weights ===
model = NeuralHashNet()
model.load_state_dict(torch.load("mobilenetv3_neuralhash.pth", map_location="cpu"))
model.eval()

# === Dummy input for export ===
dummy_input = torch.randn(1, 3, 224, 224)  # (batch_size, channels, height, width)

# === Export to ONNX ===
torch.onnx.export(
    model,
    dummy_input,
    "mobilenetv3_neuralhash.onnx",
    input_names=["input"],
    output_names=["hash_output"],
    dynamic_axes={"input": {0: "batch_size"}, "hash_output": {0: "batch_size"}},
    opset_version=11
)

print("✅ Successfully converted to mobilenetv3_neuralhash.onnx")
