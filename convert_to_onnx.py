import torch
import torch.nn as nn

# ✅ Define your original model architecture (must match training time)
class NeuralHashModel(nn.Module):
    def __init__(self, num_classes=5, input_size=(3, 384, 384)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ✅ Step 1: Load checkpoint and infer number of classes
state_dict = torch.load("best_model.pth", map_location="cpu")

# Infer num_classes from classifier.4.weight shape
num_classes = state_dict["classifier.4.weight"].shape[0]
print(f"🔍 Detected number of classes: {num_classes}")

# ✅ Step 2: Reconstruct model
model = NeuralHashModel(num_classes=num_classes)
model.load_state_dict(state_dict)
model.eval()

# ✅ Step 3: Dummy input for export
dummy_input = torch.randn(1, 3, 384, 384)

# ✅ Step 4: Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "converted_neuralhash.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)

print("✅ Successfully exported to converted_neuralhash.onnx")
