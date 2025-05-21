import torch

# Load the checkpoint
checkpoint = torch.load("best_model.pth", map_location="cpu")

# List all parameter names
print("All parameter keys:")
for key in checkpoint.keys():
    print(f"  {key}")

# Try to inspect classifier output layer weights
for k, v in checkpoint.items():
    if "classifier" in k and "weight" in k:
        print(f"\n✅ Found final classifier layer: {k}")
        print(f"Shape of weights: {v.shape} (out_features x in_features)")
        num_classes = v.shape[0]
        print(f"👉 Number of classes = {num_classes}")
        break
