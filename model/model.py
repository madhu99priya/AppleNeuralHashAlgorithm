from onnx2torch import convert
import onnx
import torch

# Load the ONNX model
onnx_model = onnx.load("./model.onnx")

# Convert to PyTorch model
torch_model = convert(onnx_model)

torch.save(torch_model, 'recovered_model.pth')
print("✅ Model saved as 'recovered_model.pth' successfully!")