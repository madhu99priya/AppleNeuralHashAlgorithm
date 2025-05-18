# from onnx2torch import convert
# import onnx
# import torch

# # Load the ONNX model
# onnx_model = onnx.load("./model.onnx")

# # Convert to PyTorch model
# torch_model = convert(onnx_model)

# torch.save(torch_model, 'recovered_model.pth')
# print("✅ Model saved as 'recovered_model.pth' successfully!")


from onnx2torch import convert
import onnx
import torch

# Load the ONNX model
onnx_model = onnx.load("./model.onnx")

# Convert to PyTorch model
torch_model = convert(onnx_model)

# (Optional) Save as state_dict so it's easier to manage later
torch.save(torch_model.state_dict(), "model_weights.pth")
print("✅ Model saved as 'model_weights.pth' successfully!")

# print(torch_model)
