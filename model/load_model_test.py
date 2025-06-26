import torch
from neuralhash_face_model import NeuralHashFaceNet  

# Step 1: Create model instance
model = NeuralHashFaceNet()

# Step 2: Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Step 3: Run a dummy input to test
dummy_input = torch.randn(1, 3, 360, 360).to(device)
with torch.no_grad():
    output = model(dummy_input)

# Step 4: Print the output shape
print("✅ Model loaded successfully.")
print("Output shape:", output.shape)  # Expect: torch.Size([1, 128])
