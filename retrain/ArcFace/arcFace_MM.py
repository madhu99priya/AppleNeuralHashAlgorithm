import torch
from arcFace import FaceRecognitionModel


# Instantiate and load your trained backbone model
model = FaceRecognitionModel(embedding_size=128)
model.load_state_dict(torch.load('arcface_resnet18_backbone_epoch9.pth'))  # adjust epoch filename if needed
model.eval()

# Create a dummy input tensor with the right shape
dummy_input = torch.randn(1, 3, 224, 224)  # same size as training input

# Export to ONNX
onnx_path = 'arcface_resnet18_backbone_epoch9.onnx'
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=['input'],
    output_names=['embedding'],
    opset_version=12,
    dynamic_axes={
        'input': {0: 'batch_size'},
        'embedding': {0: 'batch_size'},
    },
    verbose=True
)

print(f"Model exported to {onnx_path}")
