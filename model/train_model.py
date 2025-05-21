import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import onnx
from onnx2torch import convert
import os

# ✅ Helper function to analyze the ONNX model structure
def analyze_onnx_model(onnx_model_path):
    model = onnx.load(onnx_model_path)
    print(f"ONNX Model Analysis:")
    print(f"Number of inputs: {len(model.graph.input)}")
    for i, input_tensor in enumerate(model.graph.input):
        print(f"  Input {i}: {input_tensor.name}, Shape: {[d.dim_value for d in input_tensor.type.tensor_type.shape.dim]}")
    
    print(f"Number of outputs: {len(model.graph.output)}")
    for i, output_tensor in enumerate(model.graph.output):
        print(f"  Output {i}: {output_tensor.name}, Shape: {[d.dim_value for d in output_tensor.type.tensor_type.shape.dim]}")
    
    print(f"Number of nodes: {len(model.graph.node)}")
    op_types = {}
    for node in model.graph.node:
        op_types[node.op_type] = op_types.get(node.op_type, 0) + 1
    
    print("Operation types:")
    for op_type, count in op_types.items():
        print(f"  {op_type}: {count}")
    
    return model

# ✅ Create a clean PyTorch model based on the ONNX architecture
class NeuralHashModel(nn.Module):
    def __init__(self, num_classes, input_size=(3, 384, 384)):
        super().__init__()
        
        # Based on common neural hash architectures (adjust as needed)
        self.features = nn.Sequential(
            # Initial convolution layers
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Several residual blocks would normally go here
            # Simplified for this example
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
        
        # Fully connected layers
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

# ✅ Function to transfer weights from ONNX model to our PyTorch model where possible
def transfer_weights(onnx_model, torch_model):
    # This is a simplified approach - actual implementation depends on your specific models
    print("⚠️ Weight transfer is model-specific and might need customization")
    print("Skipping automatic weight transfer - will train from scratch")
    return torch_model

# ✅ Main training function
def train_model(model, train_loader, val_loader=None, epochs=5, lr=1e-4, device="cpu"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    
    model.to(device)
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f"Epoch {epoch+1}/{epochs}")
        print("-" * 10)
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward + optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Print batch progress
            if i % 10 == 0:
                print(f"Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        print(f"Training Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
        
        # Validation phase
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_loss = val_loss / len(val_loader.dataset)
            val_acc = val_correct / val_total
            print(f"Validation Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            
            # Learning rate scheduler step
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "best_model.pth")
                print("Saved best model checkpoint")
    
    print("Training complete!")
    return model

# ✅ Setup data transformations and loaders
def setup_data(data_dir, batch_size=16, img_size=384, val_split=0.2):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load the full dataset
    full_dataset = datasets.ImageFolder(data_dir, transform=transform)
    
    # Split into train and validation
    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Get number of classes
    num_classes = len(full_dataset.classes)
    print(f"Dataset loaded: {train_size} training samples, {val_size} validation samples")
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {full_dataset.classes}")
    
    return train_loader, val_loader, num_classes

# ✅ Main execution
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Analyze the ONNX model
    onnx_model_path = "./model/model.onnx"
    print(f"Analyzing ONNX model: {onnx_model_path}")
    onnx_model = analyze_onnx_model(onnx_model_path)
    
    # Setup data
    data_dir = "./dataset1/cropped"
    train_loader, val_loader, num_classes = setup_data(
        data_dir, batch_size=8, img_size=384
    )
    
    # Create a fresh PyTorch model
    print("Creating new PyTorch model")
    model = NeuralHashModel(num_classes=num_classes)
    
    # Optionally try to transfer weights (might not work perfectly)
    try:
        base_model = convert(onnx_model)
        print("Converting ONNX model to PyTorch")
        model = transfer_weights(base_model, model)
    except Exception as e:
        print(f"Weight transfer failed: {e}")
        print("Continuing with freshly initialized model")
    
    # Train the model
    print("Starting training")
    trained_model = train_model(
        model, 
        train_loader, 
        val_loader, 
        epochs=10, 
        lr=1e-4, 
        device=device
    )
    
    # Save the final model
    torch.save(trained_model.state_dict(), "fine_tuned_model_final.pth")
    print("Saved final model as fine_tuned_model_final.pth")