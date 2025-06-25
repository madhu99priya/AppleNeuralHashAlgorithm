from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((360, 360)),
    transforms.ToTensor(),
])

train_set = datasets.ImageFolder(root='E:/Accedemic/FYP/AppleNeuralHashAlgorithm-main/AppleNeuralHashAlgorithm/dataset1/cropped/', transform=transform)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

for images, labels in train_loader:
    print("✅ Batch loaded:")
    print("Images shape:", images.shape)  # should be [batch_size, 3, 360, 360]
    print("Labels:", labels)
    break  # just test the first batch