import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import VGG16_Weights

def train():
    # Set device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define directories for magnification "200"
    train_dir = "ColHis-IDS_split/train/200"

    # Define transformations for training
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],   # mean for ImageNet
                             [0.229, 0.224, 0.225])   # std for ImageNet
    ])

    # Create training dataset and dataloader
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    # Load pretrained VGG16 using the weights parameter
    model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    # Freeze feature extractor layers
    for param in model.features.parameters():
        param.requires_grad = False

    # Modify classifier for our dataset
    num_classes = len(train_dataset.classes)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-4)

    # Training loop
    epochs = 20
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Statistics
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "vgg16_model_magnification200.pth")
    print("Training complete and model saved.")

if __name__ == '__main__':
    train()
