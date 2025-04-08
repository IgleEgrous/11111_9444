import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torchvision.models import VGG16_Weights

def validate():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define directory for magnification "200" validation data
    val_dir = "ColHis-IDS_split/val/200"

    # Define transformations for validation (only resizing and normalization)
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Create validation dataset and dataloader
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Load the model architecture and update classifier
    model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    for param in model.features.parameters():
        param.requires_grad = False

    num_classes = len(val_dataset.classes)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    model.load_state_dict(torch.load("vgg16_model_magnification200.pth", map_location=device))
    model = model.to(device)
    model.eval()

    # Evaluation loop
    running_corrects = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

    val_acc = running_corrects.double() / total
    print(f"Validation Accuracy: {val_acc:.4f}")

if __name__ == '__main__':
    validate()
