import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np

# Configuration
class Config:
    num_classes = 5
    class_names = ['Normal', 'Polyp', 'Low-grade IN', 'High-grade IN', 'Adenocarcinoma']
    data_dir = "./11111_9444/EBH-HE-IDS/EBHI-Split/train"
    val_dir = "./11111_9444/EBH-HE-IDS/EBHI-Split/val"
    batch_size = 32
    learning_rate = 1e-4
    num_epochs = 50
    input_size = 224
    early_stop_patience = 7
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Dataset Class
class HistopathologyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.classes = sorted([d for d in os.listdir(root_dir) 
                             if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = []
        self.transform = transform

        # Collect samples with 200x magnification
        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            for case_id in os.listdir(cls_path):
                case_path = os.path.join(cls_path, case_id, "200")
                if os.path.exists(case_path):
                    for img_name in os.listdir(case_path):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.samples.append((
                                os.path.join(case_path, img_name),
                                self.class_to_idx[cls]
                            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Data Transforms
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomResizedCrop(Config.input_size, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((Config.input_size, Config.input_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Model Architecture
def initialize_model():
    model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    
    # Freeze feature extractor
    for param in model.features.parameters():
        param.requires_grad = False
        
    # Modify classifier
    model.classifier = nn.Sequential(
        nn.Linear(512*7*7, 4096),
        nn.ReLU(True),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(0.5),
        nn.Linear(4096, Config.num_classes)
    )
    return model.to(Config.device)

# Training Process
def train_model():
    # Initialize data loaders
    train_dataset = HistopathologyDataset(Config.data_dir, train_transform)
    val_dataset = HistopathologyDataset(Config.val_dir, val_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    model = initialize_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(Config.num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(Config.device, non_blocking=True)
            labels = labels.to(Config.device, non_blocking=True)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(Config.device, non_blocking=True)
                labels = labels.to(Config.device, non_blocking=True)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        
        # Calculate metrics
        epoch_loss = running_loss / len(train_dataset)
        val_epoch_loss = val_loss / len(val_dataset)
        val_acc = correct / total
        
        print(f"Epoch {epoch+1}/{Config.num_epochs}")
        print(f"Train Loss: {epoch_loss:.4f} | Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Early stopping and model checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved new best model")
        else:
            patience_counter += 1
            if patience_counter >= Config.early_stop_patience:
                print("Early stopping triggered")
                break
    
    print("\nTraining completed")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    train_model()