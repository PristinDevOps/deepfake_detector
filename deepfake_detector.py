import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
import random
import time
import copy

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Get dataset path
def get_dataset_path():
    default_path = Path(os.path.expanduser("~/.cache/kagglehub/datasets/manjilkarki/deepfake-and-real-images/Dataset"))
    if default_path.exists():
        return default_path
    
    # Ask for dataset path if not found at default location
    path_str = input("Enter the path to the dataset folder: ")
    path = Path(path_str)
    if path.exists():
        return path
    else:
        raise ValueError(f"Dataset path not found: {path}")

# Define dataset class
class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, split='Train', transform=None):
        """
        Args:
            root_dir (string): Root directory of the dataset.
            split (string): 'Train', 'Test', or 'Validation'.
            transform (callable, optional): Transform to be applied on images.
        """
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Collect real images
        real_dir = self.root_dir / 'Real'
        for img_path in real_dir.glob('*.jpg'):
            self.image_paths.append(img_path)
            self.labels.append(1)  # 1 for real
        
        # Collect fake images
        fake_dir = self.root_dir / 'Fake'
        for img_path in fake_dir.glob('*.jpg'):
            self.image_paths.append(img_path)
            self.labels.append(0)  # 0 for fake
        
        # Shuffle the dataset
        combined = list(zip(self.image_paths, self.labels))
        random.shuffle(combined)
        self.image_paths, self.labels = zip(*combined)
        
        print(f"Loaded {len(self.labels)} images for {split} split")
        print(f"Real: {self.labels.count(1)}, Fake: {self.labels.count(0)}")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Define data transforms
def get_transforms():
    # Data augmentation and normalization for training
    # Just normalization for validation and test
    data_transforms = {
        'Train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'Validation': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'Test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms

# Create data loaders
def get_dataloaders(dataset_path, batch_size=32):
    data_transforms = get_transforms()
    
    # Create datasets
    image_datasets = {
        split: DeepfakeDataset(
            root_dir=dataset_path,
            split=split,
            transform=data_transforms[split]
        )
        for split in ['Train', 'Validation', 'Test']
    }
    
    # Create dataloaders
    dataloaders = {
        split: DataLoader(
            image_datasets[split], 
            batch_size=batch_size,
            shuffle=True if split == 'Train' else False,
            num_workers=0  # Set to 0 for Windows compatibility
        )
        for split in ['Train', 'Validation', 'Test']
    }
    
    dataset_sizes = {split: len(image_datasets[split]) for split in ['Train', 'Validation', 'Test']}
    
    return dataloaders, dataset_sizes

# Define the model
def get_model(num_classes=2):
    # Load a pretrained ResNet50 model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # Freeze early layers
    for param in list(model.parameters())[:-20]:
        param.requires_grad = False
    
    # Replace the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, num_classes)
    )
    
    return model.to(device)

# Training function
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # For tracking metrics
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['Train', 'Validation']:
            if phase == 'Train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass + optimize only if in training phase
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'Train' and scheduler is not None:
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Record history
            if phase == 'Train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
            
            # Deep copy the model if best validation accuracy
            if phase == 'Validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best validation accuracy: {best_acc:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Plot training history
    plot_training_history(history)
    
    return model, history

# Plot training history
def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history plot saved as 'training_history.png'")

# Evaluate model on test set
def evaluate_model(model, dataloader, dataset_size):
    model.eval()
    
    running_corrects = 0
    all_preds = []
    all_labels = []
    
    # Iterate over test data
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Statistics
            running_corrects += torch.sum(preds == labels.data)
            
            # Store for confusion matrix
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate accuracy
    test_acc = running_corrects.double() / dataset_size
    print(f'Test Accuracy: {test_acc:.4f}')
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Fake', 'Real']))
    
    return test_acc, cm

# Save model
def save_model(model, path='deepfake_detector_model.pth'):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# Main function
def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Set parameters
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    
    # Get dataset path
    try:
        dataset_path = get_dataset_path()
        print(f"Dataset path: {dataset_path}")
        
        # Create dataloaders
        dataloaders, dataset_sizes = get_dataloaders(dataset_path, batch_size)
        
        # Initialize model
        model = get_model(num_classes=2)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Use learning rate scheduler
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        # Train model
        model, history = train_model(
            model, dataloaders, dataset_sizes, 
            criterion, optimizer, scheduler, num_epochs
        )
        
        # Evaluate on test set
        test_acc, confusion_mat = evaluate_model(
            model, dataloaders['Test'], dataset_sizes['Test']
        )
        
        # Save model
        save_model(model)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 