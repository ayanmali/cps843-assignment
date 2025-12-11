"""
training the model with K-fold cross validation
"""
import sys
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import create_kfold_splits, get_fold_data
from src.model.dataset import CharacterDataset
from src.model.mobilenet import create_mobilenet_model
from src.model.vgg16 import create_vgg16_model
from src.labels import LABELS

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def train_with_kfold(
    n_splits: int = 5,
    train_csv: str = "iiit-5k/traindata.csv",
    base_dir: str = "iiit-5k/IIIT5K-Word_V3.0/IIIT5K",
    model_type: str = "mobilenet",  # "mobilenet" or "vgg16"
    batch_size: int = 32,
    num_epochs: int = 10,
    learning_rate: float = 0.001,
    shuffle: bool = True,
    random_state: int = 42,
    device: str = None
):
    """
    Train model using K-fold cross validation.
    
    Args:
        n_splits: Number of folds for cross validation (default: 5)
        train_csv: Path to training CSV file
        base_dir: Base directory where images are stored
        model_type: Type of model to use ("mobilenet" or "vgg16")
        batch_size: Batch size for training
        num_epochs: Number of training epochs per fold
        learning_rate: Learning rate for optimizer
        shuffle: Whether to shuffle data before splitting
        random_state: Random seed for reproducibility
        device: Device to use ("cuda" or "cpu"), auto-detects if None
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"Using device: {device}")
    
    # Load CSV to get image-level splits
    train_df = pd.read_csv(train_csv)
    print(f"\nCreating {n_splits}-fold cross validation splits...")
    fold_generator = create_kfold_splits(train_df, n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    
    # Store results for each fold
    fold_results = []
    
    # Iterate through each fold
    for fold_num, (train_indices, val_indices) in enumerate(fold_generator, 1):
        print(f"\n{'='*60}")
        print(f"Fold {fold_num}/{n_splits}")
        print(f"{'='*60}")
        
        # Get image-level splits
        train_df_fold, val_df_fold = get_fold_data(train_df, train_indices, val_indices)
        
        print(f"Training images: {len(train_df_fold)}")
        print(f"Validation images: {len(val_df_fold)}")
        
        # Create datasets for this fold (only process images in this fold)
        print("\nProcessing training images...")
        train_dataset = CharacterDataset(
            df=train_df_fold,
            base_dir=base_dir,
            image_indices=None  # Already filtered by df
        )
        
        print("\nProcessing validation images...")
        val_dataset = CharacterDataset(
            df=val_df_fold,
            base_dir=base_dir,
            image_indices=None  # Already filtered by df
        )
        
        print(f"Training characters: {len(train_dataset)}")
        print(f"Validation characters: {len(val_dataset)}")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0  # Set to 0 to avoid multiprocessing issues
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # Create model
        num_classes = len(LABELS)
        if model_type == "mobilenet":
            model = create_mobilenet_model(num_classes=num_classes)
        elif model_type == "vgg16":
            model = create_vgg16_model(num_classes=num_classes)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model = model.to(device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        best_val_acc = 0.0
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            
            # Validate
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
        
        # Store fold results
        fold_results.append({
            'fold': fold_num,
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
            'best_val_accuracy': best_val_acc
        })
    
    # Print summary
    print(f"\n{'='*60}")
    print("K-Fold Cross Validation Summary")
    print(f"{'='*60}")
    if fold_results:
        accuracies = [r['best_val_accuracy'] for r in fold_results]
        print(f"Mean validation accuracy: {np.mean(accuracies):.4f}%")
        print(f"Std validation accuracy: {np.std(accuracies):.4f}%")
        print("\nPer-fold results:")
        for result in fold_results:
            print(f"  Fold {result['fold']}: {result['best_val_accuracy']:.2f}% "
                  f"(train: {result['train_size']}, val: {result['val_size']})")


if __name__ == "__main__":
    # Example usage
    train_with_kfold(
        n_splits=5,
        model_type="mobilenet",  # or "vgg16"
        batch_size=32,
        num_epochs=10,
        learning_rate=0.001
    )
    
    # You can also customize parameters:
    # train_with_kfold(
    #     n_splits=10,
    #     model_type="vgg16",
    #     batch_size=64,
    #     num_epochs=20,
    #     learning_rate=0.0001,
    #     shuffle=True,
    #     random_state=42
    # )
