"""
Training the character classifier models (LeNet and VGG16)
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.labels import LABELS
from src.preprocessing import load_img

TARGET_SIZE = (24, 24)

class FolderCharacterDataset(Dataset):
    def __init__(
        self,
        base_dir: str,
        split: str = "train",
        target_size: Tuple[int, int] = TARGET_SIZE,
        transform=None,
    ):
        self.base_dir = base_dir
        self.split = split
        self.target_size = target_size
        self.transform = transform
        
        self.label_to_idx = {char: idx for idx, char in enumerate(LABELS)}
        self.idx_to_label = {idx: char for char, idx in self.label_to_idx.items()}
        self.num_classes = len(LABELS)
        
        self.image_paths = []
        self.labels = []
        self.label_strings = []
        self._load_images()
        
        print(f"Loaded {len(self.image_paths)} images for {self.split} split")
        print("Class distribution:")
        label_counts = {}
        for label_str in self.label_strings:
            label_counts[label_str] = label_counts.get(label_str, 0) + 1
        for label_str, count in sorted(label_counts.items()):
            print(f"  {label_str}: {count}")
    
    def _load_images(self):
        """Load all images from the folder structure."""
        split_dir = os.path.join(self.base_dir, f"char_{self.split}")
        
        if not os.path.exists(split_dir):
            raise ValueError(f"Directory {split_dir} does not exist")
        
        for case in ["cap", "lc"]:
            case_dir = os.path.join(split_dir, case)
            if not os.path.exists(case_dir):
                continue
            
            for letter_folder in sorted(os.listdir(case_dir)):
                letter_path = os.path.join(case_dir, letter_folder)
                if not os.path.isdir(letter_path):
                    continue
                
                label = letter_folder
                
                if label not in self.label_to_idx:
                    print(f"Warning: Label '{label}' not in LABELS, skipping folder {letter_path}")
                    continue
                
                image_files = [f for f in os.listdir(letter_path) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                for img_file in image_files:
                    img_path = os.path.join(letter_path, img_file)
                    self.image_paths.append(img_path)
                    self.labels.append(self.label_to_idx[label])
                    self.label_strings.append(label)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):        
        img_path = self.image_paths[idx]
        label_idx = self.labels[idx]
            
        img = load_img(img_path)
        if img is None:
            # If image fails to load, return a black image
            img = np.zeros((*self.target_size, 3), dtype=np.uint8)
        
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if img.shape[:2] != self.target_size:
            img = cv2.resize(img, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_AREA)

        img_normalized = img.astype(np.float32) / 255.0
        
        image_tensor = torch.from_numpy(img_normalized).unsqueeze(0)
        
        if self.transform:
            image_tensor = self.transform(image_tensor)
        
        return image_tensor, label_idx


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
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


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: str = "training_curves.png"
):
 
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nTraining curves saved to: {save_path}")
    plt.close()

def save_model(model: nn.Module, model_path: str, metadata: dict = None):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    save_dict = {
        'model_state_dict': model.state_dict(),
        'metadata': metadata or {}
    }
    
    torch.save(save_dict, model_path)
    print(f"Model saved to: {model_path}")


def train_model(
    base_dir: str,
    model_type: str,
    loss_function: str,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    device,
    model_save_path: str,
    plot_save_path: str,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"Using device: {device}")
    
    from src.model.lenet import create_lenet_model
    
    print("\nLoading training dataset...")
    train_dataset = FolderCharacterDataset(
        base_dir=base_dir,
        split="train",
        target_size=TARGET_SIZE
    )
    
    print("\nLoading validation dataset...")
    val_dataset = FolderCharacterDataset(
        base_dir=base_dir,
        split="val",
        target_size=TARGET_SIZE
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    num_classes = len(LABELS)
    if model_type == "lenet":
        model = create_lenet_model(num_classes=num_classes)
    elif model_type == "vgg16":
        from src.model.vgg16 import create_vgg16_model
        model = create_vgg16_model(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from: 'lenet', 'mobilenet-small', 'vgg16'")
    
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    if loss_function == "cross_entropy":
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_function}.")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_acc = 0.0
    best_model_state = None
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 60)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"New best validation accuracy: {best_val_acc:.2f}%")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nLoaded best model with validation accuracy: {best_val_acc:.2f}%")
    
    save_model(
        model=model,
        model_path=model_save_path,
        metadata={
            'model_type': model_type,
            'best_val_accuracy': best_val_acc,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
            'num_classes': len(LABELS),
            'target_size': TARGET_SIZE,
        }
    )
    
    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs,
        save_path=plot_save_path
    )
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {model_save_path}")
    print(f"Training curves saved to: {plot_save_path}")
    print("=" * 60)


if __name__ == "__main__":
    # example 
    train_model(
        base_dir=".",  # Base directory containing char_train/ and char_val/ folders
        model_type="lenet",  # "lenet" or "vgg16"
        loss_function="cross_entropy",  # "cross_entropy"
        batch_size=32,
        num_epochs=10,
        learning_rate=0.001,
        device=None,
        model_save_path="saved_models/character_classifier.pth",
        plot_save_path="training_curves.png",
    )

