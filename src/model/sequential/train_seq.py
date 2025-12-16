"""
Training pipeline for CRNN sequence recognition on word images.
Images are named as: {word}_{number}.png where {word} is the ground truth label.
"""
import os
from model.sequential.crnn import CRNN
import torch
import torch.nn as nn
from src.preprocessing import load_img
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TARGET_HEIGHT = 32

LABELS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
]

class WordSequenceDataset(Dataset):
    def __init__(
        self,
        base_dir: str,
        split: str = "train", 
        target_height: int = TARGET_HEIGHT,
        transform=None,
    ):

        self.base_dir = base_dir
        self.split = split
        self.target_height = target_height
        self.transform = transform

        self.char_to_idx = {char: idx for idx, char in enumerate(LABELS)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.num_classes = len(LABELS) + 1
        self.blank_idx = len(LABELS)

        self.image_paths = []
        self.labels = []
        self._load_images()

        print(f"Loaded {len(self.image_paths)} images for {split} split")
        print(f"Number of character classes: {len(LABELS)} (+ 1 blank = {self.num_classes})")

        if len(self.labels) > 0:
            print("Sample labels:")
            for i in range(min(5, len(self.labels))):
                label_str = ''.join([self.idx_to_char[idx] for idx in self.labels[i]])
                print(f"  {label_str}")

    def _load_images(self):
        split_dir = os.path.join(self.base_dir, f"word_{self.split}")

        if not os.path.exists(split_dir):
            raise ValueError(f"Directory {split_dir} does not exist")

        image_files = [f for f in os.listdir(split_dir)
                      if f.lower().endswith('.png')]

        for img_file in sorted(image_files):
            img_path = os.path.join(split_dir, img_file)

            name_without_ext = os.path.splitext(img_file)[0]
            parts = name_without_ext.split('_')

            if len(parts) < 2:
                print(f"Warning: Could not parse label from filename {img_file}, skipping...")
                continue

            word = '_'.join(parts[:-1])

            label_seq = []
            for char in word:
                if char in self.char_to_idx:
                    label_seq.append(self.char_to_idx[char])
                else:
                    print(f"Warning: Character '{char}' not in LABELS, skipping from word '{word}'")

            if len(label_seq) == 0:
                print(f"Warning: No valid characters in word '{word}', skipping...")
                continue

            self.image_paths.append(img_path)
            self.labels.append(label_seq)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        import cv2

        img_path = self.image_paths[idx]
        label_seq = self.labels[idx]

        img = load_img(img_path)
        if img is None:
            img = np.zeros((self.target_height, 100), dtype=np.uint8)
        else:
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if img.max() <= 1:
                img = (img * 255).astype(np.uint8)

            h, w = img.shape[:2]
            scale = self.target_height / h
            new_w = int(w * scale)
            img = cv2.resize(img, (new_w, self.target_height), interpolation=cv2.INTER_AREA)

        img_normalized = img.astype(np.float32) / 255.0

        image_tensor = torch.from_numpy(img_normalized).unsqueeze(0)

        if self.transform:
            image_tensor = self.transform(image_tensor)

        label_tensor = torch.tensor(label_seq, dtype=torch.long)

        return image_tensor, label_tensor, len(label_seq)

def collate_fn(batch):

    images, labels, label_lengths = zip(*batch)

    max_width = max(img.shape[2] for img in images)

    padded_images = []
    image_widths = []
    for img in images:
        _, h, w = img.shape
        image_widths.append(w)
        padding = max_width - w
        padded_img = torch.nn.functional.pad(img, (0, padding), mode='constant', value=0)
        padded_images.append(padded_img)

    images_tensor = torch.stack(padded_images)

    labels_tensor = torch.cat(labels)

    label_lengths_tensor = torch.tensor(label_lengths, dtype=torch.long)
    image_widths_tensor = torch.tensor(image_widths, dtype=torch.long)

    return images_tensor, labels_tensor, label_lengths_tensor, image_widths_tensor


def calculate_input_lengths(image_widths: torch.Tensor) -> torch.Tensor:

    return (image_widths // 4).long()


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total_samples = 0

    for images, labels, label_lengths, image_widths in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)
        image_widths = image_widths.to(device)

        input_lengths = calculate_input_lengths(image_widths)
        input_lengths = input_lengths.to(device)

        optimizer.zero_grad()
        logits = model(images)

        log_probs = nn.functional.log_softmax(logits, dim=2)

        loss = criterion(log_probs, labels, input_lengths, label_lengths)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_samples += images.size(0)

    epoch_loss = running_loss / len(dataloader)
    return epoch_loss


def calculate_edit_distance(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + 1
                )

    return dp[m][n]


def calculate_cer(predicted: str, reference: str) -> float:
    # CER = Edit Distance / Length of Reference
    if len(reference) == 0:
        return 1.0 if len(predicted) > 0 else 0.0

    edit_dist = calculate_edit_distance(predicted, reference)
    return edit_dist / len(reference)


def validate(model, dataloader, criterion, device, idx_to_char):
    # validate model w/ CER
    model.eval()
    running_loss = 0.0
    total_samples = 0
    total_cer = 0.0
    total_chars = 0

    with torch.no_grad():
        for images, labels, label_lengths, image_widths in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)
            label_lengths = label_lengths.to(device)
            image_widths = image_widths.to(device)

            input_lengths = calculate_input_lengths(image_widths)
            input_lengths = input_lengths.to(device)

            logits = model(images)
            log_probs = nn.functional.log_softmax(logits, dim=2)

            loss = criterion(log_probs, labels, input_lengths, label_lengths)

            running_loss += loss.item()
            total_samples += images.size(0)

            predictions = log_probs.argmax(dim=2)
            predictions = predictions.permute(1, 0)

            batch_size = predictions.size(0)
            for i in range(batch_size):
                pred_seq = predictions[i].cpu().numpy()
                decoded = []
                prev = None
                for idx in pred_seq:
                    if idx != len(LABELS) and idx != prev:
                        decoded.append(idx_to_char[idx])
                    prev = idx
                pred_str = ''.join(decoded)

                start_idx = sum(label_lengths[:i].cpu().numpy())
                end_idx = start_idx + label_lengths[i].cpu().item()
                gt_seq = labels[start_idx:end_idx].cpu().numpy()
                gt_str = ''.join([idx_to_char[idx] for idx in gt_seq])

                cer = calculate_cer(pred_str, gt_str)
                total_cer += cer * len(gt_str)
                total_chars += len(gt_str)

    epoch_loss = running_loss / len(dataloader)
    avg_cer = total_cer / total_chars if total_chars > 0 else 1.0
    char_accuracy = (1.0 - avg_cer) * 100.0

    return epoch_loss, avg_cer, char_accuracy

def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    val_accs: List[float],
    save_path: str = "training_curves_seq.png"
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

    ax2.plot(epochs, val_accs, 'g-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Validation Accuracy', fontsize=14)
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
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    device,
    model_save_path: str,
    plot_save_path: str,
    rnn_hidden_size: int,
    cnn_out_channels: int,
    early_stopping_patience: int,
    early_stopping_min_delta: float
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"Using device: {device}")

    print("\nLoading training dataset...")
    train_dataset = WordSequenceDataset(
        base_dir=base_dir,
        split="train",
        target_height=TARGET_HEIGHT
    )

    print("\nLoading validation dataset...")
    val_dataset = WordSequenceDataset(
        base_dir=base_dir,
        split="val",
        target_height=TARGET_HEIGHT
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    num_classes = len(LABELS) + 1
    model = CRNN(
        num_classes=num_classes,
        rnn_hidden_size=rnn_hidden_size,
        cnn_out_channels=cnn_out_channels,
        dropout=0.35,
        num_rnn_layers=1
    )
    model = model.to(device)

    criterion = nn.CTCLoss(blank=len(LABELS), reduction='mean', zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    idx_to_char = {idx: char for idx, char in enumerate(LABELS)}
    idx_to_char[len(LABELS)] = '<blank>'

    train_losses = []
    val_losses = []
    val_cers = []
    val_accs = []

    best_val_acc = 0.0
    best_val_cer = 1.0
    best_model_state = None
    epochs_without_improvement = 0

    print(f"\nStarting training for {num_epochs} epochs...")
    if early_stopping_patience > 0:
        print(f"Early stopping enabled: patience={early_stopping_patience}, min_delta={early_stopping_min_delta}")
    print("=" * 60)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 60)

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        val_loss, val_cer, val_acc = validate(model, val_loader, criterion, device, idx_to_char)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_cers.append(val_cer)
        val_accs.append(val_acc)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val CER: {val_cer:.4f}, Val Char Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc + early_stopping_min_delta:
            best_val_acc = val_acc
            best_val_cer = val_cer
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
            print(f"New best validation: CER={best_val_cer:.4f}, Char Acc={best_val_acc:.2f}%")
        else:
            epochs_without_improvement += 1
            if early_stopping_patience > 0:
                print(f"No improvement for {epochs_without_improvement} epochs (best CER: {best_val_cer:.4f}, best Acc: {best_val_acc:.2f}%)")

        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            print(f"\nEarly stopping after {epoch + 1} epochs")
            print(f"No improvement in validation accuracy for {early_stopping_patience} consecutive epochs.")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nLoaded best model: CER={best_val_cer:.4f}, Char Acc={best_val_acc:.2f}%")

    save_model(
        model=model,
        model_path=model_save_path,
        metadata={
            'model_type': 'CRNN',
            'best_val_cer': best_val_cer,
            'best_val_char_accuracy': best_val_acc,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
            'num_classes': num_classes,
            'target_height': TARGET_HEIGHT,
            'rnn_hidden_size': rnn_hidden_size,
            'cnn_out_channels': cnn_out_channels,
            'early_stopping_patience': early_stopping_patience,
            'early_stopping_min_delta': early_stopping_min_delta,
            'actual_epochs_trained': len(train_losses),
        }
    )

    plot_training_curves(
        train_losses, val_losses, val_accs,
        save_path=plot_save_path
    )

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Total epochs trained: {len(train_losses)}")
    print(f"Best validation CER: {best_val_cer:.4f}")
    print(f"Best validation character accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {model_save_path}")
    print(f"Training curves saved to: {plot_save_path}")
    print("=" * 60)