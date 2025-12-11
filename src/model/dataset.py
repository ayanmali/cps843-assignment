"""
PyTorch Dataset for character recognition
Handles loading, preprocessing, segmentation, and character extraction
"""
from main import TARGET_SIZE
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
from typing import Tuple, List
import os

from src.preprocessing import load_img, preprocess, apply_morph, adjust_skew_hough
from src.segmentation import segment_characters_hybrid
from src.utils import resize_to_fixed_size
from src.labels import LABELS

class CharacterDataset(Dataset):
    """
    Dataset for character recognition.
    
    For each image:
    1. Loads and preprocesses the image
    2. Segments characters from the image
    3. Extracts each character individually
    4. Maps each character to its label from GroundTruth
    5. Returns character images and labels
    """
    def __init__(
        self,
        csv_path: str = None,
        df: pd.DataFrame = None,
        base_dir: str = "iiit-5k/IIIT5K-Word_V3.0/IIIT5K",
        target_size: Tuple[int, int] = TARGET_SIZE,
        transform=None,
        image_indices: List[int] = None
    ):
        """
        Args:
            csv_path: Path to CSV file with columns: ImgName, GroundTruth (if df not provided)
            df: DataFrame with image data (alternative to csv_path)
            base_dir: Base directory where images are stored
            target_size: Target size for character images (height, width)
            transform: Optional torchvision transforms to apply
            image_indices: Optional list of image indices to use (for K-fold splits)
        """
        if df is not None:
            self.df = df.copy()
        elif csv_path is not None:
            self.df = pd.read_csv(csv_path)
        else:
            raise ValueError("Either csv_path or df must be provided")
        
        # Filter by image_indices if provided
        if image_indices is not None:
            self.df = self.df.iloc[image_indices].reset_index(drop=True)
        self.base_dir = base_dir
        self.target_size = target_size
        self.transform = transform
        
        # Create label to index mapping
        self.label_to_idx = {char: idx for idx, char in enumerate(LABELS)}
        self.idx_to_label = {idx: char for char, idx in self.label_to_idx.items()}
        self.num_classes = len(LABELS)
        
        # Pre-process all images and extract characters
        # This creates a flat list of (character_image, label_idx) pairs
        self.characters = []
        self._process_all_images()
    
    def _process_all_images(self):
        """
        Process all images: load, preprocess, segment, and extract characters.
        Stores results in self.characters as a list of (image, label_idx) tuples.
        """
        print(f"Processing {len(self.df)} images...")
        
        for idx, row in self.df.iterrows():
            img_path = os.path.join(self.base_dir, row['ImgName'])
            ground_truth = row['GroundTruth']
            
            # Load image
            img = load_img(img_path)
            if img is None:
                print(f"Warning: Could not load image {img_path}, skipping...")
                continue
            
            # Preprocess image
            processed_img = preprocess(img)
            processed_img = apply_morph(processed_img)
            processed_img = adjust_skew_hough(processed_img)
            
            # Segment characters
            characters = segment_characters_hybrid(processed_img)
            
            # Extract each character and match with ground truth
            # GroundTruth is a string like "THA", "HOME", etc.
            gt_chars = list(ground_truth)
            
            # Match segmented characters with ground truth characters
            # Simple approach: assume characters are in order
            num_chars = min(len(characters), len(gt_chars))
            
            for i in range(num_chars):
                char_img, bbox = characters[i]
                gt_char = gt_chars[i]
                
                # Skip if character is not in our label set
                if gt_char not in self.label_to_idx:
                    continue
                
                # Resize character to target size
                # Note: preprocess outputs text=white (255), bg=black (0)
                # For model input, we might want to normalize differently
                resized_char = resize_to_fixed_size(
                    char_img, 
                    target_size=self.target_size, 
                    maintain_aspect=True,
                )
                
                # Convert to float and normalize to [0, 1]
                # Model expects: text=white (1.0), background=black (0.0)
                char_normalized = resized_char.astype(np.float32) / 255.0
                
                # Get label index
                label_idx = self.label_to_idx[gt_char]
                
                self.characters.append((char_normalized, label_idx))
            
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(self.df)} images, "
                      f"extracted {len(self.characters)} characters so far...")
        
        print(f"Total characters extracted: {len(self.characters)}")
    
    def __len__(self):
        return len(self.characters)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: Tensor of shape (1, H, W) - grayscale character image
            label: Integer label index
        """
        char_img, label_idx = self.characters[idx]
        
        # Convert numpy array to tensor
        # Add channel dimension: (H, W) -> (1, H, W)
        image_tensor = torch.from_numpy(char_img).unsqueeze(0)
        
        # Apply transforms if provided
        if self.transform:
            image_tensor = self.transform(image_tensor)
        
        return image_tensor, label_idx


def create_character_dataset(
    csv_path: str,
    base_dir: str = "iiit-5k/IIIT5K-Word_V3.0/IIIT5K",
    target_size: Tuple[int, int] = TARGET_SIZE,
    transform=None
) -> CharacterDataset:
    """
    Factory function to create a CharacterDataset.
    
    Args:
        csv_path: Path to CSV file
        base_dir: Base directory for images
        target_size: Target size for character images
        transform: Optional transforms
    
    Returns:
        CharacterDataset instance
    """
    return CharacterDataset(
        csv_path=csv_path,
        base_dir=base_dir,
        target_size=target_size,
        transform=transform
    )

