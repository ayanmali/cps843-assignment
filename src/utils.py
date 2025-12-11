"""
utility functions
"""
import cv2
import numpy as np
from typing import Tuple

def save_img(image: np.ndarray, path: str) -> None:
    """
    save image to file path
    """
    cv2.imwrite(path, image)

def resize_to_fixed_size(image: np.ndarray, target_size: Tuple[int, int], 
                         maintain_aspect: bool = True, pad_color: int = 0) -> np.ndarray:
    """
    Resize a character image to a fixed size while maintaining aspect ratio.
    
    Args:
        image: Input character image (numpy array, can be grayscale or binary)
        target_size: Target size as (height, width) tuple (default: (28, 28))
        maintain_aspect: If True, maintains aspect ratio and pads with pad_color.
                        If False, stretches image to fit target_size.
        pad_color: Color to use for padding (default: 0 for black background)
    
    Returns:
        Resized image of shape target_size
    """
    # Ensure we're working with grayscale/binary
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Ensure binary format (0 or 255)
    if image.max() <= 1:
        image = (image * 255).astype(np.uint8)
    
    target_h, target_w = target_size
    h, w = image.shape
    
    if maintain_aspect:
        # Calculate scaling factor to fit image within target size
        scale = min(target_h / h, target_w / w)
        
        # Resize while maintaining aspect ratio
        new_h = int(h * scale)
        new_w = int(w * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create a blank canvas with pad_color
        output = np.full((target_h, target_w), pad_color, dtype=np.uint8)
        
        # Calculate position to center the resized image
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        
        # Place resized image in the center
        output[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    else:
        # Stretch image to fit target size (may distort aspect ratio)
        output = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
    
    return output

def visualize_comparison(original_image: np.ndarray, processed_image: np.ndarray, output_path: str = "comparison.png") -> None:
    """
    Create a side-by-side comparison of original and processed images
    """
    # Create a 3-channel image for the comparison (for color original)
    comparison = np.zeros((original_image.shape[0], original_image.shape[1] + processed_image.shape[1], 3), dtype=np.uint8)
    
    # Place original on the left
    comparison[:original_image.shape[0], :original_image.shape[1], :] = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Place processed on the right (convert grayscale to RGB)
    processed_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    comparison[:processed_image.shape[0], original_image.shape[1]:, :] = processed_rgb
    
    # Save the comparison
    cv2.imwrite(output_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

# def visualize_preprocessing(image: np.ndarray, output_path: str = "preprocessing_comparison.png") -> None:
#     """
#     Create a side-by-side comparison of original and processed images
#     """
#     # Process the image
#     processed = compress(image)
    
#     # Convert BGR to RGB for display (OpenCV uses BGR, matplotlib uses RGB)
#     original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
#     # Create a side-by-side comparison
#     height = max(image.shape[0], processed.shape[0])
#     width = image.shape[1] + processed.shape[1]
    
#     # Create a 3-channel image for the comparison (for color original)
#     comparison = np.zeros((height, width, 3), dtype=np.uint8)
    
#     # Place original on the left
#     comparison[:image.shape[0], :image.shape[1], :] = original_rgb
    
#     # Place processed on the right (convert grayscale to RGB)
#     processed_rgb = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
#     comparison[:processed.shape[0], image.shape[1]:, :] = processed_rgb
    
#     # Save the comparison
#     cv2.imwrite(output_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
#     print(f"Comparison saved to {output_path}")
