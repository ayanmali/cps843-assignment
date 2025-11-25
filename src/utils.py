"""
utility functions
"""
import cv2
import numpy as np
from src.preprocessing import compress

def save_img(image: np.ndarray, path: str) -> None:
    """
    save image to file path
    """
    cv2.imwrite(path, image)

def visualize_preprocessing(image: np.ndarray, output_path: str = "preprocessing_comparison.png") -> None:
    """
    Create a side-by-side comparison of original and processed images
    """
    # Process the image
    processed = compress(image)
    
    # Convert BGR to RGB for display (OpenCV uses BGR, matplotlib uses RGB)
    original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create a side-by-side comparison
    height = max(image.shape[0], processed.shape[0])
    width = image.shape[1] + processed.shape[1]
    
    # Create a 3-channel image for the comparison (for color original)
    comparison = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Place original on the left
    comparison[:image.shape[0], :image.shape[1], :] = original_rgb
    
    # Place processed on the right (convert grayscale to RGB)
    processed_rgb = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
    comparison[:processed.shape[0], image.shape[1]:, :] = processed_rgb
    
    # Save the comparison
    cv2.imwrite(output_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    print(f"Comparison saved to {output_path}")
