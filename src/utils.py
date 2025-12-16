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

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if image.max() <= 1:
        image = (image * 255).astype(np.uint8)
    
    target_h, target_w = target_size
    h, w = image.shape
    
    if maintain_aspect:
        scale = min(target_h / h, target_w / w)
        
        new_h = int(h * scale)
        new_w = max(1, int(w * scale))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        output = np.full((target_h, target_w), pad_color, dtype=np.uint8)
        
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        
        output[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    else:
        output = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
    
    return output

def visualize_comparison(original_image: np.ndarray, processed_image: np.ndarray, output_path: str = "comparison.png"):
    comparison = np.zeros((original_image.shape[0], original_image.shape[1] + processed_image.shape[1], 3), dtype=np.uint8)
    
    comparison[:original_image.shape[0], :original_image.shape[1], :] = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    processed_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    comparison[:processed_image.shape[0], original_image.shape[1]:, :] = processed_rgb
    
    cv2.imwrite(output_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))