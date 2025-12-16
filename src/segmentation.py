"""
segmenting images into characters
"""
import cv2
import numpy as np
from typing import List, Tuple

def segment_characters_projection(image: np.ndarray,
                                  min_char_width: int = 2,
                                  min_gap_width: int = 1) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if image.max() <= 1:
        image = (image * 255).astype(np.uint8)
    
    white_pixels = np.sum(image == 255)
    black_pixels = np.sum(image == 0)
    invert = black_pixels > white_pixels
    
    if not invert:
        image = cv2.bitwise_not(image)
    
    height, width = image.shape
    
    vertical_projection = np.sum(image, axis=0)
    
    threshold = height * 0.1  # at least 10% of pixels in column should be white
    
    in_character = False
    char_start = 0
    characters = []
    
    for x in range(width):
        if vertical_projection[x] > threshold:
            if not in_character:
                char_start = x
                in_character = True
        else:
            if in_character:
                char_width = x - char_start
                if char_width >= min_char_width:
                    char_img = image[:, char_start:x].copy()
                    
                    horizontal_proj = np.sum(char_img, axis=1)
                    char_top = np.argmax(horizontal_proj > 0)
                    char_bottom = height - np.argmax(horizontal_proj[::-1] > 0) - 1
                    
                    if char_bottom > char_top:
                        char_img = char_img[char_top:char_bottom+1, :].copy()
                        
                        char_height = char_bottom - char_top + 1
                        max_single_char_width = char_height * 0.9  
                        
                        if char_width > max_single_char_width:
                            split_chars = split_merged_characters(char_img, char_start, char_top, invert)
                            characters.extend(split_chars)
                        else:
                            char_img = remove_fragments(char_img, keep_largest_only=True)
                            if not invert:
                                char_img = cv2.bitwise_not(char_img)
                            characters.append((char_img, (char_start, char_top, char_width, char_bottom - char_top + 1)))
                
                in_character = False
    
    if in_character:
        char_width = width - char_start
        if char_width >= min_char_width:
            char_img = image[:, char_start:].copy()
            horizontal_proj = np.sum(char_img, axis=1)
            char_top = np.argmax(horizontal_proj > 0)
            char_bottom = height - np.argmax(horizontal_proj[::-1] > 0) - 1
            
            if char_bottom > char_top:
                char_img = char_img[char_top:char_bottom+1, :].copy()
                char_height = char_bottom - char_top + 1
                max_single_char_width = char_height * 0.9  # More aggressive threshold
                
                if char_width > max_single_char_width:
                    split_chars = split_merged_characters(char_img, char_start, char_top, invert)
                    characters.extend(split_chars)
                else:
                    char_img = remove_fragments(char_img, keep_largest_only=True)
                    if not invert:
                        char_img = cv2.bitwise_not(char_img)
                    characters.append((char_img, (char_start, char_top, char_width, char_bottom - char_top + 1)))
    
    return characters


def split_merged_characters(char_img: np.ndarray, base_x: int, base_y: int, invert: bool, min_char_width: int = 5):
    height, width = char_img.shape
    characters = []
    
    vertical_proj = np.sum(char_img, axis=0)
    
    max_proj = np.max(vertical_proj)
    if max_proj == 0:
        return characters
    
    normalized_proj = vertical_proj / max_proj
    
    derivative = np.diff(normalized_proj)
    
    mean_derivative = np.mean(np.abs(derivative))
    split_points = []
    
    for i in range(1, len(derivative) - 1):
        if derivative[i] < -mean_derivative * 0.5:
            if derivative[i] < derivative[i-1] and derivative[i] < derivative[i+1]:
                split_col = i + 1  
                if normalized_proj[split_col] < np.mean(normalized_proj) * 0.6:
                    split_points.append(split_col)

    if len(split_points) == 0:
        mean_proj = np.mean(normalized_proj)
        gap_threshold = mean_proj * 0.4  
        
        gaps = []
        gap_start = None
        
        for i in range(width):
            if normalized_proj[i] < gap_threshold:
                if gap_start is None:
                    gap_start = i
            else:
                if gap_start is not None:
                    gap_width = i - gap_start
                    if gap_width >= 1:  
                        gaps.append(gap_start + gap_width // 2)
                    gap_start = None
        
        if gap_start is not None:
            gap_width = width - gap_start
            if gap_width >= 1:
                gaps.append(gap_start + gap_width // 2)
        
        split_points = gaps
    
    if len(split_points) == 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(char_img, connectivity=8)
        
        if num_labels > 2:  
            component_xs = []
            for i in range(1, num_labels):
                x, _, w, _, _ = stats[i]
                component_xs.append(x)
                component_xs.append(x + w)
            
            component_xs = sorted(set(component_xs))
            for i in range(len(component_xs) - 1):
                midpoint = (component_xs[i] + component_xs[i + 1]) // 2
                if component_xs[i + 1] - component_xs[i] > min_char_width:
                    split_points.append(midpoint)
    
    if len(split_points) == 0:
        char_height = height
        estimated_char_width = int(char_height * 0.5)  
        
        if width > estimated_char_width * 1.3:  
            num_chars = max(2, int(round(width / estimated_char_width)))
            split_width = width // num_chars
            split_points = [split_width * i for i in range(1, num_chars)]
    
    split_points = sorted(set([sp for sp in split_points if min_char_width <= sp <= width - min_char_width]))
    
    if len(split_points) > 0:
        split_points = [0] + split_points + [width]
        
        seen = set()
        split_points = [x for x in split_points if not (x in seen or seen.add(x))]
        
        # Extract individual characters
        for i in range(len(split_points) - 1):
            x_start = split_points[i]
            x_end = split_points[i + 1]
            char_width = x_end - x_start
            
            if char_width >= min_char_width:
                single_char = char_img[:, x_start:x_end].copy()
                
                horizontal_proj = np.sum(single_char, axis=1)
                if np.any(horizontal_proj > 0):
                    char_top = np.argmax(horizontal_proj > 0)
                    char_bottom = height - np.argmax(horizontal_proj[::-1] > 0) - 1
                    
                    if char_bottom > char_top:
                        single_char = single_char[char_top:char_bottom+1, :].copy()
                        
                        single_char = remove_fragments(single_char, keep_largest_only=True)
                        
                        if not invert:
                            single_char = cv2.bitwise_not(single_char)
                        
                        characters.append((single_char, (base_x + x_start, base_y + char_top, 
                                                       char_width, char_bottom - char_top + 1)))
    else:
        char_img = remove_fragments(char_img, keep_largest_only=True)
        if not invert:
            char_img = cv2.bitwise_not(char_img)
        characters.append((char_img, (base_x, base_y, width, height)))
    
    return characters


def remove_fragments(char_img: np.ndarray, min_fragment_area_ratio: float = 0.1, keep_largest_only: bool = True) -> np.ndarray:
    if char_img is None or char_img.size == 0:
        return char_img
    
    if len(char_img.shape) == 3:
        char_img = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
    
    if char_img.max() <= 1:
        char_img = (char_img * 255).astype(np.uint8)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(char_img, connectivity=8)
    
    if num_labels <= 1:  # only background, no components
        return char_img
    
    areas = stats[1:, 4]
    
    if len(areas) == 0:
        return char_img
    
    largest_area = np.max(areas)
    largest_idx = np.argmax(areas) + 1
    
    mask = np.zeros_like(labels, dtype=np.uint8)
    
    if keep_largest_only:
        mask[labels == largest_idx] = 255
    else:
        min_area = largest_area * min_fragment_area_ratio
        for i in range(1, num_labels):
            if stats[i, 4] >= min_area:
                mask[labels == i] = 255
    
    cleaned_img = cv2.bitwise_and(char_img, mask)
    
    return cleaned_img

# draws bounding boxes on the image to see the segmentation
def visualize_segmentation(image: np.ndarray, characters, output_path: str = "segmentation.png") -> None:
    if len(image.shape) == 2:
        vis_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis_img = image.copy()
    
    for i, (char_img, (x, y, w, h)) in enumerate(characters):
        cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(vis_img, str(i), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    cv2.imwrite(output_path, vis_img)