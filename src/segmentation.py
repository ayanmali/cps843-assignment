"""
segmenting images into characters
"""
import cv2
import numpy as np
from typing import List, Tuple

def segment_characters(image: np.ndarray, 
                      min_width: int = 5, 
                      min_height: int = 10,
                      max_width_ratio: float = 0.5,
                      min_area: int = 20,
                      invert: bool = None) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    Segment characters from a binarized image using connected component analysis.
    
    Args:
        image: Binary image (numpy array). Can be text=black/background=white or text=white/background=black
        min_width: Minimum character width in pixels (filters noise)
        min_height: Minimum character height in pixels (filters noise)
        max_width_ratio: Maximum width/height ratio (filters horizontal lines/noise)
        min_area: Minimum character area in pixels (filters small artifacts)
        invert: If True, assumes text is white (255) and background is black (0).
                If False, assumes text is black (0) and background is white (255).
                If None (default), auto-detects based on which color has more pixels.
                Note: preprocess() outputs text=white (255), background=black (0), so use invert=True
    
    Returns:
        List of tuples: [(character_image, (x, y, w, h)), ...]
        where (x, y, w, h) is the bounding box of the character
        Characters are sorted left-to-right, top-to-bottom
    """
    # Ensure we're working with a grayscale/binary image
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Ensure binary format (0 or 255)
    if image.max() <= 1:
        image = (image * 255).astype(np.uint8)
    
    # Auto-detect if invert is None
    if invert is None:
        # Count white and black pixels to determine which is background
        white_pixels = np.sum(image == 255)
        black_pixels = np.sum(image == 0)
        # If more white pixels, likely background is white (text is black)
        # If more black pixels, likely background is black (text is white)
        invert = black_pixels > white_pixels
    
    # Invert if needed: for connected components, we want text to be white (255)
    if not invert:
        # If text is black and background is white, invert so text becomes white
        image = cv2.bitwise_not(image)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    
    characters = []
    height, width = image.shape
    
    for i in range(1, num_labels):  # Skip label 0 (background)
        x, y, w, h, area = stats[i]
        
        # Filter based on size constraints
        if w < min_width or h < min_height:
            continue
        
        if area < min_area:
            continue
        
        # Filter based on aspect ratio (remove horizontal lines/noise)
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio > max_width_ratio:
            continue
        
        # Filter components that are too large (likely not a single character)
        if w > width * 0.8 or h > height * 0.8:
            continue
        
        # Extract character bounding box
        # Add small padding to avoid cutting off parts of characters
        padding = 2
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(width, x + w + padding)
        y_end = min(height, y + h + padding)
        
        # Extract character region
        char_img = image[y_start:y_end, x_start:x_end].copy()
        
        # Remove fragments before inverting
        char_img = remove_fragments(char_img, keep_largest_only=True)
        
        # Invert back if original was inverted
        if not invert:
            char_img = cv2.bitwise_not(char_img)
        
        characters.append((char_img, (x_start, y_start, x_end - x_start, y_end - y_start)))
    
    # Sort characters left-to-right, then top-to-bottom
    characters.sort(key=lambda char: (char[1][1] // (height // 3), char[1][0]))  # Group by approximate row, then sort by x
    
    return characters


def segment_characters_projection(image: np.ndarray,
                                  min_char_width: int = 1,
                                  min_gap_width: int = 1,
                                  invert: bool = None) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    Alternative segmentation method using vertical projection profiling.
    Works well for well-separated characters with consistent spacing.
    Now includes splitting logic for merged characters.
    
    Args:
        image: Binary image
        min_char_width: Minimum character width in pixels
        min_gap_width: Minimum gap width between characters
        invert: If True, assumes text is white (255) and background is black (0).
                If False, assumes text is black (0) and background is white (255).
                If None, auto-detects.
    
    Returns:
        List of tuples: [(character_image, (x, y, w, h)), ...]
    """
    # Ensure binary format
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if image.max() <= 1:
        image = (image * 255).astype(np.uint8)
    
    # Auto-detect if invert is None
    if invert is None:
        white_pixels = np.sum(image == 255)
        black_pixels = np.sum(image == 0)
        invert = black_pixels > white_pixels
    
    # Invert if needed
    if not invert:
        image = cv2.bitwise_not(image)
    
    height, width = image.shape
    
    # Calculate vertical projection (sum of white pixels in each column)
    vertical_projection = np.sum(image, axis=0)
    
    # Find character boundaries
    # A column is part of a character if it has enough white pixels
    threshold = height * 0.1  # At least 10% of pixels in column should be white
    
    in_character = False
    char_start = 0
    characters = []
    
    for x in range(width):
        if vertical_projection[x] > threshold:
            if not in_character:
                # Start of a new character
                char_start = x
                in_character = True
        else:
            if in_character:
                # End of character
                char_width = x - char_start
                if char_width >= min_char_width:
                    # Extract character region
                    char_img = image[:, char_start:x].copy()
                    
                    # Find top and bottom bounds
                    horizontal_proj = np.sum(char_img, axis=1)
                    char_top = np.argmax(horizontal_proj > 0)
                    char_bottom = height - np.argmax(horizontal_proj[::-1] > 0) - 1
                    
                    if char_bottom > char_top:
                        char_img = char_img[char_top:char_bottom+1, :].copy()
                        
                        # Check if this might be multiple merged characters
                        # Estimate max single character width based on height
                        # Use more aggressive threshold - typical characters are narrower than tall
                        char_height = char_bottom - char_top + 1
                        max_single_char_width = char_height * 0.9  # Reduced from 1.2 to be more aggressive
                        
                        if char_width > max_single_char_width:
                            # Try to split merged characters
                            split_chars = _split_merged_characters(char_img, char_start, char_top, invert)
                            characters.extend(split_chars)
                        else:
                            # Single character - remove fragments before inverting
                            char_img = remove_fragments(char_img, keep_largest_only=True)
                            if not invert:
                                char_img = cv2.bitwise_not(char_img)
                            characters.append((char_img, (char_start, char_top, char_width, char_bottom - char_top + 1)))
                
                in_character = False
    
    # Handle case where image ends with a character
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
                    split_chars = _split_merged_characters(char_img, char_start, char_top, invert)
                    characters.extend(split_chars)
                else:
                    # Remove fragments before inverting
                    char_img = remove_fragments(char_img, keep_largest_only=True)
                    if not invert:
                        char_img = cv2.bitwise_not(char_img)
                    characters.append((char_img, (char_start, char_top, char_width, char_bottom - char_top + 1)))
    
    return characters


def _split_merged_characters(char_img: np.ndarray, base_x: int, base_y: int, 
                             invert: bool, min_char_width: int = 5) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    Split a merged character region into individual characters.
    Uses multiple methods: vertical projection analysis, derivative-based detection,
    and connected components within the region.
    
    Args:
        char_img: Image region containing potentially merged characters
        base_x: X coordinate of the region in original image
        base_y: Y coordinate of the region in original image
        invert: Whether to invert the output
        min_char_width: Minimum width for a valid character
    
    Returns:
        List of individual character segments
    """
    height, width = char_img.shape
    characters = []
    
    # Calculate vertical projection for this region
    vertical_proj = np.sum(char_img, axis=0)
    
    # Normalize projection to 0-1 range for easier thresholding
    max_proj = np.max(vertical_proj)
    if max_proj == 0:
        return characters
    
    normalized_proj = vertical_proj / max_proj
    
    # Method 1: Use derivative to find sharp drops (better for close characters)
    # Calculate first derivative (difference between adjacent columns)
    derivative = np.diff(normalized_proj)
    
    # Find points where projection drops significantly (negative derivative peaks)
    # These indicate transitions from character to gap
    mean_derivative = np.mean(np.abs(derivative))
    split_points = []
    
    # Look for significant negative derivatives (drops)
    for i in range(1, len(derivative) - 1):
        # Check for significant drop (negative peak)
        if derivative[i] < -mean_derivative * 0.5:
            # Check if this is a local minimum in the derivative
            if derivative[i] < derivative[i-1] and derivative[i] < derivative[i+1]:
                # The split point is at the column where the drop occurs
                split_col = i + 1  # +1 because derivative is offset
                # Verify this column has low projection
                if normalized_proj[split_col] < np.mean(normalized_proj) * 0.6:
                    split_points.append(split_col)
    
    # Method 2: Find gaps using thresholding (for characters with clear separation)
    if len(split_points) == 0:
        mean_proj = np.mean(normalized_proj)
        gap_threshold = mean_proj * 0.4  # More lenient threshold
        
        # Find continuous gaps
        gaps = []
        gap_start = None
        
        for i in range(width):
            if normalized_proj[i] < gap_threshold:
                if gap_start is None:
                    gap_start = i
            else:
                if gap_start is not None:
                    gap_width = i - gap_start
                    if gap_width >= 1:  # Even single-pixel gaps can be valid
                        # Use the center of the gap
                        gaps.append(gap_start + gap_width // 2)
                    gap_start = None
        
        # Handle gap at the end
        if gap_start is not None:
            gap_width = width - gap_start
            if gap_width >= 1:
                gaps.append(gap_start + gap_width // 2)
        
        split_points = gaps
    
    # Method 3: Use connected components within the region (most robust)
    if len(split_points) == 0:
        # Find connected components in this region
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(char_img, connectivity=8)
        
        if num_labels > 2:  # More than background + 1 component means multiple characters
            # Extract x-coordinates of component boundaries
            component_xs = []
            for i in range(1, num_labels):
                x, _, w, _, _ = stats[i]
                # Add left and right boundaries (with some margin)
                component_xs.append(x)
                component_xs.append(x + w)
            
            # Sort and use midpoints between components as split points
            component_xs = sorted(set(component_xs))
            for i in range(len(component_xs) - 1):
                midpoint = (component_xs[i] + component_xs[i + 1]) // 2
                # Only add if there's a reasonable gap
                if component_xs[i + 1] - component_xs[i] > min_char_width:
                    split_points.append(midpoint)
    
    # Method 4: Estimate based on aspect ratio (fallback)
    if len(split_points) == 0:
        char_height = height
        # More conservative estimate - typical character is narrower
        estimated_char_width = int(char_height * 0.5)  # Reduced from 0.6
        
        if width > estimated_char_width * 1.3:  # More aggressive threshold
            # Estimate number of characters
            num_chars = max(2, int(round(width / estimated_char_width)))
            split_width = width // num_chars
            # Add split points at estimated boundaries
            split_points = [split_width * i for i in range(1, num_chars)]
    
    # Remove duplicate and invalid split points
    split_points = sorted(set([sp for sp in split_points if min_char_width <= sp <= width - min_char_width]))
    
    # Split the image at split points
    if len(split_points) > 0:
        # Add start and end
        split_points = [0] + split_points + [width]
        
        # Remove duplicates while preserving order
        seen = set()
        split_points = [x for x in split_points if not (x in seen or seen.add(x))]
        
        # Extract individual characters
        for i in range(len(split_points) - 1):
            x_start = split_points[i]
            x_end = split_points[i + 1]
            char_width = x_end - x_start
            
            if char_width >= min_char_width:
                # Extract character
                single_char = char_img[:, x_start:x_end].copy()
                
                # Trim top and bottom
                horizontal_proj = np.sum(single_char, axis=1)
                if np.any(horizontal_proj > 0):
                    char_top = np.argmax(horizontal_proj > 0)
                    char_bottom = height - np.argmax(horizontal_proj[::-1] > 0) - 1
                    
                    if char_bottom > char_top:
                        single_char = single_char[char_top:char_bottom+1, :].copy()
                        
                        # Remove fragments before inverting
                        single_char = remove_fragments(single_char, keep_largest_only=True)
                        
                        # Invert if needed
                        if not invert:
                            single_char = cv2.bitwise_not(single_char)
                        
                        characters.append((single_char, (base_x + x_start, base_y + char_top, 
                                                       char_width, char_bottom - char_top + 1)))
    else:
        # Couldn't split, return as single character
        # Remove fragments before inverting
        char_img = remove_fragments(char_img, keep_largest_only=True)
        if not invert:
            char_img = cv2.bitwise_not(char_img)
        characters.append((char_img, (base_x, base_y, width, height)))
    
    return characters

"""
TODO: see if keep_largest_only needs to be adjusted; add size-based filtering if necessary
"""
def remove_fragments(char_img: np.ndarray, 
                     min_fragment_area_ratio: float = 0.1,
                     keep_largest_only: bool = True) -> np.ndarray:
    """
    Remove small fragments from a character segment by keeping only the largest connected component(s).
    This helps remove remnant pieces from adjacent letters that got included in the bounding box.
    
    Args:
        char_img: Binary character image (text=white/255, background=black/0)
        min_fragment_area_ratio: Minimum area ratio (relative to largest component) to keep a fragment.
                                 Fragments smaller than this ratio will be removed.
        keep_largest_only: If True, keep only the largest component. If False, keep all components
                          above the min_fragment_area_ratio threshold.
    
    Returns:
        Cleaned character image with fragments removed
    """
    if char_img is None or char_img.size == 0:
        return char_img
    
    # Ensure binary format
    if len(char_img.shape) == 3:
        char_img = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
    
    if char_img.max() <= 1:
        char_img = (char_img * 255).astype(np.uint8)
    
    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(char_img, connectivity=8)
    
    if num_labels <= 1:  # Only background, no components
        return char_img
    
    # Get areas of all components (excluding background)
    # Stats format: [x, y, width, height, area] - area is at index 4
    areas = stats[1:, 4]  # Skip label 0 (background), get area column
    
    if len(areas) == 0:
        return char_img
    
    # Find the largest component
    largest_area = np.max(areas)
    largest_idx = np.argmax(areas) + 1  # +1 because we skipped label 0
    
    # Create mask for components to keep
    mask = np.zeros_like(labels, dtype=np.uint8)
    
    if keep_largest_only:
        # Keep only the largest component
        mask[labels == largest_idx] = 255
    else:
        # Keep all components above the threshold
        min_area = largest_area * min_fragment_area_ratio
        for i in range(1, num_labels):
            if stats[i, 4] >= min_area:  # Area is at index 4
                mask[labels == i] = 255
    
    # Apply mask to original image
    cleaned_img = cv2.bitwise_and(char_img, mask)
    
    return cleaned_img


def segment_characters_hybrid(image: np.ndarray,
                              min_width: int = 10,
                              min_height: int = 10,
                              max_width_ratio: float = 5,
                              min_area: int = 10,
                              invert: bool = None) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    Hybrid segmentation method combining connected components with projection-based splitting.
    Best for images with characters that may be close together.
    
    Args:
        image: Binary image
        min_width: Minimum character width in pixels
        min_height: Minimum character height in pixels
        max_width_ratio: Maximum width/height ratio (filters horizontal lines)
        min_area: Minimum character area in pixels
        invert: If None, auto-detects. If True, text is white. If False, text is black.
    
    Returns:
        List of tuples: [(character_image, (x, y, w, h)), ...]
    """
    # Ensure binary format
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if image.max() <= 1:
        image = (image * 255).astype(np.uint8)
    
    # Auto-detect if invert is None
    if invert is None:
        white_pixels = np.sum(image == 255)
        black_pixels = np.sum(image == 0)
        invert = black_pixels > white_pixels
    
    # Invert if needed
    working_image = image.copy()
    if not invert:
        working_image = cv2.bitwise_not(working_image)
    
    height, width = working_image.shape
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(working_image, connectivity=8)
    
    characters = []
    
    for i in range(1, num_labels):  # Skip label 0 (background)
        x, y, w, h, area = stats[i]
        
        # Filter based on size constraints
        if w < min_width or h < min_height:
            continue
        
        if area < min_area:
            continue
        
        # Filter based on aspect ratio
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio > max_width_ratio:
            continue
        
        # Filter components that are too large
        if w > width * 0.8 or h > height * 0.8:
            continue
        
        # Extract character region
        padding = 2
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(width, x + w + padding)
        y_end = min(height, y + h + padding)
        
        char_img = working_image[y_start:y_end, x_start:x_end].copy()
        
        # Check if this might be multiple merged characters
        # Estimate max single character width based on height
        char_height = y_end - y_start
        max_single_char_width = char_height * 0.9  # More aggressive threshold for better splitting
        
        char_width = x_end - x_start
        
        if char_width > max_single_char_width:
            # Try to split merged characters
            split_chars = _split_merged_characters(char_img, x_start, y_start, invert, min_width)
            characters.extend(split_chars)
        else:
            # Single character - remove fragments before inverting
            char_img = remove_fragments(char_img, keep_largest_only=True)
            if not invert:
                char_img = cv2.bitwise_not(char_img)
            characters.append((char_img, (x_start, y_start, char_width, char_height)))
    
    # Sort characters left-to-right, then top-to-bottom
    characters.sort(key=lambda char: (char[1][1] // (height // 3), char[1][0]))
    
    return characters


def visualize_segmentation(image: np.ndarray, characters: List[Tuple[np.ndarray, Tuple[int, int, int, int]]], 
                           output_path: str = "segmentation.png") -> None:
    """
    Visualize character segmentation by drawing bounding boxes on the image.
    
    Args:
        image: Original image
        characters: List of segmented characters from segment_characters()
        output_path: Path to save visualization
    """
    # Create a copy for visualization
    if len(image.shape) == 2:
        vis_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis_img = image.copy()
    
    # Draw bounding boxes
    for i, (char_img, (x, y, w, h)) in enumerate(characters):
        cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(vis_img, str(i), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    cv2.imwrite(output_path, vis_img)