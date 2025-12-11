import cv2
from src.preprocessing import adjust_skew_hough, apply_morph, correct_skew1, correct_slant, detect_and_fill, detect_edges, fill_edges, load_img, preprocess
# Alternative: remove_noise_connected_components() for more precise but slower noise removal
from src.utils import save_img, visualize_comparison, resize_to_fixed_size
# Import all segmentation methods - try different ones if hybrid doesn't work well
from src.segmentation import segment_characters, segment_characters_hybrid, segment_characters_projection, visualize_segmentation
import numpy as np
from typing import List, Tuple

TARGET_SIZE = (28, 28)

def split_wide_segments_rec(
    characters: List[Tuple[np.ndarray, Tuple[int, int, int, int]]],
    original_image_width: int,
    width_threshold: float = 0.2,
    max_depth: int = 5
) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    Recursively checks the width of each identified segment. For segments wider than
    the threshold (default 20% of original image width), applies the segmentation
    algorithm again on that segment to obtain the characters within it.
    
    Args:
        characters: List of tuples (character_image, (x, y, w, h))
        original_image_width: Width of the original image in pixels
        width_threshold: Threshold ratio (default: 0.2 = 20%)
        max_depth: Maximum recursion depth to prevent infinite loops (default: 5)
    
    Returns:
        Updated list of character segments with wide segments recursively segmented
    """
    if max_depth <= 0:
        # Base case: reached max depth, return segments as-is
        return characters
    
    threshold_width = int(original_image_width * width_threshold)
    updated_characters = []
    
    for char_img, bbox in characters:
        x, y, w, h = bbox
        
        # Check if segment width is >= threshold
        if w >= threshold_width:
            # Apply segmentation algorithm to this segment
            # The segmentation function expects a binary image
            sub_segments = segment_characters(char_img)
            
            if len(sub_segments) > 1:
                # Successfully segmented into multiple characters
                # Adjust bounding boxes to be relative to the original image
                adjusted_segments = []
                for sub_char_img, (sub_x, sub_y, sub_w, sub_h) in sub_segments:
                    # Adjust coordinates: sub_x and sub_y are relative to the char_img
                    adjusted_bbox = (x + sub_x, y + sub_y, sub_w, sub_h)
                    adjusted_segments.append((sub_char_img, adjusted_bbox))
                
                # Recursively process the sub-segments
                recursively_processed = split_wide_segments_rec(
                    adjusted_segments,
                    original_image_width,
                    width_threshold,
                    max_depth - 1
                )
                updated_characters.extend(recursively_processed)
                
                print(f"Recursively segmented segment at ({x}, {y}) with width {w} "
                      f"into {len(recursively_processed)} sub-segments")
            else:
                # Segmentation didn't split it further, keep original
                updated_characters.append((char_img, bbox))
        else:
            # Keep original segment if it's not wide enough
            updated_characters.append((char_img, bbox))
    
    return updated_characters

def split_wide_segments(characters: List[Tuple[np.ndarray, Tuple[int, int, int, int]]], 
                        original_image_width: int,
                        width_threshold: float = 0.2) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    Check the width of each identified segment. For segments wider than the threshold
    (default 20% of original image width), split them vertically down the middle.
    
    Args:
        characters: List of tuples (character_image, (x, y, w, h))
        original_image_width: Width of the original image in pixels
        width_threshold: Threshold ratio (default: 0.2 = 20%)
    
    Returns:
        Updated list of character segments with wide segments split
    """
    threshold_width = int(original_image_width * width_threshold)
    updated_characters = []
    
    for char_img, bbox in characters:
        x, y, w, h = bbox
        
        # Check if segment width is >= threshold
        if w >= threshold_width:
            # Split vertically down the middle
            split_point = w // 2
            
            # Extract left half
            left_half = char_img[:, :split_point].copy()
            left_bbox = (x, y, split_point, h)
            
            # Extract right half
            right_half = char_img[:, split_point:].copy()
            right_bbox = (x + split_point, y, w - split_point, h)
            
            # Add both halves to the updated list
            updated_characters.append((left_half, left_bbox))
            updated_characters.append((right_half, right_bbox))
            
            print(f"Split segment at ({x}, {y}) with width {w} into two segments: "
                  f"left ({split_point}px) and right ({w - split_point}px)")
        else:
            # Keep original segment if it's not wide enough
            updated_characters.append((char_img, bbox))
    
    return updated_characters

def main():
    # testing image loading
    # images to test: 32_2.png, 135_5.png, 73_5.png
    img_path = "iiit-5k/IIIT5K-Word_V3.0/IIIT5K/train/73_5.png"
    img = load_img(img_path)
    
    if img is None:
        print(f"Error: Could not load image from {img_path}")
        return
    
    # Process the image
    processed_img = preprocess(img)
    #processed_img = apply_morph(processed_img)
    #_, processed_img = correct_skew1(processed_img, limit=20, delta=0.1)
    processed_img = adjust_skew_hough(processed_img)
    processed_img = correct_slant(processed_img)
    # TODO: see if this is needed
    detected_edges = detect_edges(processed_img)
    detected_edges = fill_edges(detected_edges)
    #processed_img = detect_and_fill(processed_img)

    #visualize_comparison(img, filled_edges, "filled_edges.png")
    #print("Filled edges image saved to filled_edges.png")
    
    # Save the processed image
    save_img(processed_img, "processed.png")
    print("Processed image saved to processed.png")
    
    # Create a side-by-side comparison
    #visualize_preprocessing(img, "preprocessing_comparison.png")
    visualize_comparison(img, processed_img, "comparison.png")
    print("Comparison image saved to comparison.png")
    
    # Segment characters
    # Using hybrid method which combines connected components with projection-based splitting
    # This works better for characters that are close together (like "THA")
    # Alternatives:
    #   - segment_characters(): Uses connected components only (good for well-separated chars)
    #   - segment_characters_projection(): Uses projection profiling (now with splitting support)
    #   - segment_characters_hybrid(): Combines both methods (recommended for close characters)
    
    #rocessed_img = cv2.bitwise_not(processed_img)
    characters = segment_characters_projection(processed_img)
    # print(f"Found {len(characters)} characters after initial segmentation")
    
    # # Split wide segments (>= 20% of original image width)
    # original_width = processed_img.shape[1]
    # characters = split_wide_segments(characters, original_width, width_threshold=0.3)
    # print(f"Found {len(characters)} characters after splitting wide segments")
    
    # Visualize segmentation
    visualize_segmentation(processed_img, characters, "segmentation.png")
    print("Segmentation visualization saved to segmentation.png")
    
    # Save individual character images
    # Resize each character to fixed size (28x28) for CNN training
    for i, (char_img, bbox) in enumerate(characters):
        # Resize to fixed size while maintaining aspect ratio
        resized_char = resize_to_fixed_size(char_img, target_size=TARGET_SIZE, maintain_aspect=True)
        save_img(resized_char, f"char_{i:02d}.png")
        print(f"Original shape: {char_img.shape}, Resized shape: {resized_char.shape}")
    print(f"Saved {len(characters)} individual character images (resized to {TARGET_SIZE})")

if __name__ == "__main__":
    main()
