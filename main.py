import cv2
from src.preprocessing import adjust_skew_hough, correct_slant, load_img, preprocess, threshold
# Alternative: remove_noise_connected_components() for more precise but slower noise removal
from src.utils import save_img, visualize_comparison, resize_to_fixed_size
# Import all segmentation methods - try different ones if hybrid doesn't work well
from src.segmentation import segment_characters_projection, visualize_segmentation
import numpy as np
from typing import List, Tuple

TARGET_SIZE = (28, 28)

def remove_fragments(char_img: np.ndarray) -> np.ndarray:
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
    
    # Connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(char_img, connectivity=8)

    # Index of largest component (excluding background=0)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    # Create output mask
    clean = np.zeros_like(char_img)
    clean[labels == largest_label] = 255
    
    return clean

def extract_segments_from_image(
    source_image: np.ndarray,
    bounding_boxes: List[Tuple[int, int, int, int]]
) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    Extract character segments from source_image using bounding boxes.
    
    Args:
        source_image: The image to extract segments from
        bounding_boxes: List of bounding boxes (x, y, w, h) from segmentation
    
    Returns:
        List of tuples (character_image, bbox) extracted from source_image
    """
    segments = []
    img_height, img_width = source_image.shape[:2]
    
    for x, y, w, h in bounding_boxes:
        # Ensure bounding box is within image bounds
        x_start = max(0, x)
        y_start = max(0, y)
        x_end = min(img_width, x + w)
        y_end = min(img_height, y + h)
        
        # Extract the segment
        char_segment = source_image[y_start:y_end, x_start:x_end].copy()
        
        # Update bbox to reflect actual extracted region
        actual_w = x_end - x_start
        actual_h = y_end - y_start
        
        char_segment = remove_fragments(char_segment)
        
        segments.append((char_segment, (x_start, y_start, actual_w, actual_h)))
    
    return segments

# def split_wide_segments_rec(
#     characters: List[Tuple[np.ndarray, Tuple[int, int, int, int]]],
#     original_image_width: int,
#     width_threshold: float = 0.2,
#     max_depth: int = 5
# ) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
#     """
#     Recursively checks the width of each identified segment. For segments wider than
#     the threshold (default 20% of original image width), applies the segmentation
#     algorithm again on that segment to obtain the characters within it.
    
#     Args:
#         characters: List of tuples (character_image, (x, y, w, h))
#         original_image_width: Width of the original image in pixels
#         width_threshold: Threshold ratio (default: 0.2 = 20%)
#         max_depth: Maximum recursion depth to prevent infinite loops (default: 5)
    
#     Returns:
#         Updated list of character segments with wide segments recursively segmented
#     """
#     if max_depth <= 0:
#         # Base case: reached max depth, return segments as-is
#         return characters
    
#     threshold_width = int(original_image_width * width_threshold)
#     updated_characters = []
    
#     for char_img, bbox in characters:
#         x, y, w, h = bbox
        
#         # Check if segment width is >= threshold
#         if w >= threshold_width:
#             # Apply segmentation algorithm to this segment
#             # The segmentation function expects a binary image
#             sub_segments = segment_characters(char_img)
            
#             if len(sub_segments) > 1:
#                 # Successfully segmented into multiple characters
#                 # Adjust bounding boxes to be relative to the original image
#                 adjusted_segments = []
#                 for sub_char_img, (sub_x, sub_y, sub_w, sub_h) in sub_segments:
#                     # Adjust coordinates: sub_x and sub_y are relative to the char_img
#                     adjusted_bbox = (x + sub_x, y + sub_y, sub_w, sub_h)
#                     adjusted_segments.append((sub_char_img, adjusted_bbox))
                
#                 # Recursively process the sub-segments
#                 recursively_processed = split_wide_segments_rec(
#                     adjusted_segments,
#                     original_image_width,
#                     width_threshold,
#                     max_depth - 1
#                 )
#                 updated_characters.extend(recursively_processed)
                
#                 print(f"Recursively segmented segment at ({x}, {y}) with width {w} "
#                       f"into {len(recursively_processed)} sub-segments")
#             else:
#                 # Segmentation didn't split it further, keep original
#                 updated_characters.append((char_img, bbox))
#         else:
#             # Keep original segment if it's not wide enough
#             updated_characters.append((char_img, bbox))
    
#     return updated_characters

# def split_wide_segments(characters: List[Tuple[np.ndarray, Tuple[int, int, int, int]]], 
#                         original_image_width: int,
#                         width_threshold: float = 0.2) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
#     """
#     Check the width of each identified segment. For segments wider than the threshold
#     (default 20% of original image width), split them vertically down the middle.
    
#     Args:
#         characters: List of tuples (character_image, (x, y, w, h))
#         original_image_width: Width of the original image in pixels
#         width_threshold: Threshold ratio (default: 0.2 = 20%)
    
#     Returns:
#         Updated list of character segments with wide segments split
#     """
#     threshold_width = int(original_image_width * width_threshold)
#     updated_characters = []
    
#     for char_img, bbox in characters:
#         x, y, w, h = bbox
        
#         # Check if segment width is >= threshold
#         if w >= threshold_width:
#             # Split vertically down the middle
#             split_point = w // 2
            
#             # Extract left half
#             left_half = char_img[:, :split_point].copy()
#             left_bbox = (x, y, split_point, h)
            
#             # Extract right half
#             right_half = char_img[:, split_point:].copy()
#             right_bbox = (x + split_point, y, w - split_point, h)
            
#             # Add both halves to the updated list
#             updated_characters.append((left_half, left_bbox))
#             updated_characters.append((right_half, right_bbox))
            
#             print(f"Split segment at ({x}, {y}) with width {w} into two segments: "
#                   f"left ({split_point}px) and right ({w - split_point}px)")
#         else:
#             # Keep original segment if it's not wide enough
#             updated_characters.append((char_img, bbox))
    
#     return updated_characters

def main():
    # testing image loading
    # images to test: 32_2.png, 135_5.png, 73_5.png
    img_path = "iiit-5k/IIIT5K-Word_V3.0/IIIT5K/train/32_2.png"
    img = load_img(img_path)
    
    if img is None:
        print(f"Error: Could not load image from {img_path}")
        return
    
    # Process the image
    processed_img = preprocess(img)
    #processed_img = apply_morph(processed_img)
    processed_img = adjust_skew_hough(processed_img)
    processed_img = correct_slant(processed_img)
    #processed_img = cv2.threshold(processed_img, 245, 255, cv2.THRESH_BINARY)[1]
    thresholded = threshold(processed_img)
    # TODO: see if this is needed
   
    
    # Save the processed image
    save_img(processed_img, "processed.png")
    print("Processed image saved to processed.png")
    
    # Create a side-by-side comparison
    #visualize_preprocessing(img, "preprocessing_comparison.png")
    visualize_comparison(img, processed_img, "comparison.png")
    print("Comparison image saved to comparison.png")
    
    
    # Apply segmentation to detected_edges to get bounding boxes
    segments = segment_characters_projection(thresholded)
    print(f"Found {len(segments)} character segments from thresholded image")
    
    # Extract bounding boxes from edge segmentation
    bounding_boxes = [bbox for _, bbox in segments]
    
    # Extract actual character segments from processed_img using those bounding boxes
    characters = extract_segments_from_image(processed_img, bounding_boxes)
    #rint(f"Extracted {len(edge_characters)} character segments from processed_img")

    print(f"Extracted {len(characters)} character segments from processed_img")
    
    # # Split wide segments (>= 20% of original image width)
    # original_width = processed_img.shape[1]
    # characters = split_wide_segments(characters, original_width, width_threshold=0.3)
    # print(f"Found {len(characters)} characters after splitting wide segments")
    
    # Visualize segmentation on processed_img
    visualize_segmentation(processed_img, characters, "segmentation.png")
    print("Segmentation visualization saved to segmentation.png")
    
    # Save individual character images
    # Resize each character to fixed size (28x28) for CNN training
    for i, (char_img, _) in enumerate(characters):
        # Resize to fixed size while maintaining aspect ratio
        resized_char = resize_to_fixed_size(char_img, target_size=TARGET_SIZE, maintain_aspect=True)
        save_img(resized_char, f"char_{i:02d}.png")
        print(f"Original shape: {char_img.shape}, Resized shape: {resized_char.shape}")
    print(f"Saved {len(characters)} individual character images (resized to {TARGET_SIZE})")

if __name__ == "__main__":
    main()
