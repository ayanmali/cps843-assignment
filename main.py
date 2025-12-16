import cv2
from src.preprocessing import adjust_skew_hough, correct_slant, load_img, preprocess, threshold
from src.utils import save_img, visualize_comparison, resize_to_fixed_size
from src.segmentation import segment_characters_projection, visualize_segmentation
import numpy as np
from typing import List, Tuple

TARGET_SIZE = (24, 24)

def get_largest_component(char_img: np.ndarray) -> np.ndarray:
    if char_img is None or char_img.size == 0:
        return char_img
    
    if len(char_img.shape) == 3:
        char_img = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
    
    if char_img.max() <= 1:
        char_img = (char_img * 255).astype(np.uint8)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(char_img, connectivity=8)
    
    if num_labels <= 1:
        return np.zeros_like(char_img)

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    clean = np.zeros_like(char_img)
    clean[labels == largest_label] = 255
    
    return clean

def extract_segments_from_image(source_image: np.ndarray, bounding_boxes):
    segments = []
    img_height, img_width = source_image.shape[:2]
    
    for x, y, w, h in bounding_boxes:
        x_start = max(0, x)
        y_start = max(0, y)
        x_end = min(img_width, x + w)
        y_end = min(img_height, y + h)
        
        char_segment = source_image[y_start:y_end, x_start:x_end].copy()
        
        actual_w = x_end - x_start
        actual_h = y_end - y_start
        
        char_segment = get_largest_component(char_segment)
        #char_segment = adjust_skew_hough(char_segment)
        #char_segment = correct_slant(char_segment)
        
        segments.append((char_segment, (x_start, y_start, actual_w, actual_h)))
    
    return segments

def split_wide_segments_by_average(
    segments: List[Tuple[np.ndarray, Tuple[int, int, int, int]]],
    threshold_multiplier: float = 1.5
) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    if len(segments) <= 1:
        return segments
    
    updated_segments = []
    
    for i, (char_img, bbox) in enumerate(segments):
        x, y, w, h = bbox
        
        other_widths = [seg_bbox[2] for j, (_, seg_bbox) in enumerate(segments) if j != i]
        
        if len(other_widths) == 0:
            updated_segments.append((char_img, bbox))
            continue
        
        avg_width = np.mean(other_widths)
        
        if w > threshold_multiplier * avg_width:
            num_chars = max(2, int(round(w / avg_width)))
            
            chunk_width = w // num_chars
            
            for j in range(num_chars):
                chunk_x_start = x + (j * chunk_width)
                chunk_x_end = x + ((j + 1) * chunk_width) if j < num_chars - 1 else x + w
                
                img_x_start = j * chunk_width
                img_x_end = (j + 1) * chunk_width if j < num_chars - 1 else char_img.shape[1]
                
                chunk_img = char_img[:, img_x_start:img_x_end].copy()
                
                if chunk_img.size == 0 or chunk_img.shape[0] == 0 or chunk_img.shape[1] == 0:
                    continue
                
                if np.sum(chunk_img > 0) == 0:
                    continue
                
                chunk_w = chunk_x_end - chunk_x_start
                chunk_bbox = (chunk_x_start, y, chunk_w, h)
                
                chunk_img = get_largest_component(chunk_img)
                
                if np.sum(chunk_img > 0) == 0:
                    continue
                
                updated_segments.append((chunk_img, chunk_bbox))
            
        else:
            updated_segments.append((char_img, bbox))
    
    return updated_segments

def main():
    img_path = "iiit-5k/IIIT5K-Word_V3.0/IIIT5K/train/73_5.png"
    img = load_img(img_path)
    
    if img is None:
        print(f"Error: Could not load image from {img_path}")
        return
    
    processed_img = preprocess(img)
    processed_img = threshold(processed_img, invert=True)
    processed_img = adjust_skew_hough(processed_img)
    processed_img = correct_slant(processed_img)

    thresh2 = threshold(processed_img)   
    
    save_img(processed_img, "processed.png")
    print("Processed image saved to processed.png")
    
    visualize_comparison(img, processed_img, "comparison.png")
    print("Comparison image saved to comparison.png")
    
    segments = segment_characters_projection(thresh2)
    print(f"Found {len(segments)} character segments using projection")
    
    bounding_boxes = [bbox for _, bbox in segments]
    
    characters = extract_segments_from_image(thresh2, bounding_boxes)
    characters = split_wide_segments_by_average(characters, threshold_multiplier=2.25)
    print(f"Found {len(segments)} character segments after splitting wide segments")

    visualize_segmentation(processed_img, characters, "segmentation.png")
    print("Segmentation visualization saved to segmentation.png")
    
    for i, (char_img, _) in enumerate(characters):
        resized_char = resize_to_fixed_size(char_img, target_size=TARGET_SIZE, maintain_aspect=True)
        save_img(resized_char, f"char_{i:02d}.png")
        print(f"Original shape: {char_img.shape}, Resized shape: {resized_char.shape}")
    print(f"Saved {len(characters)} individual character images (resized to {TARGET_SIZE})")

if __name__ == "__main__":
    main()
