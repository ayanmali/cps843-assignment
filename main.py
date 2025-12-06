from src.preprocessing import adjust_skew, adjust_skew_hough, apply_morph, load_img, preprocess, remove_noise
# Alternative: remove_noise_connected_components() for more precise but slower noise removal
from src.utils import save_img, visualize_comparison
# Import all segmentation methods - try different ones if hybrid doesn't work well
from src.segmentation import segment_characters, segment_characters_projection, segment_characters_hybrid, visualize_segmentation  # noqa: F401

def main():
    # testing image loading
    img_path = "iiit-5k/IIIT5K-Word_V3.0/IIIT5K/train/135_5.png"
    img = load_img(img_path)
    
    if img is None:
        print(f"Error: Could not load image from {img_path}")
        return
    
    # Process the image
    processed_img = preprocess(img)
    processed_img = apply_morph(processed_img)
    processed_img = adjust_skew_hough(processed_img)
    
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
    characters = segment_characters_projection(processed_img)
    print(f"Found {len(characters)} characters")
    
    # # Visualize segmentation
    visualize_segmentation(processed_img, characters, "segmentation.png")
    print("Segmentation visualization saved to segmentation.png")
    
    # # Save individual character images
    for i, (char_img, bbox) in enumerate(characters):
        save_img(char_img, f"char_{i:02d}.png")
    print(f"Saved {len(characters)} individual character images")

if __name__ == "__main__":
    main()
