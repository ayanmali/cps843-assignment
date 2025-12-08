import cv2
from src.preprocessing import adjust_skew_hough, apply_morph, detect_edges, fill_edges, load_img, preprocess
# Alternative: remove_noise_connected_components() for more precise but slower noise removal
from src.utils import save_img, visualize_comparison, resize_to_fixed_size
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
    # TODO: see if this is needed
    # detected_edges = detect_edges(processed_img)
    # processed_img = fill_edges(detected_edges)

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
    
    characters = segment_characters_hybrid(processed_img)
    print(f"Found {len(characters)} characters")
    
    # # Visualize segmentation
    visualize_segmentation(processed_img, characters, "segmentation.png")
    print("Segmentation visualization saved to segmentation.png")
    
    # # Save individual character images
    # Resize each character to fixed size (28x28) for CNN training
    target_size = (28, 28)
    for i, (char_img, bbox) in enumerate(characters):
        # Resize to fixed size while maintaining aspect ratio
        # pad_color=0 means black background (since preprocess outputs text=white, bg=black)
        resized_char = resize_to_fixed_size(char_img, target_size=target_size, 
                                           maintain_aspect=True, pad_color=255)
        save_img(resized_char, f"char_{i:02d}.png")
        print(f"Original shape: {char_img.shape}, Resized shape: {resized_char.shape}")
    print(f"Saved {len(characters)} individual character images (resized to {target_size})")

if __name__ == "__main__":
    main()
