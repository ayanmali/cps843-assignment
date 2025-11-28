from src.preprocessing import adjust_skew, apply_morph, load_img, preprocess
from src.utils import save_img, visualize_comparison

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
    processed_img = adjust_skew(processed_img)
    
    # Save the processed image
    save_img(processed_img, "processed.png")
    print("Processed image saved to processed.png")
    
    # Create a side-by-side comparison
    #visualize_preprocessing(img, "preprocessing_comparison.png")
    visualize_comparison(img, processed_img, "comparison.png")
    print("Comparison image saved to comparison.png")

if __name__ == "__main__":
    main()
