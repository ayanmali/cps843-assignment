from src.preprocessing import compress, load_img
from src.utils import save_img, visualize_preprocessing

def main():
    # testing image loading
    img_path = "iiit-5k/IIIT5K-Word_V3.0/IIIT5K/train/135_5.png"
    img = load_img(img_path)
    
    if img is None:
        print(f"Error: Could not load image from {img_path}")
        return
    
    # Process the image
    processed_img = compress(img)
    
    # Save the processed image
    save_img(processed_img, "processed.png")
    print("Processed image saved to processed.png")
    
    # Create a side-by-side comparison
    visualize_preprocessing(img, "preprocessing_comparison.png")
    print("Comparison image saved to preprocessing_comparison.png")

if __name__ == "__main__":
    main()
