"""
Parse training data, preprocess, and save words in a separate directory.
"""

import os
import sys

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.preprocessing import load_img, preprocess, threshold, adjust_skew_hough, correct_slant
from src.utils import save_img


def extract_label_from_filename(filename: str) -> str:
    basename = os.path.basename(filename)
    name_without_ext = os.path.splitext(basename)[0]
    
    parts = name_without_ext.split('_')
    if len(parts) >= 3:
        label = '_'.join(parts[1:-1])
        return label
    elif len(parts) == 2:
        return parts[1]
    else:
        return name_without_ext


def create_directories(split: str):
    os.makedirs(split, exist_ok=True)


def load_existing_word_counter(split: str) -> int:
    if not os.path.exists(split):
        return 0
    
    count = len([f for f in os.listdir(split) if f.endswith('.png')])
    return count


def sanitize_filename(label: str) -> str:
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    sanitized = label
    for char in invalid_chars:
        sanitized = sanitized.replace(char, '_')
    return sanitized


def process_word_image(
    img_path: str,
    ground_truth: str,
    split: str,
    base_dir: str,
    word_counter: int
) -> tuple[bool, int]:
    full_path = os.path.join(base_dir, img_path.lstrip('./'))
    img = load_img(full_path)
    if img is None:
        print(f"Warning: Could not load image {full_path}, skipping...")
        return False, word_counter
    
    try:
        processed_img = preprocess(img)
        processed_img = threshold(processed_img, invert=True)
        processed_img = adjust_skew_hough(processed_img)
        processed_img = correct_slant(processed_img)
        processed_img = threshold(processed_img)
    except Exception as e:
        print(f"Error preprocessing image {full_path}: {e}")
        return False, word_counter
    
    sanitized_label = sanitize_filename(ground_truth)
    
    word_counter += 1
    filename = f"{sanitized_label}_{word_counter:06d}.png"
    filepath = os.path.join(split, filename)
    
    save_img(processed_img, filepath)
    
    return True, word_counter


def contains_numbers(s):
    return any(char.isdigit() for char in s)


def extract_words(
    split: str,
    annotation_file: str,
    base_dir: str,
    num_samples: int
):
    print(f"\nLoading annotation file: {annotation_file}")
    image_paths = []
    with open(annotation_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()    
            if '_' in parts[0]:
                first_part = parts[0].split('_')[1]
                if contains_numbers(first_part):
                    continue
            if len(parts) >= 1:
                image_paths.append(parts[0])
            
    print(f"\n{'='*60}")
    print(f"Processing {split} samples...")
    print(f"{'='*60}")
    
    word_counter = load_existing_word_counter(split)
    print(f"Starting from word counter: {word_counter}")

    print(f"Total images in annotation file: {len(image_paths)}")
    print(f"Word counter (already processed): {word_counter}")
    print(f"Available images after word counter: {len(image_paths) - word_counter}")
    print(f"Samples to extract: {num_samples}")
    
    available_images = len(image_paths) - word_counter
    
    if available_images < num_samples:
        print(f"Warning: Only {available_images} images available after offset {word_counter}, "
              f"but {num_samples} requested.")
        num_samples = available_images
    
    total_saved = 0
    total_failed = 0
    
    start_idx = word_counter
    end_idx = word_counter + num_samples
    
    for idx in range(start_idx, end_idx):
        img_path = image_paths[idx]
        ground_truth = extract_label_from_filename(img_path)
        
        success, word_counter = process_word_image(
            img_path, ground_truth, split, base_dir, word_counter
        )
        
        if success:
            total_saved += 1
        else:
            total_failed += 1
        
        processed_count = idx - start_idx + 1
        if processed_count % 100 == 0:
            print(f"Processed {processed_count}/{num_samples} images, "
                  f"saved {total_saved} words so far...")
    
    print(f"\n{split} extraction complete: Saved {total_saved} word images")
    
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Total words saved: {total_saved}")
    print(f"Total words failed: {total_failed}")
    print(f"Final word counter: {word_counter}")


if __name__ == "__main__":
    base_dir = "mnt/ramdisk/max/90kDICT32px"
    # base_dir = "/content/drive/MyDrive/synth-90k"
    
    train_samples = 500
    val_samples = 200

    train_annotation_file = os.path.join(base_dir, "annotation_train.txt")
    val_annotation_file = os.path.join(base_dir, "annotation_val.txt")
    
    # Create directories
    print("Creating directories...")
    create_directories('word_train')
    create_directories('word_val')

    extract_words(
        split='word_train',
        annotation_file=train_annotation_file,
        base_dir=base_dir,
        num_samples=train_samples,
    )
    
    extract_words(
        split='word_val',
        annotation_file=val_annotation_file,
        base_dir=base_dir,
        num_samples=val_samples,
    )