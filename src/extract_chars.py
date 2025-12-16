"""
Parse training data, preprocess, segment, and save characters in a separate directory.
Extracts 35,000 samples for training and 7,000 for validation.
"""
import os
import sys

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from main import TARGET_SIZE, extract_segments_from_image, split_wide_segments_by_average
from src.preprocessing import load_img, preprocess, threshold, adjust_skew_hough, correct_slant
from src.segmentation import segment_characters_projection
from src.utils import resize_to_fixed_size, save_img
from src.labels import LABELS


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


def get_char_directory(char: str, split: str) -> str:
    if char.isupper():
        return os.path.join(split, 'cap', char)
    elif char.islower():
        return os.path.join(split, 'lc', char)
    else:
        return os.path.join(split, 'other', char)


def create_directories(split: str):
    for char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        dir_path = os.path.join(split, 'cap', char)
        os.makedirs(dir_path, exist_ok=True)
    
    for char in 'abcdefghijklmnopqrstuvwxyz':
        dir_path = os.path.join(split, 'lc', char)
        os.makedirs(dir_path, exist_ok=True)


def load_existing_char_counters(split: str) -> dict:
    char_counter = {}
    
    for char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        dir_path = os.path.join(split, 'cap', char)
        if os.path.exists(dir_path):
            count = len([f for f in os.listdir(dir_path) if f.endswith('.png')])
            if count > 0:
                char_counter[char] = count
    
    for char in 'abcdefghijklmnopqrstuvwxyz':
        dir_path = os.path.join(split, 'lc', char)
        if os.path.exists(dir_path):
            count = len([f for f in os.listdir(dir_path) if f.endswith('.png')])
            if count > 0:
                char_counter[char] = count
    
    return char_counter

def process_image(
    img_path: str,
    ground_truth: str,
    split: str,
    base_dir: str,
    char_counter: dict
) -> int:
    full_path = os.path.join(base_dir, img_path.lstrip('./'))
    img = load_img(full_path)
    if img is None:
        print(f"Warning: Could not load image {full_path}, skipping...")
        return 0
    
    processed_img = preprocess(img)
    processed_img = threshold(processed_img, invert=True)
    processed_img = adjust_skew_hough(processed_img)
    processed_img = correct_slant(processed_img)
    thresh2 = threshold(processed_img)
    
    segments = segment_characters_projection(thresh2)
    
    bounding_boxes = [bbox for _, bbox in segments]
    
    characters = extract_segments_from_image(thresh2, bounding_boxes)
    
    characters = split_wide_segments_by_average(characters, threshold_multiplier=2.25)
    
    gt_chars = list(ground_truth)
    
    num_chars = min(len(characters), len(gt_chars))
    
    saved_count = 0
    failed_count = 0
    for i in range(num_chars):
        char_img, bbox = characters[i]
        gt_char = gt_chars[i]
        
        if gt_char not in LABELS:
            failed_count += 1
            print(f"Skipping character {gt_char} because it is not in our label set")
            continue
        
        try:
            resized_char = resize_to_fixed_size(
                char_img,
                target_size=TARGET_SIZE,
                maintain_aspect=True
            )
        except Exception as e:
            print(f"Error resizing character {gt_char}: {e}")
            failed_count += 1
            continue
        
        char_dir = get_char_directory(gt_char, split)
        
        if gt_char not in char_counter:
            char_counter[gt_char] = 0
        char_counter[gt_char] += 1
        
        filename = f"{gt_char}_{char_counter[gt_char]:06d}.png"
        filepath = os.path.join(char_dir, filename)
        
        save_img(resized_char, filepath)
        saved_count += 1
    
    return saved_count, failed_count

def extract_characters(
    split: str,
    annotation_file: str,
    base_dir: str,
    num_samples: int,
    offset: int
):
    print(f"\nLoading annotation file: {annotation_file}")
    image_paths = []
    with open(annotation_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 1:
                image_paths.append(parts[0])
    
    if offset >= len(image_paths):
        print(f"Error: Offset {offset} is greater than or equal to total images {len(image_paths)}")
        return
    
    available_images = len(image_paths) - offset
    
    if available_images < num_samples:
        print(f"Warning: Only {available_images} images available after offset {offset}, "
              f"but {num_samples} requested.")
        num_samples = available_images
    
    print(f"Total images in annotation file: {len(image_paths)}")
    print(f"Offset (already processed): {offset}")
    print(f"Available images after offset: {available_images}")
    print(f"Training samples to extract: {train_samples}")
    print(f"Validation samples to extract: {val_samples}")
    
    print(f"\n{'='*60}")
    print(f"Processing {split} samples...")
    print(f"{'='*60}")
    char_counter = load_existing_char_counters(split)
    total_saved = 0
    total_failed = 0
    
    start_idx = offset
    end_idx = offset + num_samples
    
    for idx in range(start_idx, end_idx):
        img_path = image_paths[idx]
        ground_truth = extract_label_from_filename(img_path)
        
        saved, failed = process_image(img_path, ground_truth, split, base_dir, char_counter)
        total_saved += saved
        total_failed += failed
        
        processed_count = idx - start_idx + 1
        if processed_count % 100 == 0:
            print(f"Processed {processed_count}/{num_samples} images, "
                  f"saved {total_saved} characters so far")
    
    print(f"{split} extraction complete: Saved {total_saved} character segments")
    print("\nCharacter counts:")
    for char in sorted(char_counter.keys()):
        print(f"  {char}: {char_counter[char]}")
    
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Total characters saved: {total_saved}")
    print(f"Total characters failed: {total_failed}")

if __name__ == "__main__":
    base_dir = "mnt/ramdisk/max/90kDICT32px"
    # base_dir = "/content/drive/MyDrive/synth-90k"
    
    train_dir = "char_train/"
    val_dir = "char_val/"
    train_samples = 500000
    val_samples = 20000
    train_offset = 120000 + 80000 + 3000 + 550
    val_offset = 21000 + 80000 + 3000 + 550

    train_annotation_file = os.path.join(base_dir, "annotation_train.txt")
    val_annotation_file = os.path.join(base_dir, "annotation_val.txt")
    
    # Create directories
    print("Creating directories...")
    create_directories('char_train')
    create_directories('char_val')

    extract_characters(split='char_train', annotation_file=train_annotation_file, base_dir=base_dir, num_samples=train_samples, offset=train_offset)
    extract_characters(split='char_val', annotation_file=val_annotation_file, base_dir=base_dir, num_samples=val_samples, offset=val_offset)