"""
1. load data
2. reduce noise
3. sharpen
4. increase contrast
5. adjust skew and slant
6. baseline extraction - feature extraction
7. Compression - thresholding
8. segment characters
9. train pytorch model to classify characters
"""

import cv2
import numpy as np
import os
import pillow as PIL

def load_data(path) -> PIL.Image.Image:
    """
    load data from path
    """
    return PIL.Image.open(path)

def reduce_noise(image: PIL.Image.Image) -> PIL.Image.Image:
    """
    reduce noise from image
    """
    return cv2.GaussianBlur(image, (5, 5), 0)

def sharpen(image: PIL.Image.Image) -> PIL.Image.Image:
    """
    sharpen image
    """
    return cv2.sharpen(image)

def increase_contrast(image: PIL.Image.Image) -> PIL.Image.Image:
    """
    increase contrast of image
    """
    return cv2.convertTo(image, cv2.CV_8U, 2, 0)

def adjust_skew(image: PIL.Image.Image) -> PIL.Image.Image:
    """
    adjust skew of image
    """
    return cv2.adjustSkew(image)

def extract_baseline(image: PIL.Image.Image) -> PIL.Image.Image:
    """
    extract baseline of image
    """
    return cv2.extractBaseline(image)

def compress_thresholding(image: PIL.Image.Image) -> PIL.Image.Image:
    """
    compress thresholding of image
    """
    return cv2.compressThresholding(image)

def segment_characters(image: PIL.Image.Image) -> PIL.Image.Image:
    """
    segment characters of image
    """
    return cv2.segmentCharacters(image)