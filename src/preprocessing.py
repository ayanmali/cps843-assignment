"""
1. load data
    data augmentation - rotate, scale, shear, brightness, blur
2. Compression - grayscale, threshold`
2. reduce noise
3. sharpen
4. increase contrast
5. adjust skew and slant
6. baseline extraction - feature extraction (?)
8. segment characters
9. train pytorch model to classify characters
"""

import cv2
import numpy as np
# from PIL import Image

def load_img(path: str) -> np.ndarray:
    """
    load image from path
    """
    return cv2.imread(path)

def compress(image):
    """
    compress image
    """
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    return thresh

# def reduce_noise(image: PIL.Image.Image) -> PIL.Image.Image:
#     """
#     reduce noise from image
#     """
#     return cv2.GaussianBlur(image, (5, 5), 0)

# def sharpen(image: PIL.Image.Image) -> PIL.Image.Image:
#     """
#     sharpen image
#     """
#     return cv2.sharpen(image)

# def increase_contrast(image: PIL.Image.Image) -> PIL.Image.Image:
#     """
#     increase contrast of image
#     """
#     return cv2.convertTo(image, cv2.CV_8U, 2, 0)

# def adjust_skew(image):
#     """
#     adjust skew of image
#     """
#     return cv2.adjustSkew(image)

# def extract_baseline(image: PIL.Image.Image) -> PIL.Image.Image:
#     """
#     extract baseline of image
#     """
#     return cv2.extractBaseline(image)

# def compress_thresholding(image: PIL.Image.Image) -> PIL.Image.Image:
#     """
#     compress thresholding of image
#     """
#     return cv2.compressThresholding(image)

# def segment_characters(image: PIL.Image.Image) -> PIL.Image.Image:
#     """
#     segment characters of image
#     """
#     return cv2.segmentCharacters(image)