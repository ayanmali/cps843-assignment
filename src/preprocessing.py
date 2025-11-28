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
from scipy.ndimage import interpolation
# from PIL import Image

def load_img(path: str) -> np.ndarray:
    """
    load image from path
    """
    return cv2.imread(path)

def preprocess(image):
    """
    compress image
    """
    # grayscaling image so that text is white, background is black
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # histogram equalization
    equalized = hist_equalization(grey)

    # correcting the background
    # se=cv2.getStructuringElement(cv2.MORPH_RECT , (3,3))
    # bg=cv2.morphologyEx(grey, cv2.MORPH_DILATE, se)
    # divide=cv2.divide(grey, bg, scale=255)

    # divide by blurred background

    # blur
    blur = cv2.GaussianBlur(grey, (0,0), sigmaX=33, sigmaY=33)
    # divide
    divide = cv2.divide(equalized, blur, scale=255)

    thresh = cv2.threshold(divide, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    return thresh

def hist_equalization(image):
    """
    equalize histogram of image
    improves contrast of image
    """
    return cv2.equalizeHist(image)

def apply_morph(image):
    """
    apply morphological operations to image (erode and dilate)
    removes noise and small artifacts
    """
    se=cv2.getStructuringElement(cv2.MORPH_RECT , (3,3))
    morph=cv2.morphologyEx(image, cv2.MORPH_CLOSE, se)
    out=cv2.divide(image, morph, scale=255)
    return out

def reduce_noise(image):
    """
    reduce noise from image
    """
    return cv2.GaussianBlur(image, (5, 5), 0)

def adjust_skew(image, limit=10, delta=0.075):
    """
    adjust skew of image
    """
    def find_score(arr: np.ndarray, angle: float) -> tuple[np.ndarray, float]:
        data = interpolation.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return score
    
    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        score = find_score(image, angle)
        scores.append(score)

    best_angle = angles[np.argmax(scores)]
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)

    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
                              borderMode=cv2.BORDER_REPLICATE)
    return rotated

# def extract_baseline(image: PIL.Image.Image) -> PIL.Image.Image:
#     """
#     extract baseline of image
#     """
#     return cv2.extractBaseline(image)
