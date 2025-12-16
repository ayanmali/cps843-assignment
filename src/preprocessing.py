"""
1. load data
2. grayscale
3. threshold
4. increase brightness and contrast
5. adjust skew and slant
"""

import cv2
import numpy as np
from scipy.ndimage import interpolation

def load_img(path: str) -> np.ndarray:
    return cv2.imread(path)

def preprocess(image):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightened = brighten(grey, gamma=0.45)
    stretched = stretch_contrast(brightened)

    equalized = cv2.equalizeHist(stretched)
    inverted = cv2.bitwise_not(stretched)
    
    blur = cv2.GaussianBlur(equalized, (0,0), sigmaX=33, sigmaY=33)
    return cv2.divide(inverted, blur, scale=255)

def threshold(image, invert=False):
    if invert:
        return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

def stretch_contrast(image):
    return cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

def brighten(image, gamma=0.5):
    look_up = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, look_up) 

# hekoer
def apply_shear(arr: np.ndarray, shear_angle: float) -> np.ndarray:
        h, w = arr.shape[:2]
        
        angle_rad = np.deg2rad(shear_angle)
        
        shear_factor = np.tan(angle_rad)
        
        new_width = int(w + abs(shear_factor) * h)
        
        tx = abs(shear_factor) * h / 2 if shear_factor < 0 else 0
        M = np.float32([[1, shear_factor, tx],
                        [0, 1, 0]])
        
        sheared = cv2.warpAffine(arr, M, (new_width, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        if new_width > w:
            start_x = (new_width - w) // 2
            sheared = sheared[:, start_x:start_x + w]
        elif new_width < w:
            pad_left = (w - new_width) // 2
            pad_right = w - new_width - pad_left
            sheared = np.pad(sheared, ((0, 0), (pad_left, pad_right)), mode='constant', constant_values=0)
        
        return sheared

# helper
def find_slant_score(arr: np.ndarray, shear_angle: float) -> float:
    sheared = apply_shear(arr, shear_angle)
    
    histogram = np.sum(sheared, axis=0, dtype=float)
    
    if np.sum(histogram) > 0:
        histogram = histogram / np.sum(histogram)
    
    variance = np.var(histogram)
    
    diff_squared = np.sum((histogram[1:] - histogram[:-1]) ** 2)
    
    sobel_x = cv2.Sobel(sheared, cv2.CV_64F, 1, 0, ksize=3)
    sobel_x = np.abs(sobel_x)
    
    vertical_edge_strength = np.sum(sobel_x)
    
    vertical_edge_score = vertical_edge_strength / (sheared.shape[0] * sheared.shape[1])
    
    score = variance * diff_squared * (1 + vertical_edge_score * 0.1)
    
    return score

# helper
def detect_slant_from_strokes(img: np.ndarray) -> float:
    edges = cv2.Canny(img, 50, 150)
    
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
    
    if lines is None or len(lines) == 0:
        return None
    
    vertical_angles = []
    for line in lines:
        _, theta = line[0]
        angle_deg = np.rad2deg(theta) - 90
        
        if abs(angle_deg) < 45:
            vertical_angles.append(angle_deg)
    
    if len(vertical_angles) == 0:
        return None
    
    detected_angle = np.median(vertical_angles)
    
    shear_angle = -detected_angle
    
    return shear_angle

def correct_slant(image, limit=30, delta=0.2):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if image.max() <= 1:
        image = (image * 255).astype(np.uint8)
    
    # first try edge detection method
    stroke_angle = detect_slant_from_strokes(image)
    
    if stroke_angle is not None and abs(stroke_angle) < limit:
        if abs(stroke_angle) >= 0.5:
            corrected = apply_shear(image, stroke_angle)
            return corrected
    
    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    
    for angle in angles:
        score = find_slant_score(image, angle)
        scores.append(score)
    
    best_angle = angles[np.argmax(scores)]
    # best_score = max(scores)
    
    if abs(best_angle) < 0.5:
        return image
    
    corrected = apply_shear(image, best_angle)
    
    return corrected

def find_score(arr: np.ndarray, angle: float) -> float:
    data = interpolation.rotate(arr, angle, reshape=False, order=0)
    
    histogram = np.sum(data, axis=1, dtype=float)
    
    variance = np.var(histogram)
    
    diff_squared = np.sum((histogram[1:] - histogram[:-1]) ** 2)
    
    score = variance * diff_squared
    return score

def adjust_skew(image, limit=20, delta=0.1):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if image.max() <= 1:
        image = (image * 255).astype(np.uint8)
    
    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    
    for angle in angles:
        score = find_score(image, angle)
        scores.append(score)

    best_angle = angles[np.argmax(scores)]
    
    if abs(best_angle) < 0.1:
        return image
    
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)

    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def adjust_skew_hough(image, limit=20, delta=0.1):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if image.max() <= 1:
        image = (image * 255).astype(np.uint8)
    
    lines = cv2.HoughLinesP(image, 1, np.pi/180, threshold=100, 
                            minLineLength=image.shape[1]//4, maxLineGap=20)
    
    if lines is None or len(lines) == 0:
        return adjust_skew(image, limit, delta)
    
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 != x1:
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if -limit <= angle <= limit:
                angles.append(angle)
    
    if len(angles) == 0:
        return adjust_skew(image, limit, delta)
    
    best_angle = np.median(angles)
    
    if abs(best_angle) < 0.1:
        return image
    
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated