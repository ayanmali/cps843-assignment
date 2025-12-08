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
    equalized = cv2.equalizeHist(grey)
    inverted = cv2.bitwise_not(grey)

    # correcting the background
    # se=cv2.getStructuringElement(cv2.MORPH_RECT , (3,3))
    # bg=cv2.morphologyEx(grey, cv2.MORPH_DILATE, se)
    # divide=cv2.divide(grey, bg, scale=255)

    # divide by blurred background
    # blur
    blur = cv2.GaussianBlur(equalized, (0,0), sigmaX=33, sigmaY=33)
    # divide
    divide = cv2.divide(inverted, blur, scale=255)

    #median = cv2.medianBlur(divide, 3)

    thresh = cv2.threshold(divide, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    return cv2.bitwise_not(thresh)
    #return remove_noise(thresh)

def apply_morph(image):
    """
    apply morphological operations to image (erode and dilate)
    removes noise and small artifacts
    """
    se = cv2.getStructuringElement(cv2.MORPH_RECT , (2,2))
    morph = cv2.morphologyEx(image, cv2.MORPH_CLOSE, se)
    out = cv2.divide(image, morph, scale=255)
    return out

def detect_edges(image):
    """
    detect edges from image
    """
    edges = cv2.Canny(image, 50, 150)
    return edges

def fill_edges(edges):
    """
    Fill edges from image, preserving holes in characters (e.g., the center of 'A', 'B', 'D', etc.).
    
    Uses contour hierarchy to distinguish between external boundaries (which should be filled)
    and internal holes (which should remain empty).
    
    Args:
        edges: Binary edge image from Canny edge detection (edges should be white/255, background black/0)
    
    Returns:
        Filled binary image with characters filled but holes preserved
    """
    # Use RETR_CCOMP to get contour hierarchy:
    # - External contours: have no parent (hierarchy[i][3] == -1)
    # - Holes: have a parent (hierarchy[i][3] != -1) and are children of external contours
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return cv2.bitwise_not(edges)
    
    # Create a blank image to draw filled contours on
    filled = np.zeros_like(edges)
    
    # Extract hierarchy array (it's nested)
    hierarchy = hierarchy[0]
    
    # Fill only external contours (those without a parent)
    # Skip holes (those with a parent) to preserve them
    for i, contour in enumerate(contours):
        # hierarchy[i][3] == -1 means no parent, so it's a top-level external contour
        # hierarchy[i][3] != -1 means it has a parent, so it's a hole (skip it)
        if hierarchy[i][3] == -1:
            # This is an external contour, fill it
            cv2.drawContours(filled, [contour], -1, 255, -1)
        # Holes are skipped, leaving them as background (0)
    
    # Apply morphological operations to smooth the result
    filled = cv2.erode(filled, None, iterations=1)
    filled = cv2.dilate(filled, None, iterations=1)

    return cv2.bitwise_not(filled)

# def fill_edges(image, method='contours', kernel_size=(3, 3)):
#     """
#     Fill character outlines after edge detection.
    
#     Args:
#         image: Binary image with edge outlines (edges should be white/255, background black/0)
#         method: Method to use for filling:
#                 - 'contours': Find contours and fill them (most accurate)
#                 - 'morphological': Use morphological closing to fill gaps (faster)
#                 - 'both': Use contours first, then morphological closing (most robust)
#         kernel_size: Size of morphological kernel (height, width) for closing operation
    
#     Returns:
#         Filled binary image with filled characters
#     """
#     # Ensure binary format
#     if len(image.shape) == 3:
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     if image.max() <= 1:
#         image = (image * 255).astype(np.uint8)
    
#     result = image.copy()
    
#     if method in ['contours', 'both']:
#         # Find contours from edges
#         # Note: cv2.RETR_EXTERNAL gets only outer contours, cv2.RETR_TREE gets all
#         contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
#         # Create a blank image
#         filled = np.zeros_like(image)
        
#         # Fill contours
#         cv2.drawContours(filled, contours, -1, 255, -1)  # -1 fills the contour
        
#         result = filled
    
#     if method in ['morphological', 'both']:
#         # Apply morphological closing to fill small gaps
#         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
#         result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=1)
    
#     return result
# TODO: see if this is needed
def remove_noise(image):
    """
    remove noise from image
    """
    se = cv2.getStructuringElement(cv2.MORPH_RECT , (3,3))
    morph = cv2.morphologyEx(image, cv2.MORPH_OPEN, se)
    out = cv2.divide(image, morph, scale=255)
    return out


# def reduce_noise(image):
#     """
#     reduce noise from image
#     """
#     return cv2.GaussianBlur(image, (5, 5), 0)

# def debug_skew_detection(image, limit=20, delta=0.1):
#     """
#     Debug function to visualize skew detection scores.
#     Useful for understanding why skew correction might not be working.
    
#     Returns:
#         dict with 'best_angle', 'scores', 'angles', and 'max_score'
#     """
#     if len(image.shape) == 3:
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     if image.max() <= 1:
#         image = (image * 255).astype(np.uint8)
    
#     def find_score(arr: np.ndarray, angle: float) -> float:
#         data = interpolation.rotate(arr, angle, reshape=False, order=0)
#         histogram = np.sum(data, axis=1, dtype=float)
#         variance = np.var(histogram)
#         diff_squared = np.sum((histogram[1:] - histogram[:-1]) ** 2)
#         score = variance * diff_squared
#         return score
    
#     scores = []
#     angles = np.arange(-limit, limit + delta, delta)
    
#     for angle in angles:
#         score = find_score(image, angle)
#         scores.append(score)
    
#     best_angle = angles[np.argmax(scores)]
#     max_score = max(scores)
    
#     return {
#         'best_angle': best_angle,
#         'scores': scores,
#         'angles': angles,
#         'max_score': max_score
#     }

"""
adjust skew of image using projection profile variance method.
fallback if hough line detection fails.
"""
def adjust_skew(image, limit=20, delta=0.1):
    """
    adjust skew of image using projection profile variance method.
    When text is properly aligned, the variance of horizontal projection is maximized.
    
    Args:
        image: Binary image (text should be white/255, background black/0)
        limit: Maximum angle to search in degrees (default: 20)
        delta: Step size for angle search in degrees (default: 0.1)
    
    Returns:
        Rotated image with corrected skew
    """
    def find_score(arr: np.ndarray, angle: float) -> float:
        """
        Calculate skew score for a given angle.
        Higher variance in horizontal projection indicates better alignment.
        """
        # Rotate the image
        data = interpolation.rotate(arr, angle, reshape=False, order=0)
        
        # Calculate horizontal projection (sum of pixels in each row)
        # For binary images, this counts white pixels per row
        histogram = np.sum(data, axis=1, dtype=float)
        
        # Calculate variance of the projection profile
        # When text is aligned, variance is high (peaks at text rows, valleys at gaps)
        # When skewed, variance decreases as text spreads across rows
        variance = np.var(histogram)
        
        # Alternative: use sum of squared differences (more sensitive)
        # This measures how "peaky" the histogram is
        diff_squared = np.sum((histogram[1:] - histogram[:-1]) ** 2)
        
        # Combine both metrics for better detection
        score = variance * diff_squared
        return score
    
    # Ensure we're working with a binary image
    # If image has multiple channels, convert to grayscale first
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Ensure binary (0 or 255)
    if image.max() <= 1:
        image = (image * 255).astype(np.uint8)
    
    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    
    for angle in angles:
        score = find_score(image, angle)
        scores.append(score)

    best_angle = angles[np.argmax(scores)]
    
    # Only rotate if the detected angle is significant enough
    if abs(best_angle) < 0.1:
        return image
    
    # Rotate the image using OpenCV
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)

    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
                              borderMode=cv2.BORDER_REPLICATE)
    return rotated

def adjust_skew_hough(image, limit=20, delta=0.1):
    """
    Alternative skew correction using Hough line detection.
    This method detects text baselines and calculates skew angle from them.
    Often more robust than projection profile method for images with clear text lines.
    
    Args:
        image: Binary image (text should be white/255, background black/0)
        limit: Maximum angle to search in degrees (default: 20)
        delta: Step size for angle search in degrees (default: 0.1)
    
    Returns:
        Rotated image with corrected skew
    """
    # Ensure binary image
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if image.max() <= 1:
        image = (image * 255).astype(np.uint8)
    
    # Detect edges/lines using Hough transform
    # Use probabilistic Hough transform to detect lines
    lines = cv2.HoughLinesP(image, 1, np.pi/180, threshold=100, 
                            minLineLength=image.shape[1]//4, maxLineGap=20)
    
    if lines is None or len(lines) == 0:
        # Fallback to projection profile method if no lines detected
        return adjust_skew(image, limit, delta)
    
    # Calculate angles of detected lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 != x1:  # Avoid division by zero
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            # Filter angles within reasonable range
            if -limit <= angle <= limit:
                angles.append(angle)
    
    if len(angles) == 0:
        return adjust_skew(image, limit, delta)
    
    # Use median angle (more robust to outliers than mean)
    best_angle = np.median(angles)
    
    if abs(best_angle) < 0.1:
        return image
    
    # Rotate the image
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return rotated

# def extract_baseline(image: PIL.Image.Image) -> PIL.Image.Image:
#     """
#     extract baseline of image
#     """
#     return cv2.extractBaseline(image)
