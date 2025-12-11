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
    brightened = brighten(grey, gamma=0.5) # 0.5 is default
    stretched = stretch_contrast(brightened)

    equalized = cv2.equalizeHist(stretched)
    inverted = cv2.bitwise_not(stretched)
    
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

    thresh = threshold(divide, invert=True)

    return thresh
    #return remove_noise(thresh)

def threshold(image, invert=False):
    if invert:
        return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

def stretch_contrast(image):
    return cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    return clahe.apply(image)

"""
Brighten the image by applying a gamma correction.
Gamma is a value between 0 and 1.
A value less than 1 will brighten the image.
A value greater than 1 will darken the image.
"""
def brighten(image, gamma=0.5):
    look_up = np.array([((i / 255.0) ** gamma) * 255
                    for i in range(256)]).astype("uint8")
    return cv2.LUT(image, look_up)

def apply_morph(image):
    """
    apply morphological operations to image (erode and dilate)
    removes noise and small artifacts
    """
    se = cv2.getStructuringElement(cv2.MORPH_RECT , (2,2))
    morph = cv2.morphologyEx(image, cv2.MORPH_CLOSE, se)
    out = cv2.divide(image, morph, scale=255)
    return out

# def detect_edges(image):
#     """
#     detect edges from image
#     """
#     edges = cv2.Canny(image, 50, 150)
#     return edges    

def correct_slant(image, limit=30, delta=0.5):
    """
    Correct slant (italic-like angle) in text images using shear transformation.
    
    Slant is different from skew:
    - Skew: Rotation of entire text line (corrected with rotation)
    - Slant: Horizontal shear/italicization of characters (corrected with shear transform)
    
    This function uses vertical projection profile analysis to find the optimal
    shear angle. When text is properly unslanted, vertical strokes are vertical,
    maximizing the variance of vertical projections.
    
    Args:
        image: Binary image (text should be white/255, background black/0)
        limit: Maximum shear angle in degrees to search (default: 30)
        delta: Step size for shear angle search in degrees (default: 0.5)
    
    Returns:
        Image with corrected slant
    """
    def apply_shear(arr: np.ndarray, shear_angle: float) -> np.ndarray:
        """
        Apply shear transformation to image.
        
        Args:
            arr: Input image
            shear_angle: Shear angle in degrees 
                        (positive = shear right/top-right, negative = shear left/top-left)
        
        Returns:
            Sheared image
        """
        h, w = arr.shape[:2]
        
        # Convert angle to radians
        angle_rad = np.deg2rad(shear_angle)
        
        # Create horizontal shear transformation matrix
        # For horizontal shear: [1, tan(angle), 0]
        #                        [0, 1,          0]
        # This shears horizontally: positive angle shears right, negative shears left
        shear_factor = np.tan(angle_rad)
        
        # Calculate new width to accommodate sheared image
        # When shearing right (positive), we need more width on the right
        # When shearing left (negative), we need more width on the left
        new_width = int(w + abs(shear_factor) * h)
        
        # Create transformation matrix
        # Translation in x to keep image centered
        tx = abs(shear_factor) * h / 2 if shear_factor < 0 else 0
        M = np.float32([[1, shear_factor, tx],
                        [0, 1, 0]])
        
        # Apply shear transformation
        sheared = cv2.warpAffine(arr, M, (new_width, h), 
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
        
        # Crop back to original width (center crop)
        if new_width > w:
            start_x = (new_width - w) // 2
            sheared = sheared[:, start_x:start_x + w]
        elif new_width < w:
            # If somehow smaller, pad it
            pad_left = (w - new_width) // 2
            pad_right = w - new_width - pad_left
            sheared = np.pad(sheared, ((0, 0), (pad_left, pad_right)), mode='constant', constant_values=0)
        
        return sheared
    
    def find_slant_score(arr: np.ndarray, shear_angle: float) -> float:
        """
        Calculate slant score for a given shear angle.
        Uses multiple metrics to detect optimal unslanting:
        1. Vertical projection variance (higher = better vertical alignment)
        2. Sum of squared differences in projection (measures "peakiness")
        3. Edge-based vertical stroke detection
        """
        # Apply shear transformation
        sheared = apply_shear(arr, shear_angle)
        
        # Method 1: Vertical projection analysis
        # Calculate vertical projection (sum of pixels in each column)
        # For binary images, this counts white pixels per column
        histogram = np.sum(sheared, axis=0, dtype=float)
        
        # Normalize histogram to avoid bias from image size
        if np.sum(histogram) > 0:
            histogram = histogram / np.sum(histogram)
        
        # Calculate variance of the vertical projection profile
        # When text is properly unslanted, vertical strokes align vertically,
        # creating peaks in the projection (high variance)
        variance = np.var(histogram)
        
        # Sum of squared differences (measures how "peaky" the histogram is)
        diff_squared = np.sum((histogram[1:] - histogram[:-1]) ** 2)
        
        # Method 2: Detect vertical edges/strokes
        # Use Sobel operator to detect vertical edges
        sobel_x = cv2.Sobel(sheared, cv2.CV_64F, 1, 0, ksize=3)
        sobel_x = np.abs(sobel_x)
        
        # Calculate vertical edge strength (sum of vertical edges)
        vertical_edge_strength = np.sum(sobel_x)
        
        # Normalize by image size
        vertical_edge_score = vertical_edge_strength / (sheared.shape[0] * sheared.shape[1])
        
        # Combine metrics (weighted combination)
        score = variance * diff_squared * (1 + vertical_edge_score * 0.1)
        
        return score
    
    # Ensure we're working with a binary image
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Ensure binary (0 or 255)
    if image.max() <= 1:
        image = (image * 255).astype(np.uint8)
    
    # Alternative method: Detect slant by analyzing vertical stroke angles
    # This can be more robust for some images
    def detect_slant_from_strokes(img: np.ndarray) -> float:
        """
        Detect slant angle by analyzing vertical stroke directions using Hough transform.
        """
        # Detect edges
        edges = cv2.Canny(img, 50, 150)
        
        # Use HoughLines to detect lines (especially vertical strokes)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
        
        if lines is None or len(lines) == 0:
            return None
        
        # Filter for near-vertical lines and calculate their angles
        vertical_angles = []
        for line in lines:
            rho, theta = line[0]
            # Convert to degrees
            angle_deg = np.rad2deg(theta) - 90  # Adjust to horizontal reference
            
            # Filter for lines that are close to vertical (within ±45 degrees)
            if abs(angle_deg) < 45:
                vertical_angles.append(angle_deg)
        
        if len(vertical_angles) == 0:
            return None
        
        # Use median angle (more robust to outliers)
        detected_angle = np.median(vertical_angles)
        
        # Convert stroke angle to shear angle
        # If strokes lean right (positive angle), we need to shear left (negative)
        # If strokes lean left (negative angle), we need to shear right (positive)
        shear_angle = -detected_angle
        
        return shear_angle
    
    # Try stroke-based detection first (often more accurate)
    stroke_angle = detect_slant_from_strokes(image)
    
    if stroke_angle is not None and abs(stroke_angle) < limit:
        print(f"Detected slant from strokes: {stroke_angle:.2f} degrees")
        if abs(stroke_angle) >= 0.5:
            corrected = apply_shear(image, stroke_angle)
            return corrected
    
    # Fallback to projection profile method
    print("Using projection profile method for slant detection...")
    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    
    for angle in angles:
        score = find_slant_score(image, angle)
        scores.append(score)
    
    best_angle = angles[np.argmax(scores)]
    best_score = max(scores)
    
    # Debug: print detected angle
    print(f"Detected slant angle: {best_angle:.2f} degrees (score: {best_score:.2e})")
    
    # Only apply correction if the detected angle is significant enough
    if abs(best_angle) < 0.5:
        print("Slant angle too small, skipping correction")
        return image
    
    # Apply the correction shear directly
    # If best_angle is negative (shear left), it corrects right-slanted text
    # If best_angle is positive (shear right), it corrects left-slanted text
    # The best_angle is the angle that maximizes vertical alignment, so apply it directly
    corrected = apply_shear(image, best_angle)
    
    return corrected

# def fill_edges(edges):
#     """
#     Fill edges from image, preserving holes in characters (e.g., the center of 'A', 'B', 'D', etc.).
    
#     Uses contour hierarchy to distinguish between external boundaries (which should be filled)
#     and internal holes (which should remain empty).
    
#     Args:
#         edges: Binary edge image from Canny edge detection (edges should be white/255, background black/0)
    
#     Returns:
#         Filled binary image with characters filled but holes preserved
#     """
#     # Use RETR_CCOMP to get contour hierarchy:
#     # - External contours: have no parent (hierarchy[i][3] == -1)
#     # - Holes: have a parent (hierarchy[i][3] != -1) and are children of external contours
#     contours, hierarchy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
#     if len(contours) == 0:
#         return cv2.bitwise_not(edges)
    
#     # Create a blank image to draw filled contours on
#     filled = np.zeros_like(edges)
    
#     # Extract hierarchy array (it's nested)
#     hierarchy = hierarchy[0]
    
#     # Fill only external contours (those without a parent)
#     # Skip holes (those with a parent) to preserve them
#     for i, contour in enumerate(contours):
#         # hierarchy[i][3] == -1 means no parent, so it's a top-level external contour
#         # hierarchy[i][3] != -1 means it has a parent, so it's a hole (skip it)
#         if hierarchy[i][3] == -1:
#             # This is an external contour, fill it
#             cv2.drawContours(filled, [contour], -1, 255, -1)
#         # Holes are skipped, leaving them as background (0)
    
#     # Apply morphological operations to smooth the result
#     # filled = cv2.erode(filled, None, iterations=1)
#     # filled = cv2.dilate(filled, None, iterations=1)

#     return cv2.bitwise_not(filled)

# def detect_and_fill(img):
#     # Find the outer contours in the binary image (using cv2.RETR_EXTERNAL)
#     contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Create a blank image with the same dimensions as the original image
#     filled_img = np.zeros(img.shape[:2], dtype=np.uint8)
#     #filled_img = cv2.bitwise_not(filled_img)

#     # Fill the outer contour with black color
#     cv2.drawContours(filled_img, contours, -1, 255, cv2.FILLED)

#     # # Find contours with hierarchy, this time use cv2.RETR_TREE
#     # contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#     # # Iterate over the contours and their hierarchies
#     # for i, contour in enumerate(contours):
#     #     # Check if the contour has no child
#     #     if hierarchy[0][i][2] < 0:
#     #         # If contour has no child, fill the contour with white color
#     #         cv2.drawContours(filled_img, [contour], -1, 0, cv2.FILLED)
    
#     # Apply morphological operations to smooth the result
#     filled_img = cv2.erode(filled_img, None, iterations=1)
#     #filled_img = cv2.dilate(filled_img, None, iterations=1)

#     return filled_img

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

def correct_curve(image, polynomial_degree=2, min_curve_threshold=2.0):
    """
    Correct curved text (text following a curved baseline) to make it appear straight.
    
    This function detects the baseline curve of the text and applies a transformation
    to straighten it. It works by:
    1. Detecting the bottom-most pixels of characters (baseline)
    2. Fitting a polynomial to the baseline curve
    3. Applying a vertical remapping to straighten the curve
    
    Args:
        image: Binary image (text should be white/255, background black/0)
        polynomial_degree: Degree of polynomial to fit to baseline (default: 2 for quadratic)
                          Higher degrees can handle more complex curves but may overfit
        min_curve_threshold: Minimum curve deviation (in pixels) to apply correction.
                            If the detected curve is less than this, no correction is applied.
    
    Returns:
        Image with corrected curve (straightened baseline)
    """
    # Ensure binary image
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if image.max() <= 1:
        image = (image * 255).astype(np.uint8)
    
    h, w = image.shape
    
    # Detect baseline by finding bottom-most white pixels in each column
    # For each column, find the row with the bottom-most text pixel
    baseline_points = []
    
    for x in range(w):
        # Find bottom-most white pixel in this column
        column = image[:, x]
        # Find all white pixels in this column
        white_pixels = np.where(column > 127)[0]
        
        if len(white_pixels) > 0:
            # Get the bottom-most white pixel (maximum row index)
            bottom_y = np.max(white_pixels)
            baseline_points.append((x, bottom_y))
    
    if len(baseline_points) < polynomial_degree + 2:
        # Not enough points to fit a polynomial, return original
        print("Not enough baseline points detected, skipping curve correction")
        return image
    
    # Extract x and y coordinates
    x_coords = np.array([p[0] for p in baseline_points])
    y_coords = np.array([p[1] for p in baseline_points])
    
    # Fit polynomial to baseline
    # Use numpy's polyfit to fit a polynomial
    coeffs = np.polyfit(x_coords, y_coords, polynomial_degree)
    poly_func = np.poly1d(coeffs)
    
    # Evaluate polynomial at all x positions
    x_all = np.arange(w)
    baseline_curve = poly_func(x_all)
    
    # Calculate the mean baseline position (target straight line)
    mean_baseline = np.mean(baseline_curve)
    
    # Calculate curve deviation
    max_deviation = np.max(np.abs(baseline_curve - mean_baseline))
    
    if max_deviation < min_curve_threshold:
        print(f"Curve deviation ({max_deviation:.2f}px) below threshold, skipping correction")
        return image
    
    print(f"Detected curve with max deviation: {max_deviation:.2f}px")
    
    # Create remapping arrays for cv2.remap
    # We'll map from the corrected (straight) image back to the original (curved) image
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    
    for y in range(h):
        for x in range(w):
            # In the corrected image, x stays the same
            map_x[y, x] = x
            
            # Calculate where this pixel should come from in the original image
            # The baseline at this x position in the original image is at baseline_curve[x]
            # The target baseline in the corrected image is at mean_baseline
            # So we need to shift by the difference
            
            # Calculate the offset from the target baseline
            offset_from_target = y - mean_baseline
            
            # In the original image, this offset corresponds to:
            # original_y = baseline_curve[x] + offset_from_target
            original_y = baseline_curve[x] + offset_from_target
            
            # Clamp to valid range
            original_y = max(0, min(h - 1, original_y))
            map_y[y, x] = original_y
    
    # Apply remapping transformation
    # This maps from corrected coordinates back to original coordinates
    corrected = cv2.remap(image, map_x, map_y, 
                         interpolation=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_REPLICATE)
    
    return corrected

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
