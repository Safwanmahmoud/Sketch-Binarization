import cv2
import numpy as np
import matplotlib.pyplot as plt

def binarize_sketch(image, kernel_size=20, std_threshold=5, mean_factor=0.8, min_contour_area=30):
    """
    Binarizes a sketch image using local adaptive thresholding and contour processing.

    Args:
        image: Input image (color or grayscale).
        kernel_size: Size of the kernel for local thresholding (odd number).
        std_threshold: Minimum standard deviation to avoid flat regions.
        mean_factor: Factor to adjust threshold (threshold = mean - factor * std).
        min_contour_area: Minimum contour area to filter small noise.
        epsilon_factor: Factor for contour approximation.

    Returns:
        binary_image: Processed binary image with approximated contours.
    """
    # Convert to grayscale if image is color
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Pad image to handle edges
    pad = kernel_size // 2
    padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

    # Initialize output binary image
    binary = np.zeros_like(image, dtype=np.uint8)

    # Perform local adaptive thresholding
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Extract local window
            window = padded[i:i + kernel_size, j:j + kernel_size]
            
            # Compute local statistics
            mean = np.mean(window)
            std = np.std(window)
            
            # Skip flat regions (low std)
            if std < std_threshold:
                binary[i, j] = 255
                continue
            
            # Apply adaptive threshold
            threshold = mean - mean_factor * std
            binary[i, j] = 255 if image[i, j] > threshold else 0

    # Find and process contours
    contours, _ = cv2.findContours(np.invert(binary), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    binary_image = np.zeros_like(binary)

    for cnt in contours:
        # Filter small contours
        if cv2.contourArea(cnt) < min_contour_area:
            continue
        
        # Approximate contour
        approx = cv2.approxPolyDP(cnt, 2, True)
        
        # Draw contours with sufficient points
        if approx.shape[0] > 2:
            cv2.drawContours(binary_image, [approx], -1, (255, 255, 255), 2)

    return binary_image

# Load an image
img = cv2.imread('yourImage.jpeg', cv2.IMREAD_GRAYSCALE)
bImage = binarize_sketch(img, kernel_size=20, std_threshold=5, mean_factor=0.8, min_contour_area=30)
# Show the result
plt.imshow(np.invert(bImage), cmap='gray')