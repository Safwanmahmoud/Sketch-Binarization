# Sketch Binarization

This repository contains an algorithm for binarizing sketch images using local adaptive thresholding and contour processing. The algorithm converts grayscale or color sketches into clean binary images, preserving important sketch lines while removing noise. The algorithm is imune to artifacts like shadows.

## Features
- **Local Adaptive Thresholding**: Uses a sliding window to compute local mean and standard deviation for binarization.
- **Contour Processing**: Detects and approximates contours to refine sketch lines.
- **Configurable Parameters**: Adjust kernel size, threshold factors, and contour filters.

## Requirements
- Python 3.8+
- OpenCV (`opencv-python`)
- NumPy

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sketch-binarization.git
   cd sketch-binarization
   ```
2. Install dependencies:
   ```bash
   pip install opencv-python numpy
   ```

## Usage
1. Place your sketch image in the project directory (e.g., `sketch.png`).
2. Run the provided script:
   ```bash
   python binarize_sketch.py
   ```
3. The output binary image will be saved (e.g., `output.png`).

### Example
```python
import cv2
from binarize_sketch import binarize_sketch

# Load image
image = cv2.imread('sketch.png')

# Binarize sketch
binary_image = binarize_sketch(image, kernel_size=15, min_contour_area=30)

# Save or display result
cv2.imwrite('output.png', binary_image)
cv2.imshow('Binary Sketch', binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## Parameters
- `kernel_size`: Size of the local window (default: 15).
- `std_threshold`: Minimum standard deviation to avoid flat regions (default: 5).
- `mean_factor`: Threshold adjustment factor (default: 0.8).
- `min_contour_area`: Minimum contour area to filter noise (default: 30).

## Algorithm Overview
1. Convert input image to grayscale.
2. Pad image to handle edge pixels.
3. Apply local adaptive thresholding using mean and standard deviation.
4. Detect contours in the binary image.
5. Filter and approximate contours to draw clean sketch lines.

## Example Results
| Input Sketch | Output Binary |
|--------------|---------------|
| ![[input](examples/input.jpeg)](https://github.com/Safwanmahmoud/Sketch-Binarization/blob/main/input.jpeg). | ![[Output](examples/output.png)](https://github.com/Safwanmahmoud/Sketch-Binarization/blob/main/output.png) |

## Contributing
Feel free to submit issues or pull requests for improvements or bug fixes.

## License
MIT License
