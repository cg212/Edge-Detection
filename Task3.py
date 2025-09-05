import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, feature, transform

# Load and preprocess the image
image_path = "C:/Documents/Computer Science/Computer Vision and Imaging/Final Assignment/Data 06 30213/Bhamimage.jpeg"
original_image = io.imread(image_path)
grayscale_image = color.rgb2gray(original_image)

# Applying Canny edge detection with a refined sigma value
# sigma = 2.0 helps to balance noise reduction and edge preservation
edges = feature.canny(grayscale_image, sigma=2.0)

# Applying Hough transform to detect lines
# Returns the Hough space, angles, and distances corresponding to possible lines
hough_space, angles, distances = transform.hough_line(edges)

# Extracting prominent lines using Hough line peaks
# Threshold set to 150 to filter weak lines
# num_peaks set to 10 to limit detected lines
hough_peaks = transform.hough_line_peaks(
    hough_space, angles, distances, threshold=150, num_peaks=10
)

# Visualise the results
fig, axes = plt.subplots(1, 3, figsize=(24, 10))

# Original image
axes[0].imshow(original_image)
axes[0].set_title("Original Image")
axes[0].axis("off")

# Canny edge-detected image
axes[1].imshow(edges, cmap="gray")
axes[1].set_title("Canny Edge Detection")
axes[1].axis("off")

# Overlay detected lines on the original image
axes[2].imshow(original_image, extent=[0, original_image.shape[1], original_image.shape[0], 0])
for _, angle, dist in zip(*hough_peaks):
    y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
    y1 = (dist - original_image.shape[1] * np.cos(angle)) / np.sin(angle)
    axes[2].plot((0, original_image.shape[1]), (y0, y1), '-r', linewidth=2)
axes[2].set_xlim([0, original_image.shape[1]])
axes[2].set_ylim([original_image.shape[0], 0])
axes[2].set_title("Detected Lines (Hough Transform)")
axes[2].axis("off")

plt.tight_layout()
plt.show()
