import numpy as np
from skimage import io, filters, color
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score  # For quantitative metrics

'''Note: Once one visualization is closed, the next one will open.
After all visualizations are closed, the quantitative metrics will be shown in the terminal.'''

# Defining edge detection kernels for the different methods
kernels = {
    "Robinson": {
        "horizontal": np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]),
        "vertical": np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]),
    },
    "Sobel": {
        "horizontal": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
        "vertical": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
    },
    "Prewitt": {
        "horizontal": np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
        "vertical": np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
    },
    "Kirsch": {
        "horizontal": np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
        "vertical": np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),
    },
}

# Generating 5x5 Gaussian kernel
def gaussian_kernel(size=5, sigma=1):  # Sigma controls the standard deviation of the Gaussian distribution
    kernel = np.zeros((size, size))
    center = size // 2
    for x in range(size):
        for y in range(size):
            diff = (x - center) ** 2 + (y - center) ** 2
            kernel[x, y] = np.exp(-diff / (2 * sigma ** 2))
    return kernel / np.sum(kernel)  # Normalise to ensure sum is 1

# Applying edge detection
def apply_edge_detection(image, kernel_h, kernel_v):
    """
    Apply edge detection using horizontal and vertical kernels.
    """
    edge_h = convolve(image, kernel_h)  # Convolve with horizontal kernel
    edge_v = convolve(image, kernel_v)  # Convolve with vertical kernel
    edge_magnitude = np.sqrt(edge_h ** 2 + edge_v ** 2)  # Compute edge magnitude
    return edge_magnitude

# Load and preprocess images
image_files = {
    "9343 AM": {
        "image": "C:/Documents/Computer Science/Computer Vision and Imaging/Final Assignment/Data 06 30213/cells/9343 AM.bmp",
        "ground_truth": "C:/Documents/Computer Science/Computer Vision and Imaging/Final Assignment/Data 06 30213/cells/9343 AM Edges.bmp",
    },
    "10905 JL": {
        "image": "C:/Documents/Computer Science/Computer Vision and Imaging/Final Assignment/Data 06 30213/cells/10905 JL.bmp",
        "ground_truth": "C:/Documents/Computer Science/Computer Vision and Imaging/Final Assignment/Data 06 30213/cells/10905 JL Edges.bmp",
    },
    "43590 AM": {
        "image": "C:/Documents/Computer Science/Computer Vision and Imaging/Final Assignment/Data 06 30213/cells/43590 AM.bmp",
        "ground_truth": "C:/Documents/Computer Science/Computer Vision and Imaging/Final Assignment/Data 06 30213/cells/43590 AM Edges.bmp",
    },
}

# Converting images to greyscale and binarize ground truth maps
images = {
    name: {
        "image": color.rgb2gray(io.imread(files["image"])),
        "ground_truth": color.rgb2gray(io.imread(files["ground_truth"])) > 0,  # Binary ground truth
    }
    for name, files in image_files.items()
}

# Generating Gaussian kernel
gaussian_k = gaussian_kernel(size=5, sigma=1)

# Applying edge detection methods and calculate metrics
results = {}
metrics = {}  # Store metrics for each method
methods = list(kernels.keys()) + ["Gaussian"]

for name, data in images.items():
    image = data["image"]
    ground_truth = data["ground_truth"]
    results[name] = {}
    metrics[name] = {}

    # Applying each edge detection method
    for method, kernel in kernels.items():
        edge_map = apply_edge_detection(image, kernel["horizontal"], kernel["vertical"])
        binary_edges = edge_map > np.max(edge_map) * 0.3  # Thresholding
        results[name][method] = edge_map

        # Computing precision, recall, and F1-score
        precision = precision_score(ground_truth.flatten(), binary_edges.flatten())
        recall = recall_score(ground_truth.flatten(), binary_edges.flatten())
        f1 = f1_score(ground_truth.flatten(), binary_edges.flatten())
        metrics[name][method] = {"Precision": precision, "Recall": recall, "F1-Score": f1}

    smoothed_image = convolve(image, gaussian_k)
    gaussian_edges = filters.sobel(smoothed_image)
    binary_gaussian = gaussian_edges > np.max(gaussian_edges) * 0.3
    results[name]["Gaussian"] = gaussian_edges

    # Compute metrics for Gaussian
    precision = precision_score(ground_truth.flatten(), binary_gaussian.flatten())
    recall = recall_score(ground_truth.flatten(), binary_gaussian.flatten())
    f1 = f1_score(ground_truth.flatten(), binary_gaussian.flatten())
    metrics[name]["Gaussian"] = {"Precision": precision, "Recall": recall, "F1-Score": f1}

# Visualise edge detection results
fig, axes = plt.subplots(len(images), len(methods) + 2, figsize=(20, 15))  # +2 for original and ground truth
for row, (name, data) in enumerate(images.items()):
    axes[row, 0].imshow(data["image"], cmap="gray")
    axes[row, 0].set_title(f"{name} - Original")
    axes[row, 0].axis("off")

    axes[row, 1].imshow(data["ground_truth"], cmap="gray")
    axes[row, 1].set_title(f"{name} - Ground Truth")
    axes[row, 1].axis("off")

    for col, method in enumerate(methods):
        enhanced_map = results[name][method] / np.max(results[name][method])  # Normalize
        axes[row, col + 2].imshow(enhanced_map, cmap="gray")
        axes[row, col + 2].set_title(f"{method}")
        axes[row, col + 2].axis("off")

plt.tight_layout()
plt.show()

# Visualise color-coded overlays with ground truth in white
fig, axes = plt.subplots(len(images), len(methods), figsize=(20, 15))
for row, (name, data) in enumerate(images.items()):
    ground_truth = data["ground_truth"]
    for col, method in enumerate(methods):
        overlay = np.zeros((*ground_truth.shape, 3), dtype=np.float32)  # Create an RGB image
        edges = results[name][method] > np.max(results[name][method]) * 0.3  # Threshold edges
        
        # Red for detected edges
        overlay[..., 0] = edges  
        
        # Yellow for overlapping edges (red + green)
        overlay[..., 1] = edges & ground_truth  
        
        # White for ground truth (set all channels to 1 where ground truth exists)
        overlay[ground_truth] = [1.0, 1.0, 1.0]  

        # Display the overlay
        axes[row, col].imshow(overlay)
        axes[row, col].set_title(f"{name} - {method} Overlay")
        axes[row, col].axis("off")

plt.tight_layout()
plt.show()

# Display metrics
for name, method_metrics in metrics.items():
    print(f"\nMetrics for {name}:")
    for method, values in method_metrics.items():
        print(f"  {method}: Precision: {values['Precision']:.2f}, Recall: {values['Recall']:.2f}, F1-Score: {values['F1-Score']:.2f}")
