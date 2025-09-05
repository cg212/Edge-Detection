import numpy as np
from skimage import io, color, feature
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


# Canny edge detection function
def apply_canny_edge_detection(image, sigma=1.0):
    """
    Apply Canny edge detection to an image.

    Parameters:
    image: Greyscale image
    sigma: Smoothing parameter for noise reduction

    Returns:
    edges: Binary edge map of the image
    """
    edges = feature.canny(image, sigma=sigma)
    return edges


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

# Load images and convert to greyscale and binarize ground truth edge maps
images = {
    name: {
        "image": color.rgb2gray(io.imread(files["image"])),
        "ground_truth": color.rgb2gray(io.imread(files["ground_truth"])) > 0,  # Binary ground truth
    }
    for name, files in image_files.items()
}

# Applying Canny edge detection
results = {}  # Store detected edges
metrics = {}  # Store evaluation metrics
for name, data in images.items():  # Extracting image and ground truth for the current dataset
    image = data["image"]
    ground_truth = data["ground_truth"]
    detected_edges = apply_canny_edge_detection(image, sigma=1.0)  # Applying Canny edge detection
    results[name] = detected_edges  # Storing edge map

    # Calculating quantitative metrics
    precision = precision_score(ground_truth.ravel(), detected_edges.ravel())
    recall = recall_score(ground_truth.ravel(), detected_edges.ravel())
    f1 = f1_score(ground_truth.ravel(), detected_edges.ravel())
    metrics[name] = {"Precision": precision, "Recall": recall, "F1-Score": f1}

# Visualise results with overlays
fig, axes = plt.subplots(len(images), 4, figsize=(20, 15))

for row, (name, data) in enumerate(images.items()):
    image = data["image"]
    ground_truth = data["ground_truth"]
    detected_edges = results[name]

    # Create colour-coded overlay: Red (detected edges), White (ground truth), Yellow (overlapping edges)
    overlay = np.zeros((*ground_truth.shape, 3), dtype=np.float32)  # Initialise overlay with 3 channels (RGB)
    edges = detected_edges  # Detected edges
    overlay[..., 0] = edges  # Red for detected edges
    overlay[..., 1] = edges & ground_truth  # Yellow for overlapping edges
    overlay[..., 2] = 0  
    overlay[ground_truth] = [1.0, 1.0, 1.0]  # White for ground truth

    # Show original image
    axes[row, 0].imshow(image, cmap="gray")
    axes[row, 0].set_title(f"{name} - Original")
    axes[row, 0].axis("off")

    # Show ground truth
    axes[row, 1].imshow(ground_truth, cmap="gray")
    axes[row, 1].set_title(f"{name} - Ground Truth")
    axes[row, 1].axis("off")

    # Show detected edges
    axes[row, 2].imshow(detected_edges, cmap="gray")
    axes[row, 2].set_title(f"{name} - Canny Edge Detection")
    axes[row, 2].axis("off")

    # Show colored overlay on black background
    axes[row, 3].imshow(overlay)
    axes[row, 3].set_title(f"{name} - Overlay")
    axes[row, 3].axis("off")

plt.tight_layout()
plt.show()

# Print metrics
for name, metric in metrics.items():
    print(f"{name}:")
    print(f"  Precision: {metric['Precision']:.2f}")
    print(f"  Recall: {metric['Recall']:.2f}")
    print(f"  F1-Score: {metric['F1-Score']:.2f}")
