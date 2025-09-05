# Edge Detection Methods Comparison and Analysis

This project implements and compares various edge detection algorithms on fluorescent cell images and applies Canny edge detection with Hough transform for line detection in architectural images.

## Project Overview

This computer vision project consists of three main tasks:
1. **Task 1**: Comparison of classical edge detection methods (Robinson, Sobel, Prewitt, Kirsch, and Gaussian)
2. **Task 2**: Canny edge detection algorithm evaluation
3. **Task 3**: Combined Canny edge detection and Hough transform for line detection

## File Structure

```
├── Task1.py                    # Multiple edge detection methods comparison
├── Task2.py                    # Canny edge detection implementation
├── Task3.py                    # Canny + Hough transform for line detection
├── README.md                   # Project documentation
└── Data/
    ├── cells/
    │   ├── 9343 AM.bmp         # Cell image 1
    │   ├── 9343 AM Edges.bmp   # Ground truth edges 1
    │   ├── 10905 JL.bmp        # Cell image 2
    │   ├── 10905 JL Edges.bmp  # Ground truth edges 2
    │   ├── 43590 AM.bmp        # Cell image 3
    │   └── 43590 AM Edges.bmp  # Ground truth edges 3
    └── Bhamimage.jpeg          # Architectural image for line detection
```

## Dependencies

Install the required packages using pip:

```bash
pip install numpy matplotlib scikit-image scipy scikit-learn
```

### Required Libraries:
- `numpy`: Numerical computations
- `matplotlib`: Visualization and plotting
- `scikit-image`: Image processing algorithms
- `scipy`: Scientific computing (convolution operations)
- `scikit-learn`: Machine learning metrics

## Task Descriptions

### Task 1: Multiple Edge Detection Methods (`Task1.py`)

**Objective**: Compare five edge detection methods on fluorescent cell images.

**Methods Implemented**:
- **Robinson**: Compass mask edge detection
- **Sobel**: Gradient-based edge detection
- **Prewitt**: Simple gradient operator
- **Kirsch**: Compass operator with different kernel
- **Gaussian**: Gaussian smoothing + Sobel edge detection

**Features**:
- Custom kernel implementations for each method
- 5x5 Gaussian kernel generation with configurable sigma
- Quantitative evaluation using Precision, Recall, and F1-Score
- Color-coded overlay visualization (Red: detected edges, White: ground truth, Yellow: correct detections)
- Comparative analysis across three cell images

**Key Results**:
- Gaussian method performed best overall with highest F1-scores
- High precision (0.51-0.77) but low recall (0.00-0.12) across methods
- Gaussian method better at detecting faint edges in noisy conditions

### Task 2: Canny Edge Detection (`Task2.py`)

**Objective**: Evaluate Canny edge detection algorithm and compare with Task 1 results.

**Features**:
- Multi-step Canny algorithm implementation
- Gaussian smoothing for noise reduction (sigma=1.0)
- Gradient computation and non-maximum suppression
- Double thresholding for edge linking
- Quantitative metrics comparison with ground truth
- Visual overlay analysis

**Key Results**:
- Better noise suppression than classical methods
- Higher precision but lower recall than Gaussian method
- Excellent for detecting prominent edges, struggles with faint details
- F1-scores: 0.07 (9343 AM), 0.02 (10905 JL), 0.28 (43590 AM)

### Task 3: Line Detection (`Task3.py`)

**Objective**: Detect straight lines in architectural images using Canny edge detection + Hough transform.

**Features**:
- Canny edge detection with increased sigma (2.0) for better noise handling
- Hough transform for line detection
- Configurable threshold (150) and peak limit (10)
- Three-panel visualization: Original → Edges → Detected Lines

**Key Results**:
- Successfully detected major structural elements (clock tower, building edges)
- Effective noise suppression from trees and grass textures
- Clear identification of vertical and horizontal architectural lines
- Minimal false positive detections

## Usage Instructions

### Running Task 1 (Multiple Edge Detection)
```bash
python Task1.py
```
**Output**: 
- Two visualization windows showing edge detection results and color overlays
- Console output with quantitative metrics for each method and image

### Running Task 2 (Canny Edge Detection)
```bash
python Task2.py
```
**Output**:
- Visualization window with original, ground truth, detected edges, and overlay
- Console output with precision, recall, and F1-scores

### Running Task 3 (Line Detection)
```bash
python Task3.py
```
**Output**:
- Single visualization window with three panels showing the line detection process

## Configuration Options

### Task 1 & 2: Adjustable Parameters
- **Threshold multiplier**: Currently set to 0.3 (30% of maximum edge strength)
- **Gaussian sigma**: Controls smoothing strength (default: 1.0)
- **Kernel size**: 5x5 Gaussian kernel (configurable in `gaussian_kernel()`)

### Task 3: Adjustable Parameters
- **Canny sigma**: Set to 2.0 for optimal noise-edge balance
- **Hough threshold**: Set to 150 to filter weak lines
- **Peak limit**: Limited to 10 most prominent lines

## Evaluation Metrics

The project uses three key metrics for quantitative evaluation:

- **Precision**: Proportion of detected edges that exist in ground truth
- **Recall**: Proportion of ground truth edges correctly detected
- **F1-Score**: Harmonic mean of precision and recall

## Key Findings

1. **Gaussian smoothing** significantly improves edge detection in noisy conditions
2. **Classical methods** (Robinson, Sobel, Prewitt, Kirsch) perform similarly with high precision but low recall
3. **Canny algorithm** excels at producing clean, sharp edge maps but may miss faint details
4. **Combined Canny + Hough** effectively detects structural lines while suppressing texture noise

## File Path Configuration

**File Paths**: The current file paths in the scripts are:

```python
# Task 1 & Task 2 - Cell images and ground truth
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

# Task 3 - Architectural image
image_path = "C:/Documents/Computer Science/Computer Vision and Imaging/Final Assignment/Data 06 30213/Bhamimage.jpeg"
```

**To run on your system**: Update these paths to match your local directory structure. For example:
```python
# Example relative paths if files are in the same directory structure
image_files = {
    "9343 AM": {
        "image": "./Data 06 30213/cells/9343 AM.bmp",
        "ground_truth": "./Data 06 30213/cells/9343 AM Edges.bmp",
    },
    # ... other images
}
```

## Troubleshooting

**Common Issues**:
- **File not found errors**: Verify image paths are correct
- **Import errors**: Ensure all dependencies are installed
- **Display issues**: Make sure matplotlib backend supports GUI (for visualization)

**Performance Notes**:
- Processing time varies with image size and complexity
- Close visualization windows to proceed to next results
- Metrics are printed to console after all visualizations are closed
