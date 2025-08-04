# VGG Test Data Directory

This directory contains raw test data for VGG backbone tests.

## File Format

Test data files should be saved in the following format:

- `.npy` files for NumPy arrays (for Python reference data)
- `.npz` files for multiple arrays in one file
- `.json` files for metadata

## Directory Structure

```
test_data/
├── README.md
├── inputs/      # Input tensors for testing
├── outputs/     # Expected output tensors
└── metadata/    # Test configuration and metadata
```

## VGG Architecture

VGG backbone extracts features at 4 different scales:

- conv1: Early convolution layers
- conv2: Mid-level features
- conv3: Higher-level features
- conv4: High-level semantic features

Each stage outputs feature maps at different spatial resolutions.