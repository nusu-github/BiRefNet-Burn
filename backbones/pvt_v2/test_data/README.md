# Test Data Directory

This directory contains raw test data for PVT v2 backbone tests.

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