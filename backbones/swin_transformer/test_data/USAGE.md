# Swin Transformer Test Data Usage

## Overview

This directory contains test data for verifying the Swin Transformer Rust implementation against the Python reference implementation.

## Generating Test Data

1. Set up the Python BiRefNet environment:
   ```bash
   cd /path/to/BiRefNet-proj/BiRefNet
   conda activate birefnet
   ```

2. Run the test data generation script:
   ```bash
   cd /path/to/BiRefNet-proj/BiRefNet-Burn/backbones/swin_transformer
   python generate_test_data.py
   ```

This will create:

- `inputs/`: Input tensors as .npy files
- `outputs/`: Expected output tensors as .npy files
- `metadata/`: JSON files with test configuration

## Running Tests

1. With test data present:
   ```bash
   cargo test --features ndarray
   ```

2. Without test data:
    - Tests will skip array comparison tests
    - Shape validation tests will still run

## Test Types

1. **Shape Tests**: Always run, verify output tensor dimensions
2. **Array Comparison Tests**: Run only when test data is present, verify numerical accuracy

## Adding New Tests

1. Create a new test data generation function in `generate_test_data.py`
2. Add corresponding test function in `src/lib.rs`
3. Use `load_numpy_array_*` functions to load test data
4. Use `assert_tensor_approx_eq` for numerical comparisons

## Tolerance

Default tolerance is 1e-4 for floating-point comparisons. This accounts for:

- Numerical precision differences between PyTorch and Burn
- Different initialization methods
- Backend-specific optimizations