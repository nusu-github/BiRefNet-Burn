# birefnet-inference

[![Crates.io](https://img.shields.io/crates/v/birefnet-inference.svg)](https://crates.io/crates/birefnet-inference)
[![Documentation](https://docs.rs/birefnet-inference/badge.svg)](https://docs.rs/birefnet-inference)

**Inference engine and post-processing utilities for BiRefNet using the Burn deep learning framework**

## Implemented Features

### ✅ Basic Post-processing

- **Thresholding**: Binary mask generation with configurable thresholds
- **Tensor to image conversion**: Conversion for saving results
- **Basic resizing**: Tensor resizing with device-aware operations

### 🚧 Advanced Post-processing (Placeholder Implementations)

- **Gaussian blur**: Function exists but returns input unchanged (needs convolution implementation)
- **Morphological operations**: Opening and closing operations return input unchanged
  - `morphological_opening`: Placeholder (needs erosion + dilation)
  - `morphological_closing`: Placeholder (needs dilation + erosion)
- **Connected component analysis**: `remove_small_components` returns input unchanged
- **Hole filling**: `fill_holes` returns input unchanged (needs morphological reconstruction)

## Core Components

### Working Functions

- **`apply_threshold`**: Binary thresholding with configurable threshold values
- **`tensor_to_image_data`**: Convert tensors to u8 image data for saving
- **`resize_tensor`**: Resize tensors to target dimensions

### Placeholder Functions (Return Input Unchanged)

- **`gaussian_blur`**: TODO - Implement separable Gaussian convolution
- **`morphological_opening`**: TODO - Implement erosion followed by dilation
- **`morphological_closing`**: TODO - Implement dilation followed by erosion
- **`remove_small_components`**: TODO - Implement connected component labeling
- **`fill_holes`**: TODO - Implement flood fill or morphological reconstruction

### Comprehensive Pipeline

- **`postprocess_mask`**: Complete pipeline combining all operations
  - Works with basic operations (threshold, resize)
  - Advanced operations are placeholders but don't break the pipeline

## Usage

```rust
use birefnet_inference::postprocessing::{
    apply_threshold, postprocess_mask, tensor_to_image_data
};

// Basic thresholding (works)
let binary_mask = apply_threshold(mask, 0.5);

// Convert to image data (works)
let image_data = tensor_to_image_data(mask);

// Comprehensive postprocessing pipeline
let processed_mask = postprocess_mask(
    mask,
    0.5,    // threshold
    5,      // blur_kernel_size (placeholder)
    1.0,    // blur_sigma (placeholder)
    3,      // morphology_kernel_size (placeholder)  
    100,    // min_component_size (placeholder)
    true,   // fill_holes_flag (placeholder)
);
```

## Post-processing Pipeline

### Working Operations

1. **Threshold**: Converts predictions to binary masks
2. **Resize**: Adjusts output to target dimensions

### Placeholder Operations (Don't Modify Input)

1. **Gaussian Blur**: Smoothing for noise reduction
2. **Morphological Opening**: Remove small noise (erosion + dilation)
3. **Morphological Closing**: Fill small gaps (dilation + erosion)
4. **Component Filtering**: Remove small connected components
5. **Hole Filling**: Fill interior holes in masks

## Current Limitations

### Advanced Post-processing Not Implemented

- **Gaussian convolution**: Needs separable kernel implementation
- **Morphological operations**: Requires structuring element operations
- **Connected components**: Needs flood fill or union-find algorithm
- **Hole filling**: Requires morphological reconstruction or border flood fill

### Efficiency Considerations

- Tensor resizing currently converts through image format (inefficient)
- No GPU-optimized post-processing operations
- Memory copies could be reduced with direct tensor operations

## Integration

This crate provides:

- Basic post-processing that works immediately
- Placeholder structure for advanced operations
- Integration points with image saving and display
- Pipeline that doesn't break with placeholder implementations

Works with:

- [`birefnet-model`](../birefnet-model): Model output tensors
- [`birefnet-util`](../birefnet-util): Image conversion utilities

## Future Work

- Implement proper Gaussian blur with separable convolution
- Add morphological operations with configurable structuring elements
- Implement connected component analysis and filtering
- Add hole filling with morphological reconstruction
- Optimize tensor operations for GPU backends

## License

MIT OR Apache-2.0