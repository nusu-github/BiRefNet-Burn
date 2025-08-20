# birefnet-util

[![Crates.io](https://img.shields.io/crates/v/birefnet-util.svg)](https://crates.io/crates/birefnet-util)
[![Documentation](https://docs.rs/birefnet-util/badge.svg)](https://docs.rs/birefnet-util)

**Utility functions for BiRefNet including image processing, weight management, and mathematical operations**

## Implemented Features

### ✅ Image Processing

- **Image loading**: Support for multiple formats (JPEG, PNG, BMP, TIFF, WebP)
- **ImageNet normalization**: Proper preprocessing for model input
- **Tensor conversion**: Conversion between image formats and tensors
- **Device-aware operations**: GPU/CPU compatible tensor operations
- **Mask application**: Apply segmentation masks to images

### ✅ Weight Management

- **PyTorch compatibility**: Direct loading of PyTorch checkpoint files
- **Model mapping**: Intelligent weight mapping between PyTorch and Burn formats
- **Managed models**: Automatic model downloading and caching
- **Weight source handling**: Local files and remote model management

### ✅ Mathematical Operations

- **Distance transforms**: Euclidean distance computation for morphology
- **Array operations**: Tensor manipulation and processing utilities
- **Morphological operations**: Basic erosion, dilation, and boundary detection
- **Filtering operations**: Image filtering and enhancement utilities

### ✅ Foreground Refinement

- **Core refinement**: Advanced foreground enhancement algorithms
- **Batch processing**: Processing of multiple images
- **Parameter tuning**: Configurable refinement parameters

## Core Modules

### Image Processing (`image.rs`)

- **`ImageUtils`**: Main image processing utilities
  - `load_image`: Load images with device placement
  - `apply_imagenet_normalization`: Standard preprocessing
  - `tensor_to_dynamic_image`: Convert tensors back to images
  - `apply_mask`: Apply segmentation masks

### Weight Management (`weights.rs`)

- **`ModelLoader`**: PyTorch checkpoint loading
- **`ManagedModel`**: Automatic model management
- **`WeightSource`**: Local and remote weight handling
- **`BiRefNetWeightLoading`**: Model-specific weight loading

### Mathematical Utilities

- **`array_ops.rs`**: Tensor array operations
- **`distance.rs`**: Distance transform computations
- **`morphology.rs`**: Morphological image operations
- **`filters.rs`**: Image filtering and enhancement

### Foreground Refinement (`foreground_refiner.rs`)

- **`refine_foreground_core`**: Core refinement algorithm
- **`refine_foreground`**: Single image refinement
- **`refine_foreground_batch`**: Batch processing

## Usage

### Image Processing

```rust
use birefnet_util::image::ImageUtils;

// Load and preprocess image
let image = ImageUtils::load_image("path/to/image.jpg", & device) ?;
let normalized = ImageUtils::apply_imagenet_normalization(image) ?;

// Convert back to image
let output_image = ImageUtils::tensor_to_dynamic_image(tensor, false) ?;
```

### Weight Loading

```rust
use birefnet_util::weights::{ManagedModel, ModelLoader};

// Load pretrained model
let managed_model = ManagedModel::from_pretrained("General")?;
let model = BiRefNet::from_managed_model(&managed_model, &device)?;
```

### Foreground Refinement

```rust
use birefnet_util::foreground_refiner::refine_foreground_core;

// Refine foreground with mask
let refined = refine_foreground_core(image, mask, radius);
```

### Mathematical Operations

```rust
use birefnet_util::{distance, morphology, filters};

// Distance transform
let distance_map = distance::euclidean_distance_transform(binary_mask);

// Morphological operations
let eroded = morphology::erosion(mask, kernel_size);
let dilated = morphology::dilation(mask, kernel_size);
```

## Key Features

### Device Compatibility

- Automatic device detection and tensor placement
- CPU and GPU backend support
- Burn framework tensor operations

### Format Support

- Multiple image formats (JPEG, PNG, BMP, TIFF, WebP)
- Tensor formats compatible with Burn framework
- PyTorch checkpoint format support

### Implementation

- Tensor operations using Burn framework
- Batch processing support
- Cross-platform compatibility

## Integration

This crate provides core utilities used by:

- [`birefnet`](../birefnet): Main CLI application
- [`birefnet-model`](../birefnet-model): Model weight loading
- [`birefnet-inference`](../birefnet-inference): Image preprocessing and postprocessing
- [`birefnet-train`](../birefnet-train): Dataset loading and augmentation

## Dependencies

- **Burn framework**: Core tensor operations
- **Image crate**: Image format support
- **Candle**: PyTorch checkpoint loading
- **Anyhow**: Error handling

## License

MIT OR Apache-2.0