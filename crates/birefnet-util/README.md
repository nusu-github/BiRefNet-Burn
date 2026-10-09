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

- **PyTorch compatibility**: Direct loading of PyTorch (`.pt`/`.pth`) and SafeTensors checkpoints through `burn::store`
- **Burnpack**: Loading of Burn's native `.bpk` files (written by `Module::save_file`, e.g. training runs)
- **Model mapping**: Intelligent weight mapping between PyTorch and Burn formats
- **Precision-aware**: Loaded float weights are converted to the device's default float dtype
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
use birefnet_util::{apply_imagenet_normalization, load_image, tensor_to_dynamic_image};
use burn::tensor::Device;

let device = Device::flex();

// Load and preprocess image: `[1, 3, height, width]` with values in [0, 1]
let image = load_image("path/to/image.jpg", &device)?;
let normalized = apply_imagenet_normalization(image.clone())?;

// Convert back to image
let output_image = tensor_to_dynamic_image(image, false)?;
```

### Weight Loading

```rust
use birefnet_model::BiRefNet;
use birefnet_util::{BiRefNetWeightLoading, ManagedModel};
use burn::tensor::Device;

let device = Device::flex();

// Load pretrained model (downloads the SafeTensors weights from the Hugging Face Hub)
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
let kernel = morphology::StructuringElement::disk(3, &device);
let eroded = morphology::erosion(mask.clone(), &kernel);
let dilated = morphology::dilation(mask, &kernel);
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