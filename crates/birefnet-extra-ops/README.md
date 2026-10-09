# birefnet-extra-ops

[![Crates.io](https://img.shields.io/crates/v/birefnet-extra-ops.svg)](https://crates.io/crates/birefnet-extra-ops)
[![Documentation](https://docs.rs/birefnet-extra-ops/badge.svg)](https://docs.rs/birefnet-extra-ops)

**Additional operations and extensions for the Burn deep learning framework**

## Implemented Features

- ✅ **DropPath**: Stochastic depth for regularization in transformer models
- ✅ **`trunc_normal`**: Proper weight initialization with truncated normal distribution
- ✅ **ErfInv**: Inverse error function for statistical computations

## Core Operations

### Regularization

- **`DropPath`**: Stochastic depth implementation
  - Randomly drops entire paths during training
  - Improves model generalization
  - Compatible with transformer architectures
  - Proper training/inference mode handling

### Weight Initialization

- **`trunc_normal`**: Advanced weight initialization
  - Truncated normal distribution sampling
  - Configurable bounds and standard deviation
  - Better convergence properties than standard normal
  - PyTorch-compatible initialization

### Mathematical Functions

- **`ErfInv`**: Inverse error function
  - High-precision implementation
  - Required for advanced statistical operations
  - Used in specialized initialization schemes

## Usage

### DropPath for Regularization

```rust
use birefnet_extra_ops::DropPathConfig;
use burn::tensor::{Device, Tensor};

// Create DropPath with 10% drop probability
let drop_path = DropPathConfig::new().with_drop_prob(0.1).init();

// Active only on an autodiff (training) device and until `valid()` clears its flag
let device = Device::flex().autodiff();
let output = drop_path.forward(Tensor::<3>::ones([4, 49, 96], &device));
```

### Weight Initialization

```rust
use birefnet_extra_ops::trunc_normal;
use burn::tensor::{Device, Tensor};

let device = Device::flex();

// Fill a tensor with N(0, 0.02^2) samples truncated to [-0.04, 0.04]
let weights = trunc_normal(Tensor::<2>::zeros([768, 768], &device), 0.0, 0.02, -0.04, 0.04);
```

### Statistical Functions

```rust
use birefnet_extra_ops::erfinv;
use burn::tensor::{Device, Tensor};

// Compute the inverse error function element-wise
let result = erfinv(Tensor::<1>::from_floats([0.5], &Device::flex())); // ~0.477
```

## Integration with Burn Framework

All operations are implemented as native Burn modules:

- Full support for automatic differentiation
- Backend-agnostic implementations
- Proper module serialization/deserialization
- Integration with Burn's training loop

## Key Features

### Training/Inference Modes

- DropPath follows Burn's `Dropout` contract: it is active only while its `Param<Flag>`
  training flag is set (cleared by `Module::valid()` and `Module::freeze()`) and its input
  lives on an autodiff device
- Proper gradient handling during training

### Backend Compatibility

- Works with every Burn backend; the device is chosen at run time
- Tensor operations using Burn framework
- Memory-aware implementations

### PyTorch-Inspired Design

- Weight initialization follows PyTorch patterns
- DropPath implementation based on torchvision
- Similar numerical approach to PyTorch

## Usage in BiRefNet

These operations are used throughout the BiRefNet architecture:

### Transformer Blocks

- **DropPath**: Regularization in Swin Transformer layers
- **`trunc_normal`**: Weight initialization for attention layers

### Model Architecture

- **ErfInv**: Advanced initialization schemes

## Mathematical Accuracy

All operations are carefully implemented:

- DropPath: Configurable probability scaling
- `trunc_normal`: Truncated sampling implementation
- ErfInv: Inverse error function implementation

## License

MIT OR Apache-2.0