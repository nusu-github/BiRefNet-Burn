# birefnet-extra-ops

[![Crates.io](https://img.shields.io/crates/v/birefnet-extra-ops.svg)](https://crates.io/crates/birefnet-extra-ops)
[![Documentation](https://docs.rs/birefnet-extra-ops/badge.svg)](https://docs.rs/birefnet-extra-ops)

**Additional operations and extensions for the Burn deep learning framework**

## Implemented Features

- ✅ **DropPath**: Stochastic depth for regularization in transformer models
- ✅ **TruncatedNormal**: Proper weight initialization with truncated normal distribution
- ✅ **Identity**: Pass-through operation for skip connections and debugging
- ✅ **ErfInv**: Inverse error function for statistical computations

## Core Operations

### Regularization

- **`DropPath`**: Stochastic depth implementation
  - Randomly drops entire paths during training
  - Improves model generalization
  - Compatible with transformer architectures
  - Proper training/inference mode handling

### Weight Initialization

- **`TruncatedNormal`**: Advanced weight initialization
  - Truncated normal distribution sampling
  - Configurable bounds and standard deviation
  - Better convergence properties than standard normal
  - PyTorch-compatible initialization

### Utility Operations

- **`Identity`**: Pass-through operation
  - Zero-cost abstraction for skip connections
  - Useful for conditional computation paths
  - Debugging and model architecture exploration

### Mathematical Functions

- **`ErfInv`**: Inverse error function
  - High-precision implementation
  - Required for advanced statistical operations
  - Used in specialized initialization schemes

## Usage

### DropPath for Regularization

```rust
use birefnet_extra_ops::DropPath;

// Create DropPath with 10% drop probability
let drop_path = DropPathConfig::new(0.1).init();

// Apply during forward pass (automatically disabled during inference)
let output = drop_path.forward(input);
```

### Weight Initialization

```rust
use birefnet_extra_ops::TruncatedNormal;

// Initialize weights with truncated normal distribution
let init = TruncatedNormalConfig::new()
.with_mean(0.0)
.with_std(0.02)
.with_bounds(- 0.04, 0.04);

// Apply to tensor
let initialized_weights = init.init_tensor(shape, & device);
```

### Identity Operation

```rust
use birefnet_extra_ops::Identity;

// Create identity operation
let identity = Identity::new();

// Pass-through operation
let output = identity.forward(input); // output == input
```

### Statistical Functions

```rust
use birefnet_extra_ops::erfinv::erfinv;

// Compute inverse error function
let result = erfinv(0.5); // Returns ~0.477
```

## Integration with Burn Framework

All operations are implemented as native Burn modules:

- Full support for automatic differentiation
- Backend-agnostic implementations
- Proper module serialization/deserialization
- Integration with Burn's training loop

## Key Features

### Training/Inference Modes

- Operations automatically adapt to training vs inference mode
- DropPath disables during inference
- Proper gradient handling during training

### Backend Compatibility

- Works with all Burn backends (ndarray, WebGPU, CUDA)
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
- **TruncatedNormal**: Weight initialization for attention layers

### Model Architecture

- **Identity**: Skip connections and conditional paths
- **ErfInv**: Advanced initialization schemes

## Mathematical Accuracy

All operations are carefully implemented:

- DropPath: Configurable probability scaling
- TruncatedNormal: Truncated sampling implementation
- ErfInv: Inverse error function implementation
- Identity: Zero computational overhead

## License

MIT OR Apache-2.0