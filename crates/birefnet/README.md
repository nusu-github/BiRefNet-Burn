# BiRefNet

[![Crates.io](https://img.shields.io/crates/v/birefnet.svg)](https://crates.io/crates/birefnet)
[![Documentation](https://docs.rs/birefnet/badge.svg)](https://docs.rs/birefnet)

**Bilateral Reference Network for high-resolution dichotomous image segmentation**

This is the main BiRefNet crate that provides a unified CLI interface for inference across multiple backends (CPU, WebGPU, CUDA, ROCm, Metal, Vulkan).

## Implemented Features

- ✅ **Cross-platform inference**: Single and batch image processing
- ✅ **Multi-backend support**: CPU (ndarray), GPU (WebGPU, CUDA, ROCm, Metal, Vulkan)
- ✅ **CLI tool**: Complete command line interface for inference
- ✅ **Model management**: Automatic PyTorch weight loading and conversion
- ✅ **Backend information**: Runtime backend detection and reporting

## Installation

### As a CLI tool

```bash
# CPU version (default)
cargo install birefnet

# GPU versions
cargo install birefnet --features wgpu --no-default-features # WebGPU (cross-platform)
cargo install birefnet --features cuda --no-default-features # NVIDIA CUDA
cargo install birefnet --features rocm --no-default-features # AMD ROCm
cargo install birefnet --features metal --no-default-features # Apple Metal
cargo install birefnet --features vulkan --no-default-features # Vulkan
```

### As a library

```toml
[dependencies]
# CPU inference
birefnet = { version = "0.1.0", features = ["inference"] }

# GPU inference examples
birefnet = { version = "0.1.0", features = ["wgpu", "inference"], default-features = false }    # WebGPU
birefnet = { version = "0.1.0", features = ["cuda", "inference"], default-features = false }   # NVIDIA CUDA
birefnet = { version = "0.1.0", features = ["rocm", "inference"], default-features = false }   # AMD ROCm
birefnet = { version = "0.1.0", features = ["metal", "inference"], default-features = false }  # Apple Metal
birefnet = { version = "0.1.0", features = ["vulkan", "inference"], default-features = false } # Vulkan
```

## Usage

### CLI

```bash
# Show backend information
birefnet info

# Single image inference
birefnet infer --input image.jpg --output results/ --model General

# Batch processing
birefnet infer --input image_folder/ --output results/ --model General

# List available models
birefnet infer --list-models
```

### Library

```rust
use birefnet::{SelectedBackend, SelectedDevice, create_device, get_backend_name};

fn main() {
    let device = create_device();
    println!("Using backend: {}", get_backend_name());

    #[cfg(feature = "inference")]
    {
        // Inference functionality is available
        // See examples in the repository
    }
}
```

## Supported Backends

- **ndarray** (default): CPU backend, good for development and compatibility
- **wgpu**: Cross-platform GPU backend, works on NVIDIA, AMD, Intel GPUs
- **cuda**: NVIDIA CUDA GPU backend
- **rocm**: AMD ROCm GPU backend
- **metal**: Apple Metal GPU backend
- **vulkan**: Vulkan GPU backend

## Architecture

This crate integrates the following specialized crates:

- [`birefnet-model`](../birefnet-model): Core BiRefNet model implementation
- [`birefnet-backbones`](../birefnet-backbones): Backbone networks (Swin, ResNet, VGG, PVT v2)
- [`birefnet-inference`](../birefnet-inference): Inference engine and post-processing
- [`birefnet-util`](../birefnet-util): Image processing utilities

## Currently Not Available

- **Training functionality**: CLI structure exists but training loop is not implemented

## License

MIT OR Apache-2.0