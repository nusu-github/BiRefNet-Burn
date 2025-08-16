# BiRefNet

[![Crates.io](https://img.shields.io/crates/v/birefnet.svg)](https://crates.io/crates/birefnet)
[![Documentation](https://docs.rs/birefnet/badge.svg)](https://docs.rs/birefnet)

**Bilateral Reference Network for high-resolution dichotomous image segmentation**

This is the main BiRefNet crate that provides a unified interface for both inference and training across multiple backends (CPU, CUDA, WGPU).

## Features

- **Multi-backend support**: CPU (ndarray), NVIDIA GPU (CUDA), Cross-platform GPU (WGPU)
- **CLI tool**: Ready-to-use command line interface
- **Library interface**: Programmatic access to all BiRefNet functionality
- **Modular design**: Optional inference and training features

## Installation

### As a CLI tool

```bash
# CPU version
cargo install birefnet

# GPU version (CUDA)
cargo install birefnet --features cuda --no-default-features

# GPU version (WGPU)
cargo install birefnet --features wgpu --no-default-features
```

### As a library

```toml
[dependencies]
# CPU inference only
birefnet = { version = "0.1.0", features = ["inference"] }

# GPU training with CUDA
birefnet = { version = "0.1.0", features = ["cuda", "train"], default-features = false }

# GPU inference with WGPU
birefnet = { version = "0.1.0", features = ["wgpu", "inference"], default-features = false }
```

## Usage

### CLI

```bash
# Show backend information
birefnet info

# Run inference
birefnet infer --input image.jpg --output results/ --model model.mpk

# Train a model  
birefnet train --config config.json --resume checkpoint.mpk
```

### Library

```rust
use birefnet::{SelectedBackend, SelectedDevice, create_device, get_backend_name};

fn main() {
    let device = create_device();
    println!("Using: {}", get_backend_name());

    // Use with inference
    #[cfg(feature = "inference")]
    {
        use birefnet::inference::BiRefNetInference;
        // Inference code here
    }

    // Use with training
    #[cfg(feature = "train")]
    {
        use birefnet::train::BiRefNetTrainer;
        // Training code here
    }
}
```

## Backends

- **ndarray** (default): CPU backend, good for development and testing
- **cuda**: NVIDIA GPU backend, best performance for NVIDIA hardware
- **wgpu**: Cross-platform GPU backend, works on NVIDIA, AMD, Intel GPUs

## Architecture

BiRefNet consists of several specialized crates:

- [`birefnet-model`](../birefnet-model): Core model implementations
- [`birefnet-backbones`](../birefnet-backbones): Backbone networks (Swin, PVT v2, ResNet, VGG)
- [`birefnet-inference`](../birefnet-inference): Inference engine
- [`birefnet-train`](../birefnet-train): Training infrastructure
- [`birefnet-loss`](../birefnet-loss): Loss functions
- [`birefnet-metric`](../birefnet-metric): Evaluation metrics
- [`birefnet-util`](../birefnet-util): Utilities and tools
- [`birefnet-extra-ops`](../birefnet-extra-ops): Additional operations

## License

MIT OR Apache-2.0