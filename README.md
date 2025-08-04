# BiRefNet-Burn

![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)
![Rust](https://img.shields.io/badge/rust-1.85.1%2B-orange.svg)
![Burn](https://img.shields.io/badge/burn-0.17.1-red.svg)

A comprehensive Rust implementation of the BiRefNet (Bilateral Reference Network) for high-resolution dichotomous image
segmentation, built using the Burn deep learning framework. This project provides a complete ecosystem for training,
inference, and deployment of BiRefNet models with cross-platform compatibility.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Inference](#inference)
  - [Training](#training)
  - [Model Conversion](#model-conversion)
- [Architecture](#architecture)
- [Current Limitations](#current-limitations)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

BiRefNet is a state-of-the-art model for high-resolution dichotomous image segmentation, as detailed in the paper "
Bilateral Reference for High-Resolution Dichotomous Image Segmentation" [1]. This Rust implementation leverages the Burn
framework to provide enhanced performance, memory safety, and cross-platform deployment capabilities.

- **Original PyTorch Implementation**: [ZhengPeng7/BiRefNet](https://github.com/ZhengPeng7/BiRefNet)
- **Burn Framework**: [tracel-ai/burn](https://github.com/tracel-ai/burn)

## Features

### ✅ Implemented

- **Complete Model Architecture**: Full BiRefNet implementation with bilateral reference mechanism
- **Multiple Backbone Support**: Swin Transformer v1 (Tiny, Small, Base, Large), ResNet, and VGG architectures
- **Cross-Platform Inference**: WebGPU, ndarray, and other Burn backends
- **Training Infrastructure**: Complete training pipeline with loss functions and metrics
- **Model Conversion**: PyTorch to Burn model conversion utility
- **Dataset Support**: DIS5K dataset loading and preprocessing
- **Configuration System**: Type-safe, hierarchical configuration with validation
- **Multiple Backends**: CPU (ndarray), GPU (WebGPU), and extensible backend system
- **Memory Efficient**: Optimized tensor operations and memory management
- **Comprehensive Examples**: Inference, training, conversion, and dataset testing

### 🔧 Core Components

- **BiRefNet Model**: Main segmentation model with decoder blocks and lateral connections
- **ASPP Module**: Atrous Spatial Pyramid Pooling for multi-scale feature extraction
- **Decoder Blocks**: Progressive refinement with attention mechanisms
- **Loss Functions**: BCE, IoU, Structure Loss, and combined loss implementations
- **Evaluation Metrics**: F-measure, IoU, MAE for model assessment
- **Data Pipeline**: Efficient dataset loading, augmentation, and batching

## Installation

### Prerequisites

- **Rust**: 1.85.1 or later
- **System Dependencies**: Based on your chosen backend (e.g., GPU drivers for WebGPU)

### Steps

1. **Install Rust**:

   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source $HOME/.cargo/env
   ```

2. **Clone the repository**:

   ```bash
   git clone https://github.com/nusu-github/BiRefNet-Burn.git
   cd BiRefNet-Burn
   ```

3. **Build the project**:

   ```bash
   # Build with all features
   cargo build --all-features --release

   # Build with specific backend
   cargo build --release --features wgpu --no-default-features
   ```

## Usage

### Inference

Run inference on single images or directories:

```bash
# Single image inference
cargo run --release --bin inference -- path/to/model.mpk path/to/image.jpg

# Batch inference on directory
cargo run --release --bin inference -- path/to/model.mpk path/to/images/ --output-dir results/

# Inference with custom settings
cargo run --release --bin inference -- path/to/model.mpk image.jpg \
  --size 1024 --threshold 0.5 --save-mask-only
```

### Training

Train BiRefNet models with custom datasets:

```bash
# Train with default settings (ndarray backend)
cargo run --release --bin training

# Train with GPU backend
cargo run --release --bin training --features wgpu --no-default-features

# Train with custom configuration
cargo run --release --bin training -- \
  --dataset path/to/dataset \
  --epochs 100 \
  --batch-size 4 \
  --learning-rate 1e-4
```

### Model Conversion

Convert PyTorch models to Burn format:

```bash
# Convert general model
cargo run --release --bin converter -- path/to/model.pth BiRefNet

# Convert specific variant
cargo run --release --bin converter -- model.pth BiRefNetLite --half

# Available model variants:
# BiRefNet, BiRefNetLite, BiRefNetMatting, BiRefNetCOD, etc.
```

### Development Commands

```bash
# Format code
cargo fmt --all

# Run linting
cargo clippy --all-targets --all-features -- -D warnings

# Run tests
cargo test --all-features

# Generate documentation
cargo doc --all-features --no-deps --open
```

## Architecture

### Model Components

- **Backbone Networks**: Swin Transformer v1, ResNet, VGG with pre-trained weights
- **Bilateral Reference**: Core mechanism for high-resolution segmentation
- **Multi-Scale Processing**: Hierarchical feature extraction and refinement
- **Attention Mechanisms**: Channel and spatial attention for feature enhancement
- **Progressive Decoding**: Stage-wise refinement with skip connections

### Supported Configurations

- **Backbones**: SwinV1-T/S/B/L, ResNet variants, VGG architectures
- **Input Resolutions**: 224×224 to 2048×2048 (configurable)
- **Multi-Scale Supervision**: Training with multiple output scales
- **Context Channels**: 0-3 context levels for enhanced feature extraction

## Current Limitations

While this project aims to provide a comprehensive Rust implementation of BiRefNet, it's important to note that full feature parity with the original PyTorch version is not always feasible due to the evolving maturity of the Burn deep learning framework and its ecosystem. The following features are not yet fully implemented or integrated:

### 🚧 Missing Features

- **`BiRefNetC2F` Model**: The coarse-to-fine model (`BiRefNetC2F`) from the Python implementation is not yet available.
- **Full Data Augmentation**: While basic resizing is implemented, advanced data augmentation techniques (e.g., random flip, crop, rotate, color enhance, pepper noise) are not fully integrated into the dataset pipeline.
- **Complete Loss Calculation**: The full training loss system, including auxiliary classification and gradient distillation (GDT) losses, is not yet fully integrated into the training step.
- **Advanced Evaluation Metrics**: Metrics such as Human Correction Efforts (HCE), Mean Boundary Accuracy (MBA), and Boundary IoU (BIoU) are not yet implemented. The Weighted F-measure (WFM) also requires a more robust Euclidean distance transform.
- **Numerical Stability for BCE Loss**: The `BCELoss` implementation needs further refinement for full numerical stability, matching PyTorch's behavior.
- **Refinement Functions**: Advanced foreground refinement functions like `refine_foreground` from the Python `image_proc.py` are not yet implemented.
- **Multi-GPU Training**: Distributed training across multiple GPUs is not yet supported.
- **Dynamic Resolution Training**: Training with variable input resolutions is not yet implemented.
- **Box-Guided Segmentation**: Bounding box guidance for segmentation is not yet available.
- **Specialized Variants**: Some task-specific optimizations and model variants found in the original implementation are not yet ported.

### 📊 Performance Gaps

- **Optimization**: Further performance tuning is necessary for production deployment.
- **Memory Usage**: Additional memory optimizations are required for very large-scale inference or training.
- **Batch Processing**: Further enhancements for highly optimized batch processing are needed.

_Note: While these performance aspects are important for production-grade applications, their full optimization is a secondary goal, as the primary focus is on establishing core functionality and architectural integrity within the Burn framework._

## License

This project is dual-licensed under both the MIT and Apache-2.0 licenses. You may choose either license when using this
project.

- [MIT License](LICENSE-MIT)
- [Apache-2.0 License](LICENSE-APACHE)

## Project Status

| Component  | Status      | Notes                                       |
| ---------- | ----------- | ------------------------------------------- |
| Core Model | ✅ Complete | Full BiRefNet architecture implemented      |
| Inference  | ✅ Complete | Single/batch image processing               |
| Training   | ✅ Complete | Full training pipeline with metrics         |
| Conversion | ✅ Complete | PyTorch to Burn model conversion            |
| Datasets   | 🔄 Partial  | DIS5K supported, others need implementation |
| Backbones  | 🔄 Partial  | Swin v1 complete, PVT v2 missing            |
| Evaluation | 🔄 Partial  | Basic metrics, advanced benchmarking needed |

## Contributing

We welcome contributions! Please see our contribution guidelines:

1. **Fork** the repository
2. **Create** a feature branch
3. **Implement** your changes with tests
4. **Ensure** all tests pass and code is formatted
5. **Submit** a pull request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/BiRefNet-Burn.git
cd BiRefNet-Burn

# Install development dependencies
cargo install cargo-llvm-cov  # For coverage
cargo install cargo-expand    # For macro expansion debugging

# Run development checks
cargo fmt --all -- --check
cargo clippy --all-targets --all-features
cargo test --all-features
```

## Performance Benchmarks

🚧 In progress

## Acknowledgements

### Core Contributors

- **Original BiRefNet**: [ZhengPeng7/BiRefNet](https://github.com/ZhengPeng7/BiRefNet) - The groundbreaking research and
  PyTorch implementation
- **Burn Framework**: [Tracel.ai Team](https://github.com/tracel-ai/burn) - The modern deep learning framework that
  makes this project possible

### Research Foundation

This implementation is based on extensive research in computer vision and deep learning:

- **Swin Transformer**: Liu et al., for the hierarchical vision transformer architecture
- **Image Segmentation**: The broader computer vision community for advances in semantic segmentation
- **Rust ML Ecosystem**: Contributors to candle, tch, ort, and other Rust ML libraries

### Special Thanks

- The Rust community for feedback and contributions
- Beta testers who helped identify and resolve issues
- Documentation reviewers and technical writers

## References

### Primary Paper

- Zheng, P., Gao, D., Fan, D., Liu, L., Laaksonen, J., Ouyang, W., & Sebe, N. (2024). "Bilateral Reference for High-Resolution Dichotomous Image Segmentation". _CAAI Artificial Intelligence Research_, 3, [arXiv:2401.03407](https://arxiv.org/abs/2401.03407)

### Backbone Architectures

- Simonyan, K., & Zisserman, A. (2015). "Very Deep Convolutional Networks for Large-Scale Image Recognition". _International Conference on Learning Representations (ICLR)_. [arXiv:1409.1556](https://arxiv.org/abs/1409.1556)
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep Residual Learning for Image Recognition". _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- Liu, Z., Lin, Y., Cao, Y., et al. (2021). "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows". _ICCV 2021_. [arXiv:2103.14030](https://arxiv.org/abs/2103.14030)

### Implementation Details & Inspirations

Many components and design patterns in BiRefNet-Burn are inspired by or directly adapted from established deep learning libraries and research implementations.

- PyTorch: A widely used open-source machine learning framework. [https://pytorch.org/](https://pytorch.org/)
- OpenCV: An open-source computer vision and machine learning software library. [https://opencv.org/](https://opencv.org/)
- timm (PyTorch Image Models): A collection of SOTA image models, often used for backbone implementations. [https://github.com/rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models)
- Kornia: An open-source computer vision library for PyTorch. [https://kornia.readthedocs.io/](https://kornia.readthedocs.io/)

### Key Dependencies

- Burn Framework Documentation: [https://burn.dev](https://burn.dev)

---

## Support

- **Issues**: [GitHub Issues](https://github.com/nusu-github/BiRefNet-Burn/issues)
- **Discussions**: [GitHub Discussions](https://github.com/nusu-github/BiRefNet-Burn/discussions)
- **Documentation**: [Online Docs](https://docs.rs/birefnet-burn)

For commercial support or custom development, please contact the maintainers.
