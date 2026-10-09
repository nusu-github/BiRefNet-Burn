# BiRefNet

[![Crates.io](https://img.shields.io/crates/v/birefnet.svg)](https://crates.io/crates/birefnet)
[![Documentation](https://docs.rs/birefnet/badge.svg)](https://docs.rs/birefnet)

**Bilateral Reference Network for high-resolution dichotomous image segmentation**

This is the main BiRefNet crate that provides a unified CLI for inference and training across multiple backends (CPU, WebGPU, Vulkan, Metal, CUDA, ROCm).

Built on [Burn](https://burn.dev) 0.22: the backend is a run-time `Device` value. Cargo features only decide which backends are compiled in, and `--device` picks one when the program runs.

## Implemented Features

- ✅ **Cross-platform inference**: Single and batch image processing
- ✅ **Multi-backend support**: CPU (Flex, CubeCL CPU), GPU (WebGPU, Vulkan, Metal, CUDA, ROCm), several in one binary
- ✅ **CLI tool**: Inference, training and device information
- ✅ **Model management**: PyTorch / SafeTensors weight loading, Burnpack (`.bpk`) checkpoints
- ✅ **Precision selection**: `--precision f32|f16|bf16` at run time

## Installation

### As a CLI tool

```bash
# CPU only (default: Flex backend, inference, kernel fusion)
cargo install birefnet

# Add GPU backends next to the CPU one; `--device` chooses at run time
cargo install birefnet --features wgpu   # WebGPU (cross-platform)
cargo install birefnet --features vulkan # Vulkan (pins wgpu to SPIR-V)
cargo install birefnet --features metal  # Apple Metal
cargo install birefnet --features cuda   # NVIDIA CUDA
cargo install birefnet --features rocm   # AMD ROCm

# Training support
cargo install birefnet --features train
```

### Cargo features

| Feature     | Default | Effect                                                           |
| ----------- | ------- | ---------------------------------------------------------------- |
| `inference` | yes     | `infer` subcommand                                               |
| `train`     | no      | `train` subcommand (Burn `train`, TUI dashboard, system metrics) |
| `flex`      | yes     | Pure-Rust CPU backend with SIMD and threads                      |
| `cpu`       | no      | CubeCL CPU backend (LLVM JIT; downloads an LLVM bundle at build) |
| `wgpu`      | no      | WebGPU backend (Vulkan / Metal / DX12 chosen at run time)        |
| `vulkan`    | no      | wgpu pinned to Vulkan                                            |
| `metal`     | no      | wgpu pinned to Metal                                             |
| `cuda`      | no      | NVIDIA CUDA backend                                              |
| `rocm`      | no      | AMD ROCm backend                                                 |
| `fusion`    | yes     | Kernel fusion and autotuning for the CubeCL backends             |

## Usage

### CLI

```bash
# Show the devices compiled into this binary and the selected one
birefnet info
birefnet --device wgpu info

# Single image inference
birefnet infer --input image.jpg --output results/ --model General

# Batch processing
birefnet infer --input image_folder/ --output results/ --model General

# List available models
birefnet infer --list-models

# Pick a device and precision explicitly
birefnet --device cuda:1 --precision f16 infer --input image.jpg --output results/ --model General
```

`--device` accepts `default`, `flex`, `cpu`, `wgpu`, `wgpu-cpu`, `vulkan`, `metal`, `cuda[:N]` and `rocm[:N]` (only the ones compiled in). `default` uses Burn's `Device::default()`: the first compiled-in backend in the order CUDA, Metal, ROCm, Vulkan, wgpu, CPU, Flex, or the backend named by the `BURN_DEVICE` environment variable.

### Library

```rust
use birefnet::device::select_device;

fn main() -> anyhow::Result<()> {
    let device = select_device("default")?;
    println!("Using device: {device:?}");
    Ok(())
}
```

## Architecture

This crate integrates the following specialized crates:

- [`birefnet-model`](../birefnet-model): Core BiRefNet model implementation
- [`birefnet-backbones`](../birefnet-backbones): Backbone networks (Swin, ResNet, VGG, PVT v2)
- [`birefnet-inference`](../birefnet-inference): Inference engine and post-processing
- [`birefnet-util`](../birefnet-util): Image processing utilities

## License

MIT OR Apache-2.0