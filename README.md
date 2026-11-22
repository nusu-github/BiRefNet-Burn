# BiRefNet-Burn

![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)
![Rust](https://img.shields.io/badge/rust-1.100%2B-orange.svg)
![Burn](https://img.shields.io/badge/burn-0.22.1-red.svg)
[![CI](https://github.com/nusu-github/BiRefNet-Burn/actions/workflows/rust.yml/badge.svg)](https://github.com/nusu-github/BiRefNet-Burn/actions/workflows/rust.yml)

A Rust workspace implementing **BiRefNet** (Bilateral Reference Network) for high-resolution dichotomous image segmentation on the [Burn](https://burn.dev) deep learning framework (`0.22.1`, Edition `2024`, Rust `1.100+`).

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Inference](#inference)
  - [Training](#training)
  - [Backends](#backends)
  - [Backend Information](#backend-information)
- [Architecture](#architecture)
- [Workspace Crates](#workspace-crates)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [References](#references)

## Overview

BiRefNet is an encoder-decoder segmentation architecture with bilateral reference pathways introduced in *"Bilateral Reference for High-Resolution Dichotomous Image Segmentation"* [1].

- **Original PyTorch Implementation**: [ZhengPeng7/BiRefNet](https://github.com/ZhengPeng7/BiRefNet)
- **Burn Framework**: [tracel-ai/burn](https://github.com/tracel-ai/burn)

## Features

- **Complete BiRefNet Model**: Encoder, squeeze stage, bilateral multi-scale decoder (`BasicDecBlk`, `ResBlk`, `ASPP`, deformable ASPP), lateral blocks, and optional refinement (`RefUNet`, `Refiner`, `RefinerPVTInChannels4`).
- **Backbone Networks**: `SwinV1` (`Tiny`, `Small`, `Base`, `Large`), `PVTv2` (`B0`, `B1`, `B2`, `B5`), `ResNet` (`18`, `50`), and `VGG` (`16`, `16-BN`).
- **Pretrained Weight Loading**: 15 registered pretrained Hugging Face presets with optional Hub downloads (`hub` feature), SafeTensors key remapping, PyTorch import, and native Burnpack (`.bpk`) round-tripping.
- **Inference & Postprocessing**: High-level `Segmenter` pipeline with ImageNet preprocessing, native Burn spatial resizing, sigmoid probability mapping, Gaussian blur, disk morphology (opening/closing), 4-connected small-component filtering, hole filling, and blur-fusion foreground refinement.
- **Evaluation Suite**: `MAE`, `MSE`, `F-measure` (adaptive + curves), `Weighted F-measure`, `S-measure`, `E-measure` (adaptive + curve), and `BIoU`, matching the upstream `evaluation/metrics.py` (py_sod_metrics) to within 1e-9, plus `MetricsAggregator` and a Burn `Metric` adapter for validation.
- **Training Pipeline**: `BiRefNetDataset` discovery/batching with self-contained image augmentations (flip, crop, rotate, color enhancement, pepper noise, background synthesis, 32-aligned dynamic sizing), combined `BiRefNetLoss`, and `SupervisedTraining` integration in `birefnet-cli`.
- **Runtime Backend & Precision Selection**: Compile one or more Burn backends (`flex`, `cpu`, `wgpu`, `vulkan`, `metal`, `cuda`, `rocm`) and choose the active `--device` and `--precision` (`f32`, `f16`, `bf16`) at run time.

## Installation

### Prerequisites

- **Rust**: `1.100` or later (declared via `rust-version` in root `Cargo.toml`)
- **System Dependencies**: Native drivers/SDKs only for optional GPU backends (`wgpu`, `vulkan`, `cuda`, `rocm`)

### Build the Workspace or CLI

```bash
git clone https://github.com/nusu-github/BiRefNet-Burn.git
cd BiRefNet-Burn

# Default CLI binary (`birefnet` from `crates/birefnet-cli`): Flex CPU, Hugging Face Hub, fusion
cargo build --release -p birefnet-cli

# Add GPU backends and training support (`fusion` is on by default)
cargo build --release -p birefnet-cli --features wgpu
cargo build --release -p birefnet-cli --features cuda,train

# Or install the binary locally
cargo install --path crates/birefnet-cli
```

## Usage

### Inference

```bash
# Single image (auto-downloads pretrained SafeTensors weights for named presets)
cargo run --release -p birefnet-cli -- infer --input path/to/image.jpg --output results/ --model General

# Directory batch inference
cargo run --release -p birefnet-cli -- infer --input path/to/images/ --output results/ --model General

# List registered pretrained presets without passing -i/-o/-m
cargo run --release -p birefnet-cli -- infer --list-models

# Skip foreground color refinement
cargo run --release -p birefnet-cli -- infer --input image.jpg --output results/ --model General --no-refine

# Explicit GPU backend and half precision
cargo run --release -p birefnet-cli --features wgpu -- --device wgpu --precision f16 infer --input image.jpg --output results/ --model General
```

`--model` also accepts local checkpoints in Burnpack (`.bpk`), SafeTensors (`.safetensors`), or PyTorch (`.pt`/`.pth`) formats; the format is detected from the file header. Their architecture comes from `--config <model_config.json>`, a `model_config.json` next to the checkpoint, or `--preset <name>`.

### Training

```bash
# Train from a JSON configuration (writes checkpoints & logs to ./artifacts)
cargo run --release -p birefnet-cli --features train,cuda -- --device cuda train --config path/to/config.json

# Resume after epoch 10 from ./artifacts/checkpoint/
cargo run --release -p birefnet-cli --features train,cuda -- --device cuda train --config path/to/config.json --resume 10
```

A minimal configuration names a preset architecture and the dataset root; every other key has a default (see [`crates/birefnet-cli`](crates/birefnet-cli/README.md#training) for the full schema):

```json
{ "model": "General", "dataset": { "data_root_dir": "datasets" } }
```

### Backends

| Feature  | `--device`         | Notes                                                               |
| -------- | ------------------ | ------------------------------------------------------------------- |
| `flex`   | `flex`             | Default pure-Rust CPU backend with SIMD and threads                 |
| `cpu`    | `cpu`              | CubeCL CPU backend (LLVM JIT; downloads an LLVM bundle at build)    |
| `wgpu`   | `wgpu`, `wgpu-cpu` | WebGPU backend (Vulkan / Metal / DX12 chosen at run time)           |
| `vulkan` | `vulkan`           | WebGPU pinned to Vulkan                                             |
| `metal`  | `metal`            | WebGPU pinned to Metal (macOS)                                      |
| `cuda`   | `cuda`, `cuda:N`   | NVIDIA CUDA backend                                                 |
| `rocm`   | `rocm`, `rocm:N`   | AMD ROCm backend                                                    |

- `--device default` uses `Device::default()`, selecting the highest-priority compiled backend (or `BURN_DEVICE`).
- `fusion` (CubeCL kernel fusion and autotuning) is a default CLI feature; build with `--no-default-features --features <backend>` to compare against an unfused backend.
- `--precision f32|f16|bf16` configures the device float dtype before the first allocation.

### Backend Information

```bash
cargo run --release -p birefnet-cli -- info
cargo run --release -p birefnet-cli --features wgpu -- --device wgpu info
```

### Development Commands

```bash
export FEATURES=birefnet/train,birefnet-model/train,birefnet-cli/train,birefnet-util/image-io,birefnet-weights/hub,birefnet-weights/test-util
cargo fmt --all -- --check
CARGO_BUILD_WARNINGS=deny cargo clippy --workspace --all-targets --features "$FEATURES"
cargo test --workspace --features "$FEATURES"
CARGO_BUILD_WARNINGS=deny cargo doc --workspace --no-deps --features "$FEATURES"
cargo tree -d -p birefnet-cli   # must not list two copies of any burn crate
```

The same checks run in CI (`.github/workflows/rust.yml`).

## Architecture

- **Library-first separation**: Core crates never depend on CLI crates (`clap`, `anyhow`, `stacker`) or enable a backend directly; only dev/test targets and `birefnet-cli` opt into backends.
- **Typed library errors**: Each library crate exposes a boxed `Error + Send + Sync + 'static` error type with `is_*` predicates and `Error::source()` chains.

## Workspace Crates

| Crate                                                 | Role                                                                       |
| ----------------------------------------------------- | -------------------------------------------------------------------------- |
| [`birefnet`](crates/birefnet)                         | Pure umbrella library crate + runtime device/precision selection           |
| [`birefnet-cli`](crates/birefnet-cli)                 | Binary CLI (`birefnet`) for `infer`, `train`, and `info`                   |
| [`birefnet-model`](crates/birefnet-model)             | `BiRefNet` architecture, decoder, squeeze, ASPP, refinement, and configs   |
| [`birefnet-backbones`](crates/birefnet-backbones)     | Encoder backbones (`SwinV1`, `PVTv2`, `ResNet`, `VGG`)                     |
| [`birefnet-weights`](crates/birefnet-weights)         | Model preset catalog, optional Hub fetcher, and Burnpack/SafeTensors/PT IO |
| [`birefnet-inference`](crates/birefnet-inference)     | High-level `Segmenter` and mask postprocessing                             |
| [`birefnet-train`](crates/birefnet-train)             | `BiRefNetDataset`, batcher, and self-contained image augmentations         |
| [`birefnet-loss`](crates/birefnet-loss)               | `BiRefNetLoss`, BCE/IoU/SSIM/Structure/Contour/PatchIoU/MAE/Threshold loss |
| [`birefnet-metric`](crates/birefnet-metric)           | Segmentation evaluation metrics, Burn `Metric` adapters, and aggregators   |
| [`birefnet-util`](crates/birefnet-util)               | Pure tensor math, EDT, morphology, foreground refinement, optional codecs  |
| [`birefnet-extra-ops`](crates/birefnet-extra-ops)     | `DropPath`, `erfinv`, and `trunc_normal` operations for Burn               |

## License

Dual-licensed under either of:

- [MIT License](LICENSE-MIT)
- [Apache-2.0 License](LICENSE-APACHE)

## Acknowledgements

- [ZhengPeng7/BiRefNet](https://github.com/ZhengPeng7/BiRefNet) for the original PyTorch BiRefNet implementation.
- [Tracel.ai / Burn](https://github.com/tracel-ai/burn) for the Burn deep learning framework.

## References

- Zheng, P., Gao, D., Fan, D., Liu, L., Laaksonen, J., Ouyang, W., & Sebe, N. (2024). *Bilateral Reference for High-Resolution Dichotomous Image Segmentation*. CAAI Artificial Intelligence Research, 3, [arXiv:2401.03407](https://arxiv.org/abs/2401.03407).
- Liu, Z., Lin, Y., Cao, Y., et al. (2021). *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows*. ICCV 2021, [arXiv:2103.14030](https://arxiv.org/abs/2103.14030).
- Wang, W., Xie, E., Li, X., et al. (2022). *PVTv2: Improved Baselines with Pyramid Vision Transformer*. Computational Visual Media, 8(3), [arXiv:2106.13797](https://arxiv.org/abs/2106.13797).
