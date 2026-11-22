# BiRefNet-Burn

![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)
![Rust](https://img.shields.io/badge/rust-1.100%2B-orange.svg)
![Burn](https://img.shields.io/badge/burn-0.22.1-red.svg)
[![CI](https://github.com/nusu-github/BiRefNet-Burn/actions/workflows/rust.yml/badge.svg)](https://github.com/nusu-github/BiRefNet-Burn/actions/workflows/rust.yml)

A research and reference port of **BiRefNet** (Bilateral Reference Network) for high-resolution
dichotomous image segmentation on the [Burn](https://burn.dev) deep learning framework (`0.22.1`,
Rust Edition `2024`, MSRV `1.100`).

## Purpose and Scope

This repository implements the full BiRefNet pipeline in pure Rust (`#![forbid(unsafe_code)]`) to
explore running and reproducing a large vision architecture on Burn: encoder backbones, bilateral
decoder, optional refinement heads, pretrained Hugging Face SafeTensors / PyTorch weight loading,
postprocessing, `py_sod_metrics`-matched evaluation, and supervised training.

**Non-goals**:

- **Not a speed-optimized production replacement for upstream BiRefNet.** Burn's current backend
  optimization is still maturing, so both inference and training are noticeably slower than the
  [original PyTorch implementation](https://github.com/ZhengPeng7/BiRefNet) or ONNX Runtime /
  TensorRT exports.
- **Not published to crates.io.** Use it directly from this workspace as a reference codebase or
  git/path dependency.

See [`docs/parity-and-limitations.md`](docs/parity-and-limitations.md) for upstream parity
details, known backend limitations, and platform caveats.

## Quickstart

### CLI Inference

```bash
git clone https://github.com/nusu-github/BiRefNet-Burn.git
cd BiRefNet-Burn

# Segment one image (auto-downloads the `General` SafeTensors preset from Hugging Face Hub)
cargo run --release -p birefnet-cli -- infer -i photo.jpg -o results/ -m General

# Use WebGPU (Vulkan / Metal / DX12) with f16 precision, or list all 15 presets
cargo run --release -p birefnet-cli --features wgpu -- --device wgpu --precision f16 infer -i photo.jpg -o results/ -m General-Lite
cargo run --release -p birefnet-cli -- infer --list-models
```

Each processed image is written to `<output>/<stem>_mask.png` as an RGBA cutout with blur-fusion
foreground refinement enabled by default (pass `--no-refine` to skip foreground refinement).

### Rust Library Inference

Add the umbrella crate [`crates/birefnet`](crates/birefnet) plus at least one compute backend
feature (for example `flex` or `wgpu`) to your binary's `Cargo.toml`:

```rust
use birefnet::{
    device::{Precision, configure_precision, select_device},
    inference::{Segmenter, SegmenterConfig},
    model::BiRefNet,
    weights::{BiRefNetWeightLoading, HfHubFetcher, ManagedModel},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut device = select_device("default")?;
    configure_precision(&mut device, Precision::F32)?;

    let managed = ManagedModel::from_pretrained("General")?;
    let model = BiRefNet::from_managed(&managed, &HfHubFetcher::new()?, &device)?;

    let segmenter = Segmenter::new(&model, SegmenterConfig::new().with_input_size([1024, 1024])?);
    let output = segmenter.segment_file("photo.png", &device)?;
    output.to_rgba_image()?.save("photo_mask.png")?;
    Ok(())
}
```

> **Windows caveat**: Initializing and running deep Burn module trees (such as Swin-L) recurses
> past Windows' 1 MiB default main-thread stack. The CLI handles this automatically via
> `stacker::grow`; library callers on Windows should run model setup and inference on a thread
> with at least a 64 MiB stack.

## What Is Implemented

- **BiRefNet Architecture (`birefnet-model`)**: Multi-scale encoder fusion (`None`, `Add`, `Cat`),
  context concatenation, squeeze stage (`BasicDecBlk`, `ResBlk`, `ASPP`, `ASPPDeformable`),
  bilateral multi-scale decoder with split image pyramid (`dec_ipt` / `dec_ipt_split`), gradient
  guidance (`out_ref`), and optional refinement (`RefUNet`, `Refiner`, `RefinerPVTInChannels4`).
- **Backbones (`birefnet-backbones`)**: `SwinV1` (`T/S/B/L`), `PVTv2` (`B0..=B5`), `ResNet`
  (`18/34/50/101/152`), and `VGG` (`16`, `16-BN`). Inside `birefnet-model`, the upstream BiRefNet
  variants (`SwinV1T/S/B/L`, `PvtV2B0/B1/B2/B5`, `ResNet50`, `Vgg16`, `Vgg16Bn`) are wired into
  `ModelConfig`.
- **Pretrained Weights (`birefnet-weights`)**: All 15 official `ZhengPeng7` Hugging Face presets
  with SafeTensors key remapping, PyTorch `.pt`/`.pth` loading, and native Burnpack (`.bpk`)
  round-tripping.
- **Inference & Postprocessing (`birefnet-inference`, `birefnet-util`)**: `Segmenter` pipeline,
  mask thresholding, MATLAB-compatible Gaussian blur, disk opening/closing, 4-connected
  small-component filtering, hole filling, and two-pass blur-fusion foreground color estimation.
- **Evaluation (`birefnet-metric`)**: `MAE`, `MSE`, `F-measure` (adaptive + curve),
  `Weighted F-measure`, `S-measure`, `E-measure` (adaptive + curve), and `BIoU`, verified against
  upstream `py_sod_metrics` to within `1e-9`, plus a Burn `Metric` adapter (`train` feature).
- **Training (`birefnet-train`, `birefnet-loss`, `birefnet-cli`)**: `BiRefNetDataset`,
  32-aligned dynamic batch sizing, self-contained augmentations, multi-scale `BiRefNetLoss`, and
  `SupervisedTraining` CLI integration.

## Backends and Precision

Compile one or more Burn backends into `birefnet-cli` (or your own binary) and select the active
device and float precision at run time:

| Feature  | `--device`         | Notes                                                               |
| -------- | ------------------ | ------------------------------------------------------------------- |
| `flex`   | `flex`             | Default pure-Rust CPU backend with SIMD and threads                 |
| `cpu`    | `cpu`              | CubeCL CPU backend (LLVM JIT; downloads an LLVM bundle at build)    |
| `wgpu`   | `wgpu`, `wgpu-cpu` | WebGPU backend (Vulkan / Metal / DX12 chosen at run time)           |
| `vulkan` | `vulkan`           | WebGPU pinned to Vulkan                                             |
| `metal`  | `metal`            | WebGPU pinned to Metal (macOS)                                      |
| `cuda`   | `cuda`, `cuda:N`   | NVIDIA CUDA backend                                                 |
| `rocm`   | `rocm`, `rocm:N`   | AMD ROCm backend                                                    |

- `--device default` defers to `Device::default()` (honoring `BURN_DEVICE`, otherwise picking the
  highest-priority compiled backend).
- `fusion` (CubeCL kernel fusion + autotuning) is enabled by default in `birefnet-cli`; build
  with `--no-default-features --features <backend>` to test an unfused CubeCL backend.
- `--precision f32|f16|bf16` configures the device float dtype before the first allocation.
- Inspect the compiled backends and supported dtypes with `cargo run --release -p birefnet-cli -- info`.

## Documentation

| Document                                                       | Contents                                                                      |
| -------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| [`docs/parity-and-limitations.md`](docs/parity-and-limitations.md) | Upstream PyTorch mapping, numerical parity checks, and known Burn constraints |
| [`docs/architecture.md`](docs/architecture.md)                 | Workspace crate graph, model dataflow, and Burn 0.22 design decisions         |
| [`docs/pretrained-models.md`](docs/pretrained-models.md)       | All 15 pretrained presets, checkpoint formats, and key remapping rules        |
| [`docs/training.md`](docs/training.md)                         | Dataset layout, `train.json` / `model_config.json`, and CubeCL training notes |

Each crate also has its own `README.md` and rustdoc (`cargo doc --workspace --no-deps --open`).

## Workspace Crates

| Crate                                             | Role                                                                       |
| ------------------------------------------------- | -------------------------------------------------------------------------- |
| [`birefnet`](crates/birefnet)                     | Pure umbrella library crate + runtime device/precision selection           |
| [`birefnet-cli`](crates/birefnet-cli)             | Binary CLI (`birefnet`) for `infer`, `train`, and `info`                   |
| [`birefnet-model`](crates/birefnet-model)         | `BiRefNet` architecture, decoder, squeeze, ASPP, refinement, and configs   |
| [`birefnet-backbones`](crates/birefnet-backbones) | Encoder backbones (`SwinV1`, `PVTv2`, `ResNet`, `VGG`)                     |
| [`birefnet-weights`](crates/birefnet-weights)     | Model preset catalog, optional Hub fetcher, and Burnpack/SafeTensors/PT IO |
| [`birefnet-inference`](crates/birefnet-inference) | High-level `Segmenter` and mask postprocessing                             |
| [`birefnet-train`](crates/birefnet-train)         | `BiRefNetDataset`, batcher, and self-contained image augmentations         |
| [`birefnet-loss`](crates/birefnet-loss)           | `BiRefNetLoss` and component losses (BCE, IoU, SSIM, Structure, Contour…)  |
| [`birefnet-metric`](crates/birefnet-metric)       | Segmentation evaluation metrics, Burn `Metric` adapters, and aggregators   |
| [`birefnet-util`](crates/birefnet-util)           | Pure tensor math, EDT, morphology, foreground refinement, optional codecs  |
| [`birefnet-extra-ops`](crates/birefnet-extra-ops) | `DropPath`, `erfinv`, and `trunc_normal` operations for Burn               |

## Development Checks

The commands below match [`.github/workflows/rust.yml`](.github/workflows/rust.yml):

```bash
export FEATURES=birefnet/train,birefnet-model/train,birefnet-cli/train,birefnet-util/image-io,birefnet-weights/hub,birefnet-weights/test-util
cargo fmt --all -- --check
CARGO_BUILD_WARNINGS=deny cargo clippy --workspace --all-targets --features "$FEATURES"
cargo test --workspace --features "$FEATURES"
CARGO_BUILD_WARNINGS=deny cargo doc --workspace --no-deps --features "$FEATURES"
cargo tree -d -p birefnet-cli --features train -e normal --depth 0
```

## License

Dual-licensed under either of:

- [MIT License](LICENSE-MIT)
- [Apache-2.0 License](LICENSE-APACHE)

## Acknowledgements and References

- Original PyTorch implementation: [ZhengPeng7/BiRefNet](https://github.com/ZhengPeng7/BiRefNet)
- Deep learning framework: [tracel-ai/burn](https://github.com/tracel-ai/burn)
- Zheng, P., Gao, D., Fan, D., Liu, L., Laaksonen, J., Ouyang, W., & Sebe, N. (2024). *Bilateral Reference for High-Resolution Dichotomous Image Segmentation*. CAAI Artificial Intelligence Research, 3, [arXiv:2401.03407](https://arxiv.org/abs/2401.03407).
- Liu, Z., Lin, Y., Cao, Y., et al. (2021). *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows*. ICCV 2021, [arXiv:2103.14030](https://arxiv.org/abs/2103.14030).
- Wang, W., Xie, E., Li, X., et al. (2022). *PVTv2: Improved Baselines with Pyramid Vision Transformer*. Computational Visual Media, 8(3), [arXiv:2106.13797](https://arxiv.org/abs/2106.13797).
