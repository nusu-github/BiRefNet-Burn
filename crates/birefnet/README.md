# birefnet

[![Crates.io](https://img.shields.io/crates/v/birefnet.svg)](https://crates.io/crates/birefnet)
[![Documentation](https://docs.rs/birefnet/badge.svg)](https://docs.rs/birefnet)

**Umbrella library crate for `BiRefNet` segmentation, model weights, inference, training, and runtime device selection**

Built on [Burn](https://burn.dev) `0.22.1`. `birefnet` is a pure library facade with no CLI dependencies (`clap`, `anyhow`, `stacker`); for the command-line executable, see [`birefnet-cli`](../birefnet-cli).

## Features

| Feature     | Default | Effect                                                                    |
| ----------- | ------- | ------------------------------------------------------------------------- |
| `weights`   | via `hub`/`inference` | Re-exports `birefnet-weights` as `birefnet::weights`        |
| `hub`       | yes     | Enables Hugging Face Hub downloads inside `birefnet::weights`             |
| `inference` | yes     | Enables `birefnet::inference` (`Segmenter`) and `birefnet-util/image-io`  |
| `train`     | no      | Enables `birefnet::{loss, metric, train}` and `BiRefNet` `TrainStep` impl |
| `flex`      | yes     | Pure-Rust CPU backend with SIMD and threads                               |
| `cpu`       | no      | CubeCL CPU backend (LLVM JIT)                                             |
| `wgpu`      | no      | WebGPU backend (Vulkan / Metal / DX12 selected at run time)               |
| `vulkan`    | no      | WebGPU pinned to Vulkan                                                   |
| `metal`     | no      | WebGPU pinned to Metal                                                    |
| `cuda`      | no      | NVIDIA CUDA backend                                                       |
| `rocm`      | no      | AMD ROCm backend                                                          |
| `fusion`    | no      | Kernel fusion and autotuning for CubeCL backends                          |

## Usage

### Selecting Device & Precision

```rust
use birefnet::device::{Precision, configure_precision, select_device};

let mut device = select_device("default")?;
configure_precision(&mut device, Precision::F32)?;
# Ok::<(), birefnet::device::DeviceError>(())
```

All device/precision errors are returned as typed [`device::DeviceError`] values (`is_invalid_index`, `is_unsupported_device`, `is_precision_configuration`).

### Loading & Segmenting via Umbrella Re-exports

```rust,no_run
use birefnet::{
    device::select_device,
    inference::{Segmenter, SegmenterConfig},
    model::BiRefNet,
    weights::{BiRefNetWeightLoading, ManagedModel},
};

let device = select_device("flex")?;
let managed = ManagedModel::from_pretrained("General")?;
let model = BiRefNet::from_managed_model(&managed, &device)?;
let segmenter = Segmenter::new(model, SegmenterConfig::default());
let output = segmenter.segment_file("photo.png", &device)?;
output.to_rgba_image()?.save("photo_mask.png")?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Sub-crates

- [`birefnet-model`](../birefnet-model) -> `birefnet::model`
- [`birefnet-util`](../birefnet-util) -> `birefnet::util`
- [`birefnet-weights`](../birefnet-weights) -> `birefnet::weights` (`weights` feature)
- [`birefnet-inference`](../birefnet-inference) -> `birefnet::inference` (`inference` feature)
- [`birefnet-train`](../birefnet-train) -> `birefnet::train` (`train` feature)
- [`birefnet-loss`](../birefnet-loss) -> `birefnet::loss` (`train` feature)
- [`birefnet-metric`](../birefnet-metric) -> `birefnet::metric` (`train` feature)

## License

MIT OR Apache-2.0
