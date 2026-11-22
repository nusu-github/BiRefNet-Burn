# birefnet

[![Crates.io](https://img.shields.io/crates/v/birefnet.svg)](https://crates.io/crates/birefnet)
[![Documentation](https://docs.rs/birefnet/badge.svg)](https://docs.rs/birefnet)

Umbrella crate for BiRefNet (Bilateral Reference Network) dichotomous image segmentation on
[Burn](https://burn.dev) `0.22.1`: the model, pretrained weights, inference, training, and
run-time device selection behind one dependency.

`birefnet` is a library only. The `birefnet` command-line program lives in
[`birefnet-cli`](https://crates.io/crates/birefnet-cli).

## Features

The library never enables a backend by default; the final binary or test target chooses which
backends are compiled in, and [`device::select_device`] picks one at run time.

| Feature     | Default | Effect                                                                        |
| ----------- | ------- | ----------------------------------------------------------------------------- |
| `weights`   | via `hub` / `inference` | `birefnet::weights` (Burnpack, SafeTensors, PyTorch checkpoints) |
| `hub`       | yes     | Hugging Face Hub downloads in `birefnet::weights`                             |
| `inference` | yes     | `birefnet::inference` (`Segmenter`) and image decoding                        |
| `train`     | no      | `birefnet::{loss, metric, train}` and the `TrainStep` / `InferenceStep` impls  |
| `flex`      | no      | Pure-Rust CPU backend with SIMD and threads                                   |
| `cpu`       | no      | CubeCL CPU backend (LLVM JIT)                                                 |
| `wgpu`      | no      | WebGPU backend (Vulkan / Metal / DX12 chosen at run time)                     |
| `vulkan`    | no      | WebGPU pinned to Vulkan                                                       |
| `metal`     | no      | WebGPU pinned to Metal                                                        |
| `cuda`      | no      | NVIDIA CUDA backend                                                           |
| `rocm`      | no      | AMD `ROCm` backend                                                            |
| `fusion`    | no      | Kernel fusion and autotuning for the CubeCL backends; enable it with any GPU backend |

## Modules

| Module      | Crate                                                        | Feature     |
| ----------- | ------------------------------------------------------------ | ----------- |
| `device`    | this crate                                                   | always      |
| `model`     | [`birefnet-model`](https://docs.rs/birefnet-model)           | always      |
| `util`      | [`birefnet-util`](https://docs.rs/birefnet-util)             | always      |
| `weights`   | [`birefnet-weights`](https://docs.rs/birefnet-weights)       | `weights`   |
| `inference` | [`birefnet-inference`](https://docs.rs/birefnet-inference)   | `inference` |
| `train`     | [`birefnet-train`](https://docs.rs/birefnet-train)           | `train`     |
| `loss`      | [`birefnet-loss`](https://docs.rs/birefnet-loss)             | `train`     |
| `metric`    | [`birefnet-metric`](https://docs.rs/birefnet-metric)         | `train`     |

## Examples

Device names and precisions are parsed into typed values; failures are [`device::DeviceError`]
values with `is_*` predicates.

```rust
use birefnet::device::{Precision, select_device};

let precision: Precision = "bf16".parse()?;
assert_eq!(precision.to_string(), "bf16");

let err = select_device("tpu").unwrap_err();
assert!(err.is_unsupported_device());
assert!(select_device("cuda:x").unwrap_err().is_invalid_index());
# Ok::<(), birefnet::device::DeviceError>(())
```

Segmenting a photo with a pretrained preset downloaded from Hugging Face Hub (needs the `hub`
feature and a backend feature such as `flex` in the final binary):

```rust,no_run
# #[cfg(all(feature = "inference", feature = "hub"))]
# fn main() -> Result<(), Box<dyn std::error::Error>> {
use birefnet::{
    device::{Precision, configure_precision, select_device},
    inference::{Segmenter, SegmenterConfig},
    model::BiRefNet,
    weights::{BiRefNetWeightLoading, HfHubFetcher, ManagedModel},
};

let mut device = select_device("default")?;
configure_precision(&mut device, Precision::F32)?;

let managed = ManagedModel::from_pretrained("General")?;
let model = BiRefNet::from_managed(&managed, &HfHubFetcher::new()?, &device)?;

let segmenter = Segmenter::new(&model, SegmenterConfig::new().with_input_size([1024, 1024])?);
let output = segmenter.segment_file("photo.png", &device)?;
output.to_rgba_image()?.save("photo_mask.png")?;
# Ok(())
# }
# #[cfg(not(all(feature = "inference", feature = "hub")))]
# fn main() {}
```

## License

MIT OR Apache-2.0
