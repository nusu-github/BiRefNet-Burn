# birefnet

Umbrella library crate for BiRefNet (Bilateral Reference Network) on [Burn](https://burn.dev)
`0.22.1`: model architecture, pretrained weights, inference, training components, and runtime
device selection behind one workspace dependency.

`birefnet` is a library only. The `birefnet` command-line program lives in
[`birefnet-cli`](../birefnet-cli).

## Features

The library never enables a backend by default. The final binary or test target enables one or
more backend features and chooses one at run time with `birefnet::device::select_device`.

| Feature     | Default                 | Effect                                                                               |
| ----------- | ----------------------- | ------------------------------------------------------------------------------------ |
| `weights`   | via `hub` / `inference` | `birefnet::weights` (Burnpack, SafeTensors, PyTorch checkpoints)                     |
| `hub`       | yes                     | Hugging Face Hub downloads in `birefnet::weights`                                    |
| `inference` | yes                     | `birefnet::inference` (`Segmenter`) and image codecs                                 |
| `train`     | no                      | `birefnet::{loss, metric, train}` and `TrainStep` / `InferenceStep` implementations  |
| `flex`      | no                      | Pure-Rust CPU backend with SIMD and threads                                          |
| `cpu`       | no                      | CubeCL CPU backend (LLVM JIT)                                                        |
| `wgpu`      | no                      | WebGPU backend (Vulkan / Metal / DX12 chosen at run time)                            |
| `vulkan`    | no                      | WebGPU pinned to Vulkan                                                              |
| `metal`     | no                      | WebGPU pinned to Metal                                                               |
| `cuda`      | no                      | NVIDIA CUDA backend                                                                  |
| `rocm`      | no                      | AMD ROCm backend                                                                     |
| `fusion`    | no                      | Kernel fusion and autotuning for CubeCL backends; enable it alongside GPU backends   |

## Re-exported Modules

| Module      | Crate                                            | Feature     |
| ----------- | ------------------------------------------------ | ----------- |
| `device`    | this crate                                       | always      |
| `model`     | [`birefnet-model`](../birefnet-model)            | always      |
| `util`      | [`birefnet-util`](../birefnet-util)              | always      |
| `weights`   | [`birefnet-weights`](../birefnet-weights)        | `weights`   |
| `inference` | [`birefnet-inference`](../birefnet-inference)    | `inference` |
| `train`     | [`birefnet-train`](../birefnet-train)            | `train`     |
| `loss`      | [`birefnet-loss`](../birefnet-loss)              | `train`     |
| `metric`    | [`birefnet-metric`](../birefnet-metric)          | `train`     |

Low-level building blocks ([`birefnet-backbones`](../birefnet-backbones) and
[`birefnet-extra-ops`](../birefnet-extra-ops)) are separate workspace crates for callers that need
standalone encoders or Burn ops.

## Example

Add `birefnet` with a backend feature (for example `flex` or `wgpu`) to your binary's `Cargo.toml`:

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

> **Windows note**: Building and running Swin-L models recurses past Windows' 1 MiB default
> main-thread stack. Run model initialization, weight loading, and inference on a worker thread
> with at least a 64 MiB stack (or via `stacker::grow`).

## License

MIT OR Apache-2.0
