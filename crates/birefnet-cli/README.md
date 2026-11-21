# birefnet-cli

**Command-line interface (`birefnet`) for BiRefNet segmentation inference, model training, and device inspection**

## Features

| Feature  | Default | Effect                                                                 |
| -------- | ------- | ---------------------------------------------------------------------- |
| `flex`   | yes     | Pure-Rust CPU backend with SIMD and threads                            |
| `hub`    | yes     | Enables downloading pretrained SafeTensors presets from Hugging Face   |
| `train`  | no      | Enables the `train` subcommand with `burn::train`                      |
| `cpu`    | no      | CubeCL CPU backend (LLVM JIT)                                          |
| `wgpu`   | no      | WebGPU backend (Vulkan / Metal / DX12)                                 |
| `vulkan` | no      | WebGPU pinned to Vulkan                                                |
| `metal`  | no      | WebGPU pinned to Metal                                                 |
| `cuda`   | no      | NVIDIA CUDA backend                                                    |
| `rocm`   | no      | AMD ROCm backend                                                       |
| `fusion` | no      | Kernel fusion and autotuning for CubeCL backends                       |

## Usage

```bash
# Show compiled backends and active float dtype
cargo run --release -p birefnet-cli -- info

# List all registered pretrained model presets (no --input/--output/--model required)
cargo run --release -p birefnet-cli -- infer --list-models

# Run inference on an image or directory (outputs `<stem>_mask.png` RGBA images)
cargo run --release -p birefnet-cli -- infer -i photo.jpg -o results/ -m General

# Run offline inference from a local `.bpk`, `.safetensors`, or `.pt` checkpoint
cargo run --release -p birefnet-cli --no-default-features --features flex -- \
  infer -i photo.jpg -o results/ -m path/to/model.bpk --no-refine

# Train on CUDA with kernel fusion enabled
cargo run --release -p birefnet-cli --features train,cuda,fusion -- \
  --device cuda train --config config.json --artifact-dir ./artifacts
```

## License

MIT OR Apache-2.0
