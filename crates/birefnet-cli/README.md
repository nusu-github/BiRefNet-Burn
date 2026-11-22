# birefnet-cli

Command-line program (`birefnet`) for BiRefNet segmentation inference, training, and backend
inspection.

## Features

| Feature  | Default | Effect                                                                   |
| -------- | ------- | ------------------------------------------------------------------------ |
| `flex`   | yes     | Pure-Rust CPU backend with SIMD and threads                              |
| `hub`    | yes     | Download pretrained presets from Hugging Face Hub                        |
| `fusion` | yes     | Kernel fusion and autotuning for the CubeCL backends below               |
| `train`  | no      | The `train` subcommand                                                   |
| `cpu`    | no      | CubeCL CPU backend (LLVM JIT; downloads an LLVM bundle at build time)    |
| `wgpu`   | no      | WebGPU backend (Vulkan / Metal / DX12 chosen at run time)                |
| `vulkan` | no      | WebGPU pinned to Vulkan                                                  |
| `metal`  | no      | WebGPU pinned to Metal                                                   |
| `cuda`   | no      | NVIDIA CUDA backend                                                      |
| `rocm`   | no      | AMD ROCm backend                                                         |

`fusion` has no effect on `flex`. For an unfused GPU build, use
`--no-default-features --features wgpu` (plus `hub` if needed).

## Global options

- `--device <name>`: `default` (Burn's highest-priority compiled backend, or `BURN_DEVICE`),
  `flex`, `cpu`, `wgpu`, `wgpu-cpu`, `vulkan`, `metal`, `cuda[:N]`, `rocm[:N]`. GPU indices are
  checked against the devices present.
- `--precision f32|f16|bf16`: default float dtype of the device, set before the first
  allocation. Fails if the backend does not support it.

## Inference

```bash
# List the registered pretrained presets
birefnet infer --list-models

# Segment an image or every supported image in a directory; writes `<stem>_mask.png` (RGBA)
birefnet infer -i photo.jpg -o results/ -m General
birefnet infer -i photos/ -o results/ -m General-Lite --no-refine

# Local checkpoint (`.bpk`, `.safetensors`, `.pt`/`.pth`; the format is read from the file header)
birefnet infer -i photo.jpg -o results/ -m artifacts/model.bpk
birefnet infer -i photo.jpg -o results/ -m weights.safetensors --preset General-HR
birefnet infer -i photo.jpg -o results/ -m weights.bpk --config my_model_config.json
```

The architecture of a local checkpoint comes from, in order: `--config`, a
`model_config.json` next to the checkpoint (written by `birefnet train`), `--preset`, or the
`General` layout. `--preset` also selects that preset's inference resolution; otherwise
1024×1024 is used.

In directory mode, images that fail are logged and skipped; the command fails only when every
image fails.

## Training

```bash
cargo run --release -p birefnet-cli --features train,cuda -- \
  --device cuda train --config train.json --artifact-dir ./artifacts

# Resume after epoch 10 from ./artifacts/checkpoint
cargo run --release -p birefnet-cli --features train,cuda -- \
  --device cuda train --config train.json --artifact-dir ./artifacts --resume 10
```

The configuration is JSON. Only `model` and `dataset.data_root_dir` are required; unknown keys
are rejected.

```json
{
  "model": "General",
  "dataset": {
    "data_root_dir": "datasets",
    "training_sets": ["DIS-TR"],
    "validation_set": "DIS-VD",
    "size": [1024, 1024],
    "dynamic_size": null,
    "background_synthesis": false
  },
  "optimizer": {
    "kind": "adamw",
    "learning_rate": 1e-4,
    "weight_decay": 1e-2,
    "lr_decay_epochs": [],
    "lr_decay_rate": 0.5
  },
  "num_epochs": 120,
  "batch_size": 4,
  "num_workers": 4,
  "keep_checkpoints": 2,
  "gradient_checkpointing": false,
  "seed": 42
}
```

| Key | Meaning |
| --- | --- |
| `model` | A preset name, or a full `ModelConfig` object (the format of `model_config.json`) |
| `dataset.data_root_dir` | Directory containing `<task>/<split>/{im,gt}` |
| `dataset.training_sets` | Training splits, concatenated |
| `dataset.validation_set` | Validation split; never augmented |
| `dataset.size` | Training and validation `[width, height]` |
| `dataset.dynamic_size` | `[[min_w, max_w], [min_h, max_h]]`: a random size per batch, rounded to multiples of 32 |
| `dataset.background_synthesis` | Replace backgrounds with synthetic colors during training |
| `optimizer.kind` | `adamw` (decoupled weight decay), `adam`, or `sgd` (L2 weight decay) |
| `optimizer.lr_decay_epochs` | Epochs after which the learning rate is multiplied by `lr_decay_rate`; negative values count back from the last epoch |
| `keep_checkpoints` | Number of most recent checkpoints kept for `--resume` |
| `gradient_checkpointing` | Recompute activations in the backward pass to save memory |

The artifact directory receives `config.json`, `model_config.json`, checkpoints, metric logs
(loss, validation MAE, and adaptive F-measure), and the final `model.bpk`, which
`birefnet infer -m <artifact_dir>/model.bpk` loads directly.

## Diagnostics

```bash
birefnet info
birefnet --device wgpu --precision f16 info
```

`info` prints the compiled device names, the selected device and backend, its default float
dtype, and which float dtypes it supports.

## License

MIT OR Apache-2.0
