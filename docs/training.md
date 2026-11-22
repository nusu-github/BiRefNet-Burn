# Training Guide and Configuration Reference

Training lives in `birefnet-train` (dataset, augmentations, batcher), `birefnet-loss`
(multi-scale segmentation losses), `birefnet-model` (`train` feature), `birefnet-metric`
(`train` feature), and `birefnet-cli` (`train` feature).

## Dataset Directory Layout

`BiRefNetDataset` discovers paired images and ground-truth masks by matching file stems under:

```text
<data_root_dir>/
└── <task_dir>/
    ├── <train_split>/
    │   ├── im/          # RGB images (PNG, JPEG, BMP, GIF, TIFF, WebP)
    │   └── gt/          # Same-stem grayscale masks
    └── <val_split>/
        ├── im/
        └── gt/
```

`<task_dir>` is determined by `ModelConfig::task.task`:

| `Task` variant | Expected `<task_dir>` |
| -------------- | --------------------- |
| `DIS5K`        | `DIS5K`               |
| `COD`          | `COD`                 |
| `HRSOD`        | `HRSOD`               |
| `General`      | `General`             |
| `General2k`    | `General2k`           |
| `Matting`      | `Matting`             |

Default split names in `birefnet train` are `training_sets: ["DIS-TR"]` and
`validation_set: "DIS-VD"`. Multiple entries in `training_sets` are concatenated via Burn's
`ComposedDataset`; validation data is never augmented.

## Running `birefnet train`

```bash
# Train on Flex CPU (supports both Bilinear and Nearest interpolation)
cargo run --release -p birefnet-cli --features train -- \
  --device flex train --config train.json --artifact-dir ./artifacts

# Train on a CubeCL GPU backend (requires `"interpolation": "Nearest"` in `model`)
cargo run --release -p birefnet-cli --features train,cuda -- \
  --device cuda train --config train_nearest.json --artifact-dir ./artifacts

# Resume after epoch 10 from ./artifacts/checkpoint
cargo run --release -p birefnet-cli --features train -- \
  --device flex train --config train.json --artifact-dir ./artifacts --resume 10
```

## Important: CubeCL Backend Interpolation Constraint

Burn `0.22.1` implements the `Bilinear` interpolation backward pass on `flex` only; CubeCL
backends (`cpu`, `wgpu`, `vulkan`, `metal`, `cuda`, `rocm`) differentiate `Nearest` only.

- Passing a preset string such as `"model": "General"` resolves to
  `InterpolationStrategy::Bilinear`.
- To train on any CubeCL backend (`cuda`, `wgpu`, `vulkan`, `metal`, `rocm`, `cpu`), pass an
  explicit `ModelConfig` object with `"interpolation": "Nearest"`:

```json
{
  "model": {
    "task": { "task": "DIS5K", "batch_size": 4 },
    "backbone": { "backbone": "SwinV1T" },
    "decoder": {
      "ms_supervision": true,
      "out_ref": true,
      "dec_ipt": true,
      "dec_ipt_split": true,
      "cxt_num": 3,
      "mul_scl_ipt": "Cat",
      "dec_att": "ASPPDeformable",
      "squeeze_block": { "BasicDecBlk": 1 },
      "dec_blk": "BasicDecBlk",
      "lat_blk": "BasicLatBlk",
      "dec_channels_inter": "Fixed"
    },
    "refine": { "refine": "None" },
    "interpolation": "Nearest"
  },
  "dataset": {
    "data_root_dir": "datasets",
    "training_sets": ["DIS-TR"],
    "validation_set": "DIS-VD",
    "size": [512, 512]
  },
  "num_epochs": 10,
  "batch_size": 2
}
```

## `train.json` Schema

Only `model` and `dataset.data_root_dir` are required. Unknown JSON keys are rejected
(`deny_unknown_fields`).

| Key                            | Type / Values                                       | Default       | Notes                                                                                                       |
| ------------------------------ | --------------------------------------------------- | ------------- | ----------------------------------------------------------------------------------------------------------- |
| `model`                        | Preset string or `ModelConfig` object               | *(required)*  | Preset strings use `Bilinear`; use a `ModelConfig` object with `"interpolation": "Nearest"` for CubeCL GPUs |
| `dataset.data_root_dir`        | Path string                                         | *(required)*  | Root directory containing `<task>/<split>/{im,gt}`                                                          |
| `dataset.training_sets`        | `Vec<String>` (non-empty)                           | `["DIS-TR"]`  | Training split directories, concatenated                                                                    |
| `dataset.validation_set`       | `String`                                            | `"DIS-VD"`    | Validation split directory (never augmented, always fixed `size`)                                           |
| `dataset.size`                 | `[width, height]`                                   | `[1024, 1024]` | Fixed `[width, height]` for validation and (unless `dynamic_size` is set) training                          |
| `dataset.dynamic_size`         | `null` or `[[min_w, max_w], [min_h, max_h]]`        | `null`        | Random per-batch training size drawn from multiples of 32 inside each range                                 |
| `dataset.background_synthesis` | `bool`                                              | `false`       | Replaces training backgrounds with synthetic colors before other augmentations                              |
| `optimizer.kind`               | `"adamw"`, `"adam"`, `"sgd"`                        | `"adamw"`     | `adamw` uses decoupled weight decay; `adam` and `sgd` use L2 weight decay                                   |
| `optimizer.learning_rate`      | `f64` (`> 0`, finite)                               | `1e-4`        | Base learning rate                                                                                          |
| `optimizer.weight_decay`       | `f64` (`>= 0`, finite)                              | `1e-2`        | Weight decay coefficient                                                                                    |
| `optimizer.lr_decay_epochs`    | `Vec<i64>`                                          | `[]`          | Epoch milestones for `MultiStepLR`; negative values count back from `num_epochs`                            |
| `optimizer.lr_decay_rate`      | `f64` (`> 0`, finite)                               | `0.5`         | Multiplicative LR decay factor applied at each milestone                                                    |
| `num_epochs`                   | `usize` (`>= 1`)                                    | `120`         | Total training epochs                                                                                       |
| `batch_size`                   | `usize` (`>= 1`)                                    | `4`           | Samples per batch                                                                                           |
| `num_workers`                  | `usize`                                             | `4`           | Data loader worker threads                                                                                  |
| `keep_checkpoints`             | `usize`                                             | `2`           | Number of latest checkpoints kept (`max(1)` at runtime)                                                     |
| `gradient_checkpointing`       | `bool`                                              | `false`       | Recomputes activations in backward pass to reduce memory                                                    |
| `seed`                         | `u64`                                               | `42`          | Seeds device RNG, dataset shuffling, dynamic sizing, and per-split augmentors                               |

## Augmentations, Losses, and Artifacts

- **Default augmentations (`birefnet train`)**: `AugmentationConfig::new()` applies `Flip`
  (50% horizontal flip), `Enhance` (brightness/contrast/hue/sharpness), `Rotate` (20% chance,
  ±15°), and `Pepper` (0.15% salt-and-pepper noise), matching upstream `preproc_methods`. `Crop`
  is available in `birefnet-train` via `AugmentationConfig::with_methods`.
- **Default training loss (`birefnet train`)**: `BiRefNetLossConfig::new()` uses the upstream
  `General` weights (`bce = 30.0`, `iou = 0.5`, `mae = 100.0`, `ssim = 10.0`, auxiliary
  `classification_weight = 5.0`). Library callers can pick `dis_optimized()`,
  `matting_optimized()`, or custom `LossWeightsConfig` values.
- **Validation metrics**: Logged per epoch for `Loss`, `MAE`, and `Adaptive F-measure`
  (`β² = 0.3`).
- **Artifacts written to `--artifact-dir`**:
  - `config.json` — resolved `TrainingConfig` snapshot
  - `model_config.json` — resolved `ModelConfig` (auto-discovered by `birefnet infer`)
  - `checkpoint/` — Burn checkpoints retained for `--resume <epoch>`
  - `train/` & `valid/` — Burn metric logs
  - `model.bpk` — final Burnpack weights loadable via `birefnet infer -m <artifact_dir>/model.bpk`
