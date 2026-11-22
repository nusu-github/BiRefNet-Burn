# Upstream Parity and Known Limitations

This document maps `BiRefNet-Burn` to the upstream PyTorch repository
([ZhengPeng7/BiRefNet](https://github.com/ZhengPeng7/BiRefNet)), describes what is numerically
verified, and lists current framework and implementation constraints.

## Upstream Component Mapping

| Upstream (`ZhengPeng7/BiRefNet`)         | Workspace Crate & Module                                           | Notes                                                                    |
| ---------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------ |
| `models/birefnet.py`                     | `birefnet-model` (`models/birefnet.rs`, `models/decoder/`)         | Encoder fusion, context concat, squeeze stage, bilateral decoder         |
| `models/modules/` (ASPP, DCNv2, lateral) | `birefnet-model` (`models/modules/`)                               | Includes Burn `deform_conv2d`-backed `DeformableConv2d`                  |
| `models/refinement/`                     | `birefnet-model` (`models/refinement/`)                            | `RefUNet`, `Refiner`, `RefinerPVTInChannels4` (`Refine::Itself` omitted) |
| `models/backbones/`                      | `birefnet-backbones` (`src/backbones/`)                            | `SwinV1` T/S/B/L, `PVTv2` B0–B5, `ResNet` 18–152, `VGG` 16/16-BN         |
| `config.py`                              | `birefnet-model` (`src/config/`)                                   | Typed enums, defaults, and `ModelConfig::validate()`                     |
| `loss.py`                                | `birefnet-loss`                                                    | `PixLoss`, `BiRefNetLoss`, and all component losses                      |
| `evaluation/metrics.py` (`py_sod_metrics`) | `birefnet-metric`                                                | Host-side evaluation + `SegmentationMetric` Burn adapter                 |
| `dataset.py`, `image_proc.py`            | `birefnet-train`                                                   | Dataset discovery, `preproc_methods`, background synthesis, dynamic size |
| `utils.py` (blur-fusion foreground)      | `birefnet-util` (`foreground.rs`, `filters.rs`, `morphology.rs`)   | Two-pass blur-fusion refinement, MATLAB Gaussian filter, EDT, morphology |
| `inference.py` / Hub loading             | `birefnet-weights`, `birefnet-inference`, `birefnet-cli`           | 15 official Hugging Face presets, SafeTensors/PT/Burnpack loaders        |

## Numerical Parity and Verification

### Evaluation Metrics (`py_sod_metrics`)

`birefnet-metric` is tested against reference values generated directly from the upstream
`py_sod_metrics` implementation:

- Generator script: [`crates/birefnet-metric/scripts/gen_upstream_reference.py`](../crates/birefnet-metric/scripts/gen_upstream_reference.py)
- Generated fixtures: [`crates/birefnet-metric/tests/support/upstream_reference.rs`](../crates/birefnet-metric/tests/support/upstream_reference.rs)
- Parity test: [`crates/birefnet-metric/tests/upstream_parity.rs`](../crates/birefnet-metric/tests/upstream_parity.rs)

Verified tolerances on 8-bit reference samples:

- `MAE`, `MSE`, `adpFm`, `adpEm`: `<= 1e-12`
- `S-measure`, `Weighted F-measure`, 256-bin `F-measure` / `E-measure` / `BIoU` curves, and
  dataset-level aggregated curve maxima: `<= 1e-9`

### Half-Precision and Backend Tests

- `crates/birefnet-inference/tests/half_precision.rs`,
  `crates/birefnet-metric/tests/half_precision.rs`, and
  `crates/birefnet-util/tests/half_precision.rs` verify `f16` / `bf16` dtype preservation and
  output consistency on Burn's `flex` backend.
- Pretrained weight download tests that touch `huggingface.co` are marked `#[ignore]`; offline
  loading paths are exercised with `birefnet_weights::MockFetcher`.

## Known Limitations and Caveats

### 1. Performance vs. PyTorch / ONNX Runtime

Burn's kernel optimization and graph fusion are still weaker than mature PyTorch/cuDNN/TensorRT
toolchains for large high-resolution vision models (`1024x1024` / `2048x2048` Swin-L with
deformable convolutions). Expect significantly slower inference and training compared to upstream
PyTorch. Treat this workspace as a research and reference implementation rather than a
latency-critical runtime.

### 2. CubeCL Interpolation Backward Pass (`Nearest` Only)

In Burn `0.22.1`:

- The `flex` CPU backend differentiates `Nearest`, `Bilinear`, and `Bicubic` interpolation.
- CubeCL backends (`cpu`, `wgpu`, `vulkan`, `metal`, `cuda`, `rocm`) implement the backward pass
  for `Nearest` interpolation only.

Implications:

- `birefnet-loss::PixLoss` already uses nearest-neighbor resizing when aligning multi-scale
  predictions to target masks.
- Inside `BiRefNet`, resizing is controlled by `ModelConfig::interpolation`. Pretrained preset
  configs (such as `ModelSpec::build_config()` and `"model": "General"` in `birefnet train`) use
  `InterpolationStrategy::Bilinear` for smooth inference masks. When training on a CubeCL backend,
  provide an explicit `ModelConfig` with `"interpolation": "Nearest"` (see
  [`docs/training.md`](training.md)).

### 3. Deep Module Stack Usage on Windows

Constructing, loading, or running deep Burn module trees (especially `SwinV1L`) recurses past
Windows' default 1 MiB main-thread stack and can trigger a stack overflow.

- `birefnet-cli` wraps inference and training in `stacker::grow(64 * 1024 * 1024, ...)` (64 MiB).
- Library callers embedding `birefnet` on Windows should run `BiRefNetConfig::init`, weight
  loading, and forward passes on a worker thread with a 64 MiB stack (or use `stacker`).

### 4. Unimplemented or Restricted Upstream Options

- **`Refine::Itself`**: Present in the enum for upstream schema symmetry, but rejected by
  `ModelConfig::validate()`. Use `Refine::None`, `RefUNet`, `Refiner`, or
  `RefinerPVTInChannels4`.
- **Backbone subset in `birefnet-model`**: `birefnet-backbones` implements `ResNet` 18/34/50/101/152
  and `PVTv2` B0–B5, while `birefnet-model::Backbone` exposes the subset used by upstream BiRefNet
  configs (`SwinV1T/S/B/L`, `PvtV2B0/B1/B2/B5`, `ResNet50`, `Vgg16`, `Vgg16Bn`).
- **Checkpoint formats**:
  - Legacy Burn recorder formats (`.mpk`, `.bin`) are not supported; Burn checkpoints use
    Burnpack (`.bpk`) via `burn::store`.
  - PyTorch checkpoints (`.pt` / `.pth`) can only be loaded from filesystem paths, not from
    in-memory byte slices (`load_weights_from_bytes` returns `WeightError::is_unsupported_format`).
- **Spatial size alignment**: Backbones have a total stride of `32`; `DynamicSize` training ranges
  and custom input sizes should respect multiples of 32 for clean multi-scale alignment.

### 5. Coordinate Ordering Convention (`[H, W]` vs. `[W, H]`)

Two conventions exist across the workspace (matching Burn tensors vs. `image` / upstream config):

- **Tensor / inference APIs use `[height, width]`**: `SegmenterConfig::with_input_size([h, w])`,
  `SegmentationOutput::original_size()`, `ModelSpec::default_resolution`, and tensor shapes
  `[B, C, H, W]`.
- **Training dataset / CLI config APIs use `[width, height]`**: `Resolution::fixed(width, height)`,
  `DynamicSize::new(width_range, height_range)`, `birefnet-util::image::resize_image_file`, and
  `dataset.size` / `dataset.dynamic_size` in `birefnet train`'s JSON config.
