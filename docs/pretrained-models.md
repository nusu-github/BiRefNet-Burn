# Pretrained Models and Weight Loading

## Registered Hugging Face Presets

`birefnet-weights::MODEL_SPECS` registers all 15 official `ZhengPeng7` presets. Every preset
downloads `model.safetensors` from Hugging Face Hub when the `hub` feature is enabled.

| Preset Name        | Hugging Face Repository               | Backbone  | Task        | Default Resolution `[H, W]` | Dynamic Resolution |
| ------------------ | ------------------------------------- | --------- | ----------- | --------------------------- | ------------------ |
| `General`          | `ZhengPeng7/BiRefNet`                 | `SwinV1L` | `General`   | `1024x1024`                 | No                 |
| `General-HR`       | `ZhengPeng7/BiRefNet_HR`              | `SwinV1L` | `General2k` | `2048x2048`                 | No                 |
| `Matting-HR`       | `ZhengPeng7/BiRefNet_HR-matting`      | `SwinV1L` | `Matting`   | `2048x2048`                 | No                 |
| `Matting`          | `ZhengPeng7/BiRefNet-matting`         | `SwinV1L` | `Matting`   | `1024x1024`                 | No                 |
| `Portrait`         | `ZhengPeng7/BiRefNet-portrait`        | `SwinV1L` | `Matting`   | `1024x1024`                 | No                 |
| `General-reso_512` | `ZhengPeng7/BiRefNet_512x512`         | `SwinV1L` | `General`   | `512x512`                   | No                 |
| `General-Lite`     | `ZhengPeng7/BiRefNet_lite`            | `SwinV1T` | `General`   | `1024x1024`                 | No                 |
| `General-Lite-2K`  | `ZhengPeng7/BiRefNet_lite-2K`         | `SwinV1T` | `General2k` | `2048x2048`                 | No                 |
| `DIS`              | `ZhengPeng7/BiRefNet-DIS5K`           | `SwinV1L` | `DIS5K`     | `1024x1024`                 | No                 |
| `HRSOD`            | `ZhengPeng7/BiRefNet-HRSOD`           | `SwinV1L` | `HRSOD`     | `1024x1024`                 | No                 |
| `COD`              | `ZhengPeng7/BiRefNet-COD`             | `SwinV1L` | `COD`       | `1024x1024`                 | No                 |
| `DIS-TR_TEs`       | `ZhengPeng7/BiRefNet-DIS5K-TR_TEs`    | `SwinV1L` | `DIS5K`     | `1024x1024`                 | No                 |
| `General-legacy`   | `ZhengPeng7/BiRefNet-legacy`          | `SwinV1L` | `General`   | `1024x1024`                 | No                 |
| `General-dynamic`  | `ZhengPeng7/BiRefNet_dynamic`         | `SwinV1L` | `General`   | `1024x1024`                 | Yes                |
| `Matting-dynamic`  | `ZhengPeng7/BiRefNet_dynamic-matting` | `SwinV1L` | `Matting`   | `1024x1024`                 | Yes                |

Every preset builds its `BiRefNetConfig` via `ModelSpec::build_config()` using
`InterpolationStrategy::Bilinear`, default `DecoderConfig`, and `Refine::None`.

For quick experiments on CPU (`flex`), `General-Lite` (`SwinV1T`) or `General-reso_512`
(`512x512`) reduces memory and runtime compared to `General` or `*-HR`.

## Supported Checkpoint Formats

`birefnet-weights` detects formats from the first 9 header bytes (`WeightFormat::detect`),
ignoring file extensions:

| Format        | Header Signature                                | File Path | In-Memory Bytes | Notes                                                   |
| ------------- | ----------------------------------------------- | --------- | --------------- | ------------------------------------------------------- |
| `Burnpack`    | Little-endian `0x4255524E` (`"BURN"`)           | Yes       | Yes             | Native Burn `Module::save_file` format (`.bpk`)         |
| `SafeTensors` | 8-byte `u64` header length followed by `{`      | Yes       | Yes             | Upstream Hugging Face format; remapped and dtype-cast   |
| `PyTorch`     | Zip (`PK\x03\x04`) or legacy pickle (`0x80`)    | Yes       | No              | Upstream `.pt`/`.pth`; remapped and dtype-cast          |

Legacy Burn recorder formats (`.mpk`, `.bin`) are not supported.

### Key Remapping and Dtype Casting

When loading SafeTensors or PyTorch checkpoints:

- Upstream PyTorch keys are transformed by `PyTorchToBurnAdapter` plus 8 regex rules in
  `BIREFNET_KEY_REMAPPINGS` ([`crates/birefnet-weights/src/loader.rs`](../crates/birefnet-weights/src/loader.rs))
  covering `decoder.conv_out1.0.*`, `decoder.gdt_convs_attn_{2..4}.0.*`,
  `decoder.gdt_convs_pred_{2..4}.0.*`, `bb.norm{0..3}.*`, `*.gdt_convs_{2..4}.{0,1}.*`, and
  `*.global_avg_pool.{1,2}.*`.
- Missing keys or apply errors fail loudly; unused upstream metadata keys (such as PyTorch
  `num_batches_tracked`) are logged at `DEBUG` level.
- Floating-point parameters are cast to the active device's configured float dtype
  (`f32`, `f16`, or `bf16`).

## Local Checkpoints in `birefnet-cli`

When `--model` points to an existing local file, `birefnet infer` resolves its architecture in the
following precedence order:

1. Explicit `--config <model_config.json>`
2. Sibling `model_config.json` in the same directory as the checkpoint (written automatically by
   `birefnet train`)
3. Explicit `--preset <name>` (also sets the preset's default inference resolution)
4. Fallback: `General` architecture at `1024x1024`

`--config` and `--preset` are mutually exclusive and apply only to local checkpoint files.

## Offline Fetching and Custom Storage

Remote downloads go through the `birefnet_weights::Fetcher` trait:

- `HfHubFetcher` (enabled by the `hub` feature) downloads and caches files from Hugging Face Hub.
- `MockFetcher` (enabled by the `test-util` feature) serves local fixtures for offline tests.
- Callers can implement `Fetcher` for custom caches, object storage, or air-gapped environments.
