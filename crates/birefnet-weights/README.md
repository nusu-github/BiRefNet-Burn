# birefnet-weights

**Model preset registry, Hugging Face Hub fetcher, and Burnpack / SafeTensors / PyTorch checkpoint loading for `BiRefNet`**

## Features

| Feature     | Default | Effect                                                                    |
| ----------- | ------- | ------------------------------------------------------------------------- |
| `hub`       | no      | Enables [`HfHubFetcher`] and `ManagedModel::resolve_path` / `is_available` via `hf-hub` |
| `test-util` | no      | Exposes [`MockFetcher`] for offline integration testing                   |

## Supported Formats

- **Burnpack (`.bpk`)**: Native Burn checkpoints saved with `Module::save_file`, automatically cast to the target device float dtype.
- **SafeTensors (`.safetensors`)**: Pretrained PyTorch SafeTensors checkpoints converted via `PyTorchToBurnAdapter`, device float dtype casting, and upstream `BiRefNet` parameter key remapping.
- **PyTorch (`.pt`, `.pth`)**: Legacy PyTorch checkpoints converted via `PytorchStore` and cast to the target device float dtype.
- **Auto-detection**: Unknown extensions attempt Burnpack, SafeTensors, then PyTorch in order.

## Pretrained Presets (`MODEL_SPECS`)

`MODEL_SPECS` includes all 15 official `ZhengPeng7` presets (`General`, `General-HR`, `Matting-HR`, `Matting`, `Portrait`, `General-reso_512`, `General-Lite`, `General-Lite-2K`, `DIS`, `HRSOD`, `COD`, `DIS-TR_TEs`, `General-legacy`, `General-dynamic`, `Matting-dynamic`).

## Usage

```rust,no_run
use birefnet_model::BiRefNet;
use birefnet_weights::{BiRefNetWeightLoading, ManagedModel};
use burn::tensor::Device;

let device = Device::flex();

// Load a local Burnpack checkpoint offline (works with default features)
let local = ManagedModel::from_local("custom", "artifacts/model.bpk", None);
let model = local.build_config().init(&device)?.load_weights("artifacts/model.bpk", &device)?;

// With `--features hub`, fetch and load a named Hugging Face preset
#[cfg(feature = "hub")]
{
    let managed = ManagedModel::from_pretrained("General")?;
    let pretrained = BiRefNet::from_managed_model(&managed, &device)?;
    let _ = pretrained;
}
# let _ = model;
# Ok::<(), birefnet_weights::WeightError>(())
```

## License

MIT OR Apache-2.0
