# birefnet-weights

Pretrained BiRefNet model presets, Hugging Face Hub fetching, and checkpoint loading on Burn.

## Features

| Feature     | Default | Effect                                                  |
| ----------- | ------- | ------------------------------------------------------- |
| `hub`       | no      | `HfHubFetcher`, a blocking Hugging Face Hub `Fetcher`   |
| `test-util` | no      | `MockFetcher`, an offline `Fetcher` for tests           |

## Checkpoint Formats

Formats are detected from the file header (`WeightFormat::detect`), not from the file extension:

- **Burnpack** (`.bpk`), written by `burn::module::Module::save_file` (files and bytes)
- **SafeTensors** (`.safetensors`), used by the upstream Hugging Face releases (files and bytes)
- **PyTorch** (`.pt` / `.pth`), `torch.save` zip archives or legacy pickle (files only)

Upstream PyTorch parameter names are remapped to the Burn module tree, and floating-point
parameters are cast to the device's default float dtype. `MODEL_SPECS` lists all 15 official
presets (`General`, `General-HR`, `Matting`, `General-Lite`, `DIS`, etc.; see
[`docs/pretrained-models.md`](../../docs/pretrained-models.md)).

## Examples

Inspecting a preset and round-tripping a Burnpack checkpoint:

```rust
use birefnet_weights::{WeightFormat, load_weights_from_bytes, load_weights_from_path, model_spec};
use burn::{module::Module, nn::LinearConfig, prelude::*};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let spec = model_spec("General-Lite").expect("registered preset");
    assert_eq!(spec.repo_id(), "ZhengPeng7/BiRefNet_lite");

    let device = Device::flex();
    let path = std::env::temp_dir().join("linear.bpk");
    let saved = LinearConfig::new(4, 2).init(&device);
    saved.clone().save_file(&path)?;

    assert_eq!(WeightFormat::from_path(&path)?, WeightFormat::Burnpack);
    let from_file = load_weights_from_path(LinearConfig::new(4, 2).init(&device), &path, &device)?;
    let from_bytes =
        load_weights_from_bytes(LinearConfig::new(4, 2).init(&device), std::fs::read(&path)?, &device)?;
    from_file.weight.val().into_data().assert_eq(&saved.weight.val().into_data(), true);
    from_bytes.weight.val().into_data().assert_eq(&saved.weight.val().into_data(), true);
    Ok(())
}
```

Downloading and loading a pretrained preset with the `hub` feature enabled:

```rust
use birefnet_model::BiRefNet;
use birefnet_weights::{BiRefNetWeightLoading, HfHubFetcher, ManagedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let fetcher = HfHubFetcher::new()?;
    let managed = ManagedModel::from_pretrained("General")?;
    let model = BiRefNet::from_managed(&managed, &fetcher, &burn::tensor::Device::flex())?;
    let _ = model;
    Ok(())
}
```

## License

MIT OR Apache-2.0
