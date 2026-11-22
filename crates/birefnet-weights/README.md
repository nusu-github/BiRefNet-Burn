# birefnet-weights

Pretrained BiRefNet model presets, Hugging Face Hub fetching, and checkpoint loading.

| Feature | Default | Effect |
| --- | --- | --- |
| `hub` | no | `HfHubFetcher`, a blocking Hugging Face Hub [`Fetcher`] |
| `test-util` | no | `MockFetcher`, an offline [`Fetcher`] for tests |

Checkpoint formats are detected from the file header, not the extension:

- **Burnpack**, written by `burn::module::Module::save_file`;
- **SafeTensors**, the format of the upstream Hugging Face releases;
- **PyTorch** `torch.save` archives (files only).

Upstream PyTorch parameter names are remapped to the Burn module tree, and parameters are
cast to the device's default float dtype. [`MODEL_SPECS`] lists the 15 official presets
(`General`, `General-HR`, `Matting`, `General-Lite`, `DIS`, ...).

## Examples

```rust
use birefnet_weights::{WeightFormat, load_weights_from_bytes, load_weights_from_path, model_spec};
use burn::{module::Module, nn::LinearConfig, prelude::*};

let spec = model_spec("General-Lite").expect("registered preset");
assert_eq!(spec.repo_id(), "ZhengPeng7/BiRefNet_lite");

// Any Burn module round-trips through a Burnpack checkpoint, on disk or in memory.
let device = Device::flex();
let dir = std::env::temp_dir().join(format!("birefnet-weights-doc-{}", std::process::id()));
std::fs::create_dir_all(&dir)?;
let path = dir.join("linear.bpk");
let saved = LinearConfig::new(4, 2).init(&device);
saved.clone().save_file(&path)?;

assert_eq!(WeightFormat::from_path(&path)?, WeightFormat::Burnpack);
let from_file = load_weights_from_path(LinearConfig::new(4, 2).init(&device), &path, &device)?;
let from_bytes =
    load_weights_from_bytes(LinearConfig::new(4, 2).init(&device), std::fs::read(&path)?, &device)?;
from_file.weight.val().into_data().assert_eq(&saved.weight.val().into_data(), true);
from_bytes.weight.val().into_data().assert_eq(&saved.weight.val().into_data(), true);
# std::fs::remove_dir_all(&dir)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Pretrained presets are fetched through any [`Fetcher`]; with the `hub` feature:

```rust,ignore
use birefnet_model::BiRefNet;
use birefnet_weights::{BiRefNetWeightLoading, HfHubFetcher, ManagedModel};

let fetcher = HfHubFetcher::new()?;
let managed = ManagedModel::from_pretrained("General")?;
let model = BiRefNet::from_managed(&managed, &fetcher, &burn::tensor::Device::flex())?;
```

## License

MIT OR Apache-2.0
