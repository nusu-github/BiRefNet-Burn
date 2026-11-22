//! `birefnet infer`: segment an image or a directory of images.

use std::{
    fs,
    num::NonZeroUsize,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use birefnet::{
    inference::{Segmenter, SegmenterConfig},
    model::{BiRefNet, BiRefNetConfig, ModelConfig},
    util::image::is_supported_image_format,
    weights::{BiRefNetWeightLoading, ManagedModel, ModelName, ModelSpec, model_spec},
};
use burn::{config::Config, tensor::Device};

/// File name of the architecture written next to trained checkpoints by `birefnet train`.
pub(crate) const MODEL_CONFIG_FILE: &str = "model_config.json";

/// Fallback inference resolution for checkpoints without a registered preset.
const DEFAULT_INPUT_SIZE: [usize; 2] = [1024, 1024];
/// First-pass foreground refinement radius (reference implementation value).
const REFINE_RADIUS: NonZeroUsize = NonZeroUsize::new(90).unwrap();

/// Where the model weights and architecture come from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ModelSource {
    /// Registered preset name or path to a local checkpoint.
    pub(crate) model: String,
    /// Architecture JSON (`ModelConfig`) for a local checkpoint.
    pub(crate) config: Option<PathBuf>,
    /// Registered preset whose architecture a local checkpoint uses.
    pub(crate) preset: Option<String>,
}

pub(crate) struct InferenceCliArgs {
    pub(crate) input_path: PathBuf,
    pub(crate) output_path: PathBuf,
    pub(crate) source: ModelSource,
    pub(crate) refine_foreground: bool,
}

/// Picks the architecture of a local checkpoint: `--config`, then a `model_config.json` next
/// to the checkpoint, then `--preset`, then the `General` layout.
fn local_architecture(
    checkpoint: &Path,
    source: &ModelSource,
) -> Result<(ManagedModel, [usize; 2])> {
    let sibling = checkpoint.with_file_name(MODEL_CONFIG_FILE);
    let config_path = source
        .config
        .clone()
        .or_else(|| sibling.is_file().then_some(sibling));
    let preset = source.preset.as_deref().map(|name| {
        model_spec(name).with_context(|| format!("unknown preset `{name}` for --preset"))
    });
    let preset = preset.transpose()?;

    let config = match &config_path {
        Some(path) => {
            let model_config = ModelConfig::load(path)
                .with_context(|| format!("failed to read model config `{}`", path.display()))?;
            tracing::info!(
                name: "birefnet.cli.infer.config",
                path = %path.display(),
                "using architecture from {}",
                path.display()
            );
            Some(BiRefNetConfig::new(model_config))
        }
        None => preset.map(ModelSpec::build_config),
    };
    let name = preset.map_or("custom", |spec| spec.name);
    let input_size = preset.map_or(DEFAULT_INPUT_SIZE, |spec| {
        spec.default_resolution.map(|v| v as usize)
    });
    Ok((
        ManagedModel::from_local(ModelName::new(name)?, checkpoint, config),
        input_size,
    ))
}

fn load_model(source: &ModelSource, device: &Device) -> Result<(BiRefNet, [usize; 2])> {
    let local_path = Path::new(&source.model);
    if local_path.is_file() {
        let (managed, input_size) = local_architecture(local_path, source)?;
        let model = managed
            .build_config()
            .init(device)?
            .load_weights(local_path, device)?;
        return Ok((model, input_size));
    }
    if source.config.is_some() || source.preset.is_some() {
        bail!(
            "`{}` is not a file; --config and --preset only apply to local checkpoints",
            source.model
        );
    }

    let managed = ManagedModel::from_pretrained(&source.model)?;
    let input_size = managed
        .default_resolution()
        .map_or(DEFAULT_INPUT_SIZE, |size| size.map(|v| v as usize));
    load_pretrained(&managed, device).map(|model| (model, input_size))
}

#[cfg(feature = "hub")]
fn load_pretrained(managed: &ManagedModel, device: &Device) -> Result<BiRefNet> {
    let fetcher = birefnet::weights::HfHubFetcher::new()?;
    if !managed.is_available(&fetcher) {
        tracing::info!(
            name: "birefnet.cli.infer.download",
            model = %managed.name(),
            "downloading {} from Hugging Face Hub",
            managed.weight_source()
        );
    }
    Ok(BiRefNet::from_managed(managed, &fetcher, device)?)
}

#[cfg(not(feature = "hub"))]
fn load_pretrained(managed: &ManagedModel, _device: &Device) -> Result<BiRefNet> {
    bail!(
        "pretrained model `{}` must be downloaded from Hugging Face Hub; rebuild birefnet-cli \
         with `--features hub` or pass a local checkpoint path",
        managed.name()
    )
}

pub(crate) fn run_inference(args: &InferenceCliArgs, device: &Device) -> Result<()> {
    let (model, input_size) = load_model(&args.source, device)?;
    let config = SegmenterConfig::new()
        .with_input_size(input_size)?
        .with_foreground_refinement(args.refine_foreground.then_some(REFINE_RADIUS));
    let segmenter = Segmenter::new(&model, config);

    fs::create_dir_all(&args.output_path).with_context(|| {
        format!(
            "failed to create output directory `{}`",
            args.output_path.display()
        )
    })?;

    if args.input_path.is_file() {
        process_image(&segmenter, &args.input_path, &args.output_path, device)
    } else if args.input_path.is_dir() {
        process_directory(&segmenter, &args.input_path, &args.output_path, device)
    } else {
        bail!("input path does not exist: {}", args.input_path.display());
    }
}

fn process_image(
    segmenter: &Segmenter,
    input_path: &Path,
    output_dir: &Path,
    device: &Device,
) -> Result<()> {
    let output = segmenter
        .segment_file(input_path, device)
        .with_context(|| format!("failed to segment `{}`", input_path.display()))?;
    let stem = input_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    let output_path = output_dir.join(format!("{stem}_mask.png"));
    output
        .to_rgba_image()?
        .save(&output_path)
        .with_context(|| format!("failed to write `{}`", output_path.display()))?;
    tracing::info!(
        name: "birefnet.cli.infer.saved",
        path = %output_path.display(),
        "saved {}",
        output_path.display()
    );
    Ok(())
}

fn process_directory(
    segmenter: &Segmenter,
    input_dir: &Path,
    output_dir: &Path,
    device: &Device,
) -> Result<()> {
    let mut images = Vec::new();
    for entry in fs::read_dir(input_dir)
        .with_context(|| format!("failed to read directory `{}`", input_dir.display()))?
    {
        let path = entry
            .with_context(|| format!("failed to list directory `{}`", input_dir.display()))?
            .path();
        if path.is_file() && is_supported_image_format(&path) {
            images.push(path);
        }
    }
    images.sort_unstable();
    if images.is_empty() {
        bail!("no supported images found in `{}`", input_dir.display());
    }

    let mut failed = 0_usize;
    for path in &images {
        if let Err(err) = process_image(segmenter, path, output_dir, device) {
            failed += 1;
            tracing::error!(
                name: "birefnet.cli.infer.failed",
                path = %path.display(),
                "{err:#}"
            );
        }
    }

    let total = images.len();
    if failed == total {
        bail!("all {total} images in `{}` failed", input_dir.display());
    }
    if failed > 0 {
        tracing::warn!(
            name: "birefnet.cli.infer.partial",
            failed,
            total,
            "{failed} of {total} images failed"
        );
    }
    Ok(())
}
