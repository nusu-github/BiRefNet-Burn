//! CLI inference runner delegating segmentation and weight loading to `birefnet`.

use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::Result;
use birefnet::{
    inference::{Segmenter, SegmenterConfig},
    model::BiRefNet,
    util::is_supported_image_format,
    weights::{BiRefNetWeightLoading, ManagedModel},
};
use burn::tensor::Device;

pub(crate) struct InferenceCliArgs {
    pub(crate) input_path: PathBuf,
    pub(crate) output_path: PathBuf,
    pub(crate) model_spec: String,
    pub(crate) refine_foreground: bool,
}

impl InferenceCliArgs {
    pub(crate) const fn new(
        input_path: PathBuf,
        output_path: PathBuf,
        model_spec: String,
        refine_foreground: bool,
    ) -> Self {
        Self {
            input_path,
            output_path,
            model_spec,
            refine_foreground,
        }
    }
}

fn input_size_for(managed: &ManagedModel) -> (usize, usize) {
    managed
        .default_resolution()
        .map_or((1024, 1024), |(height, width)| {
            (height as usize, width as usize)
        })
}

fn load_model_and_size(model_spec: &str, device: &Device) -> Result<(BiRefNet, (usize, usize))> {
    let local_path = Path::new(model_spec);
    if local_path.exists() {
        let managed = ManagedModel::from_local("custom", local_path, None);
        let input_size = input_size_for(&managed);
        let model = managed
            .build_config()
            .init(device)?
            .load_weights(local_path, device)?;
        return Ok((model, input_size));
    }

    let managed = ManagedModel::from_pretrained(model_spec)?;

    #[cfg(feature = "hub")]
    {
        let input_size = input_size_for(&managed);
        if !managed.is_available() {
            tracing::info!(model = %model_spec, "downloading pretrained weights from HuggingFace Hub");
        }
        let model = BiRefNet::from_managed_model(&managed, device)?;
        Ok((model, input_size))
    }

    #[cfg(not(feature = "hub"))]
    {
        anyhow::bail!(
            "pretrained model `{}` requires downloading from HuggingFace Hub; rebuild `birefnet-cli` with `--features hub` or pass a local checkpoint path",
            managed.name()
        );
    }
}

pub(crate) fn run_inference(args: &InferenceCliArgs, device: &Device) -> Result<()> {
    tracing::info!(
        input = %args.input_path.display(),
        output = %args.output_path.display(),
        model = %args.model_spec,
        refine_foreground = args.refine_foreground,
        "running inference",
    );

    let (model, input_size) = load_model_and_size(&args.model_spec, device)?;
    let segmenter = Segmenter::new(
        model,
        SegmenterConfig {
            input_size,
            refine_foreground: args.refine_foreground,
            ..SegmenterConfig::default()
        },
    );

    fs::create_dir_all(&args.output_path)?;

    if args.input_path.is_file() {
        process_single_image(&segmenter, &args.input_path, &args.output_path, device)?;
    } else if args.input_path.is_dir() {
        process_directory(&segmenter, &args.input_path, &args.output_path, device)?;
    } else {
        anyhow::bail!("input path does not exist: {}", args.input_path.display());
    }

    tracing::info!("inference completed");
    Ok(())
}

fn process_single_image(
    segmenter: &Segmenter,
    input_path: &Path,
    output_dir: &Path,
    device: &Device,
) -> Result<()> {
    let output = segmenter.segment_file(input_path, device)?;
    let stem = input_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    let output_path = output_dir.join(format!("{stem}_mask.png"));
    output.to_rgba_image()?.save(&output_path)?;
    tracing::info!(path = %output_path.display(), "saved segmentation result");
    Ok(())
}

fn process_directory(
    segmenter: &Segmenter,
    input_dir: &Path,
    output_dir: &Path,
    device: &Device,
) -> Result<()> {
    let mut entries: Vec<PathBuf> = fs::read_dir(input_dir)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_file() && is_supported_image_format(path))
        .collect();
    entries.sort();

    for path in entries {
        if let Err(err) = process_single_image(segmenter, &path, output_dir, device) {
            tracing::error!(path = %path.display(), error = %err, "failed to process image");
        }
    }

    Ok(())
}
