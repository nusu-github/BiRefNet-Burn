use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::Result;
use birefnet_util::{
    BiRefNetWeightLoading, ManagedModel, ModelLoader, ModelName, WeightSource,
    apply_imagenet_normalization, apply_mask, is_supported_image_format, load_image,
    refine_foreground_core, tensor_to_dynamic_image,
};
use burn::tensor::{
    activation::sigmoid,
    module::interpolate,
    ops::{InterpolateMode, InterpolateOptions},
};

use crate::burn_backend_types::{InferenceBackend, InferenceDevice};

/// Model input resolution used for `BiRefNet` inference.
///
/// `BiRefNet` is trained on 1024x1024 images; using a different size
/// may degrade segmentation quality.
const MODEL_INPUT_SIZE: usize = 1024;

/// Blur radius for foreground refinement matting.
///
/// Higher values produce smoother edges but increase processing time.
/// 90 provides a good balance for typical photographic subjects.
const FOREGROUND_REFINE_RADIUS: usize = 90;

/// Inference configuration.
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Input path (file or directory).
    pub input_path: PathBuf,
    /// Output directory.
    pub output_path: PathBuf,
    /// Model specification (pretrained model name or file path).
    pub model_spec: String,
}

impl InferenceConfig {
    /// Creates a new inference configuration.
    pub fn new(
        input_path: impl Into<PathBuf>,
        output_path: impl Into<PathBuf>,
        model_spec: String,
    ) -> Self {
        Self {
            input_path: input_path.into(),
            output_path: output_path.into(),
            model_spec,
        }
    }
}

/// Runs `BiRefNet` inference on an image or directory of images.
///
/// Loads the specified model, processes each image through the network,
/// and saves foreground-matted results to the output directory.
///
/// # Errors
///
/// Returns an error if the model cannot be loaded, the input path does
/// not exist, or an image fails to process.
pub fn run_inference(config: &InferenceConfig, device: &InferenceDevice) -> Result<()> {
    use birefnet_model::BiRefNet;

    tracing::info!(
        input = %config.input_path.display(),
        output = %config.output_path.display(),
        model = %config.model_spec,
        "running inference",
    );

    // Determine if model_spec is a pretrained model name or a file path
    let managed_model = if Path::new(&config.model_spec).exists() {
        let model_name = ModelName::new("custom");
        let weights = WeightSource::Local {
            path: PathBuf::from(&config.model_spec),
        };
        ManagedModel::new(model_name, None, weights)
    } else {
        ManagedModel::from_pretrained(&config.model_spec)?
    };

    if !<ManagedModel as ModelLoader<InferenceBackend>>::is_available(&managed_model) {
        anyhow::bail!("Model weights are not available. Please check your model specification.");
    }

    tracing::info!("loading model");
    let model: BiRefNet<InferenceBackend> = BiRefNet::from_managed_model(&managed_model, device)?;
    tracing::info!("model loaded successfully");

    fs::create_dir_all(&config.output_path)?;

    if config.input_path.is_file() {
        process_single_image(&model, &config.input_path, &config.output_path, device)?;
    } else if config.input_path.is_dir() {
        process_directory(&model, &config.input_path, &config.output_path, device)?;
    } else {
        anyhow::bail!(
            "Input path does not exist: {}",
            config.input_path.display()
        );
    }

    tracing::info!("inference completed");
    Ok(())
}

/// Processes inference for a single image.
fn process_single_image(
    model: &birefnet_model::BiRefNet<InferenceBackend>,
    input_path: &Path,
    output_dir: &Path,
    device: &InferenceDevice,
) -> Result<()> {
    tracing::info!(path = %input_path.display(), "processing image");

    let image = load_image(input_path, device)?;
    let [_, _, h, w] = image.dims();

    let image_normalized = apply_imagenet_normalization(image.clone())?;
    let image_tensor = interpolate(
        image_normalized,
        [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
        InterpolateOptions::new(InterpolateMode::Bicubic),
    );

    let mask = model.forward(image_tensor)?;
    let mask = sigmoid(mask);

    let mask = interpolate(
        mask,
        [h, w],
        InterpolateOptions::new(InterpolateMode::Bicubic),
    );

    let file_stem = input_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    let output_path = output_dir.join(format!("{file_stem}_mask.png"));

    let output = refine_foreground_core(image, mask.clone(), FOREGROUND_REFINE_RADIUS);
    let output = apply_mask(output, mask)?;
    let output_image = tensor_to_dynamic_image(output, false)?;
    output_image.save(&output_path)?;

    tracing::info!(path = %output_path.display(), "saved result");
    Ok(())
}

/// Processes inference for all images in a directory.
fn process_directory(
    model: &birefnet_model::BiRefNet<InferenceBackend>,
    input_dir: &Path,
    output_dir: &Path,
    device: &InferenceDevice,
) -> Result<()> {
    for entry in fs::read_dir(input_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file()
            && is_supported_image_format(&path)
            && let Err(e) = process_single_image(model, &path, output_dir, device)
        {
            tracing::error!(path = %path.display(), error = %e, "failed to process image");
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use birefnet_util::is_extension_supported;

    use super::*;

    #[test]
    fn is_supported_image_format_returns_correct_results() {
        assert!(is_supported_image_format("test.jpg"));
        assert!(is_supported_image_format("test.jpeg"));
        assert!(is_supported_image_format("test.png"));
        assert!(is_supported_image_format("test.bmp"));

        assert!(is_supported_image_format("test.JPG"));
        assert!(is_supported_image_format("test.PNG"));

        assert!(!is_supported_image_format("test.txt"));
        assert!(!is_supported_image_format("test.pdf"));

        assert!(!is_supported_image_format("test"));

        assert!(is_supported_image_format("/path/to/image.jpg"));
        assert!(is_supported_image_format("./relative/path/image.png"));
    }

    #[test]
    fn image_extension_support_works_correctly() {
        assert!(is_extension_supported("jpg"));
        assert!(is_extension_supported("jpeg"));
        assert!(is_extension_supported("png"));

        assert!(is_extension_supported(".jpg"));
        assert!(is_extension_supported(".png"));

        assert!(!is_extension_supported("txt"));
        assert!(!is_extension_supported("pdf"));
    }

    #[test]
    fn inference_config_creation_works() {
        let config = InferenceConfig::new("input.jpg", "output/", "General".to_string());

        assert_eq!(config.input_path, PathBuf::from("input.jpg"));
        assert_eq!(config.output_path, PathBuf::from("output/"));
        assert_eq!(config.model_spec, "General");
    }
}
