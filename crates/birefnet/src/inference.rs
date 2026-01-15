use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::Result;
use birefnet_util::{
    BiRefNetWeightLoading, ImageUtils, ManagedModel, ModelLoader, ModelName, WeightSource,
    refine_foreground_core,
};
use burn::tensor::{
    activation::sigmoid,
    module::interpolate,
    ops::{InterpolateMode, InterpolateOptions},
};

use crate::burn_backend_types::{InferenceBackend, InferenceDevice};

/// Inference configuration
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Input path (file or directory)
    pub input_path: String,
    /// Output directory
    pub output_path: String,
    /// Model specification (pretrained model name or file path)
    pub model_spec: String,
}

impl InferenceConfig {
    pub const fn new(input_path: String, output_path: String, model_spec: String) -> Self {
        Self {
            input_path,
            output_path,
            model_spec,
        }
    }
}

/// Run inference using BiRefNet model
///
/// # Arguments
/// * `config` - Inference configuration
/// * `device` - Device to use for inference
///
/// # Returns
/// * `Result<()>` - `Ok(())` on success, error on failure
pub fn run_inference(config: InferenceConfig, device: InferenceDevice) -> Result<()> {
    use birefnet_model::BiRefNet;

    println!("Running inference:");
    println!("  Input: {}", config.input_path);
    println!("  Output: {}", config.output_path);
    println!("  Model: {}", config.model_spec);

    // Determine if model_spec is a pretrained model name or a file path
    let managed_model = if Path::new(&config.model_spec).exists() {
        // It's a file path
        let model_name = ModelName::new("custom");
        let weights = WeightSource::Local {
            path: PathBuf::from(&config.model_spec),
        };
        ManagedModel::new(model_name, None, weights)
    } else {
        // It's a pretrained model name
        ManagedModel::from_pretrained(&config.model_spec)?
    };

    // Check if weights are available
    if !<ManagedModel as ModelLoader<InferenceBackend>>::is_available(&managed_model) {
        anyhow::bail!("Model weights are not available. Please check your model specification.");
    }

    println!("Loading model...");

    // Load the model with weights
    let model: BiRefNet<InferenceBackend> = BiRefNet::from_managed_model(&managed_model, &device)?;

    println!("Model loaded successfully!");

    // Create output directory if it doesn't exist
    fs::create_dir_all(&config.output_path)?;

    let input_path = Path::new(&config.input_path);
    if input_path.is_file() {
        // Process single image
        process_single_image(&model, input_path, &config.output_path, &device)?;
    } else if input_path.is_dir() {
        // Process all images in directory
        process_directory(&model, input_path, &config.output_path, &device)?;
    } else {
        anyhow::bail!("Input path does not exist: {}", input_path.display());
    }

    println!("Inference completed!");
    Ok(())
}

/// Process inference for a single image
fn process_single_image(
    model: &birefnet_model::BiRefNet<InferenceBackend>,
    input_path: &Path,
    output_dir: &str,
    device: &InferenceDevice,
) -> Result<()> {
    println!("Processing: {}", input_path.display());

    // Load and preprocess image with ImageNet normalization
    let image = ImageUtils::load_image(input_path, device)?;
    let [_, _, h, w] = image.dims();

    let image_nomalized = ImageUtils::apply_imagenet_normalization(image.clone())?;
    let image_tensor = interpolate(
        image_nomalized,
        [1024, 1024],
        InterpolateOptions::new(InterpolateMode::Bicubic),
    );

    // Run inference
    let mask = model.forward(image_tensor)?;
    let mask = sigmoid(mask);

    let mask = interpolate(
        mask,
        [h, w],
        InterpolateOptions::new(InterpolateMode::Bicubic),
    );

    // Get the output file name
    let file_stem = input_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    let output_path = Path::new(output_dir).join(format!("{}_mask.png", file_stem));

    // Convert tensor to image and save
    let output = refine_foreground_core(image, mask.clone(), 90);
    let output = ImageUtils::apply_mask(output, mask)?;
    let output_image = ImageUtils::tensor_to_dynamic_image(output, false)?;
    output_image.save(&output_path)?;

    println!("Saved: {}", output_path.display());
    Ok(())
}

/// Process inference for all images in a directory
fn process_directory(
    model: &birefnet_model::BiRefNet<InferenceBackend>,
    input_dir: &Path,
    output_dir: &str,
    device: &InferenceDevice,
) -> Result<()> {
    for entry in fs::read_dir(input_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() && ImageUtils::is_supported_image_format(&path) {
            if let Err(e) = process_single_image(model, &path, output_dir, device) {
                eprintln!("Error processing {}: {}", path.display(), e);
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_supported_image_format_returns_correct_results() {
        // Supported extensions
        assert!(ImageUtils::is_supported_image_format("test.jpg"));
        assert!(ImageUtils::is_supported_image_format("test.jpeg"));
        assert!(ImageUtils::is_supported_image_format("test.png"));
        assert!(ImageUtils::is_supported_image_format("test.bmp"));

        // Case insensitive
        assert!(ImageUtils::is_supported_image_format("test.JPG"));
        assert!(ImageUtils::is_supported_image_format("test.PNG"));

        // Unsupported extensions
        assert!(!ImageUtils::is_supported_image_format("test.txt"));
        assert!(!ImageUtils::is_supported_image_format("test.pdf"));

        // No extension
        assert!(!ImageUtils::is_supported_image_format("test"));

        // Complex paths
        assert!(ImageUtils::is_supported_image_format("/path/to/image.jpg"));
        assert!(ImageUtils::is_supported_image_format(
            "./relative/path/image.png"
        ));
    }

    #[test]
    fn image_extension_support_works_correctly() {
        // Test that common formats are supported
        assert!(ImageUtils::is_extension_supported("jpg"));
        assert!(ImageUtils::is_extension_supported("jpeg"));
        assert!(ImageUtils::is_extension_supported("png"));

        // Test with dot prefix
        assert!(ImageUtils::is_extension_supported(".jpg"));
        assert!(ImageUtils::is_extension_supported(".png"));

        // Test unsupported formats
        assert!(!ImageUtils::is_extension_supported("txt"));
        assert!(!ImageUtils::is_extension_supported("pdf"));
    }

    #[test]
    fn inference_config_creation_works() {
        let config = InferenceConfig::new(
            "input.jpg".to_string(),
            "output/".to_string(),
            "General".to_string(),
        );

        assert_eq!(config.input_path, "input.jpg");
        assert_eq!(config.output_path, "output/");
        assert_eq!(config.model_spec, "General");
    }
}
