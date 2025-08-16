use std::path::{Path, PathBuf};

use anyhow::Result;
use birefnet::{create_device, get_backend_name, SelectedBackend};
#[cfg(feature = "inference")]
use birefnet_util::ImageUtils;
use birefnet_util::{BiRefNetWeightLoading, ManagedModel, ModelLoader, ModelName, WeightSource};
use burn::tensor::{
    module::interpolate,
    ops::{InterpolateMode, InterpolateOptions},
};
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "birefnet")]
#[command(
    about = "BiRefNet: Bilateral Reference Network for high-resolution dichotomous image segmentation"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run inference on images
    #[cfg(feature = "inference")]
    Infer {
        /// Input image path or directory
        #[arg(short, long)]
        input: String,

        /// Output directory for results
        #[arg(short, long)]
        output: String,

        /// Pretrained model name (e.g. "General", "General-HR", "Matting") or path to model file
        #[arg(short, long)]
        model: String,

        /// List available pretrained models
        #[arg(long)]
        list_models: bool,
    },

    /// Train a BiRefNet model
    #[cfg(feature = "train")]
    Train {
        /// Training configuration file
        #[arg(short, long)]
        config: String,

        /// Resume from checkpoint
        #[arg(short, long)]
        resume: Option<String>,
    },

    /// Show backend information
    Info,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize device
    let device = create_device();
    println!("Using backend: {}", get_backend_name());

    match cli.command {
        #[cfg(feature = "inference")]
        Commands::Infer {
            input,
            output,
            model,
            list_models,
        } => {
            if list_models {
                println!("Available pretrained models:");
                for model_name in ManagedModel::list_available_models() {
                    println!("  - {}", model_name);
                }
                return Ok(());
            }

            println!("Running inference:");
            println!("  Input: {input}");
            println!("  Output: {output}");
            println!("  Model: {model}");

            run_inference(&input, &output, &model, device)?;
            Ok(())
        }

        #[cfg(feature = "train")]
        Commands::Train { config, resume } => {
            println!("Starting training:");
            println!("  Config: {}", config);
            if let Some(checkpoint) = resume {
                println!("  Resume from: {}", checkpoint);
            }
            // TODO: Implement training logic
            Ok(())
        }

        Commands::Info => {
            println!("BiRefNet Information:");
            println!("  Backend: {}", get_backend_name());
            println!("  Device: {device:?}");
            Ok(())
        }
    }
}

#[cfg(feature = "inference")]
fn run_inference(
    input_path: &str,
    output_path: &str,
    model_spec: &str,
    device: birefnet::SelectedDevice,
) -> Result<()> {
    use birefnet_model::BiRefNet;

    // Determine if model_spec is a pretrained model name or a file path
    let managed_model = if Path::new(model_spec).exists() {
        // It's a file path
        let model_name = ModelName::new("custom");
        let weights = WeightSource::Local {
            path: PathBuf::from(model_spec),
        };
        ManagedModel::new(model_name, None, weights)
    } else {
        // It's a pretrained model name
        ManagedModel::from_pretrained(model_spec)?
    };

    // Check if weights are available
    if !<ManagedModel as ModelLoader<SelectedBackend>>::is_available(&managed_model) {
        anyhow::bail!("Model weights are not available. Please check your model specification.");
    }

    println!("Loading model...");

    // Load the model with weights
    let model: BiRefNet<SelectedBackend> = BiRefNet::from_managed_model(&managed_model, &device)?;

    println!("Model loaded successfully!");

    // Create output directory if it doesn't exist
    std::fs::create_dir_all(output_path)?;

    let input_path = Path::new(input_path);
    if input_path.is_file() {
        // Process single image
        process_single_image(&model, input_path, output_path, &device)?;
    } else if input_path.is_dir() {
        // Process all images in directory
        process_directory(&model, input_path, output_path, &device)?;
    } else {
        anyhow::bail!("Input path does not exist: {}", input_path.display());
    }

    println!("Inference completed!");
    Ok(())
}

#[cfg(feature = "inference")]
fn process_single_image(
    model: &birefnet_model::BiRefNet<SelectedBackend>,
    input_path: &Path,
    output_dir: &str,
    device: &birefnet::SelectedDevice,
) -> Result<()> {
    println!("Processing: {}", input_path.display());

    // Load and preprocess image
    let image_tensor = ImageUtils::load_image(input_path, device)?;

    let image_tensor = interpolate(
        image_tensor,
        [1024, 1024],
        InterpolateOptions::new(InterpolateMode::Bicubic),
    );

    // Run inference
    let output = model.forward(image_tensor)?;

    // Get the output file name
    let file_stem = input_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    let output_path = Path::new(output_dir).join(format!("{}_mask.png", file_stem));

    // Convert tensor to image and save (output is a mask, so is_mask = true)
    let output_image = ImageUtils::tensor_to_dynamic_image(output, true)?;
    output_image.save(&output_path)?;

    println!("Saved: {}", output_path.display());
    Ok(())
}

#[cfg(feature = "inference")]
fn process_directory(
    model: &birefnet_model::BiRefNet<SelectedBackend>,
    input_dir: &Path,
    output_dir: &str,
    device: &birefnet::SelectedDevice,
) -> Result<()> {
    let supported_extensions = ["jpg", "jpeg", "png", "bmp", "tiff", "webp"];

    for entry in std::fs::read_dir(input_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() {
            if let Some(extension) = path.extension().and_then(|s| s.to_str()) {
                if supported_extensions.contains(&extension.to_lowercase().as_str()) {
                    if let Err(e) = process_single_image(model, &path, output_dir, device) {
                        eprintln!("Error processing {}: {}", path.display(), e);
                    }
                }
            }
        }
    }

    Ok(())
}
