use anyhow::Result;
use birefnet::burn_backend_types::{InferenceDevice, NAME};
use birefnet_util::ManagedModel;
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
    let device = InferenceDevice::default();
    println!("Using backend: {}", NAME);

    match cli.command {
        #[cfg(feature = "inference")]
        Commands::Infer {
            input,
            output,
            model,
            list_models,
        } => {
            if list_models {
                println!("利用可能な事前学習済みモデル:");
                for model_name in ManagedModel::list_available_models() {
                    println!("  - {}", model_name);
                }
                return Ok(());
            }

            use birefnet::inference::{InferenceConfig, run_inference};

            let inference_config = InferenceConfig::new(input, output, model);

            // Stack overflow error workaround
            // Windows stack size is too small for large models
            stacker::grow(4096 * 1024, || run_inference(inference_config, device))?;
            Ok(())
        }

        #[cfg(feature = "train")]
        Commands::Train { config, resume } => {
            use birefnet::training::{TrainingConfig, run_training};

            let training_config = TrainingConfig::new(config, resume);
            run_training(training_config)
        }

        Commands::Info => {
            println!("BiRefNet Information:");
            println!("  Backend: {}", NAME);
            println!("  Device: {device:?}");
            Ok(())
        }
    }
}
