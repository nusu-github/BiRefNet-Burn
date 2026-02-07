use anyhow::Result;
use birefnet::burn_backend_types::{InferenceDevice, NAME};
use birefnet_util::ManagedModel;
use clap::{Parser, Subcommand};
use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

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
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    let device = InferenceDevice::default();
    tracing::info!(backend = NAME, "device initialized");

    match cli.command {
        #[cfg(feature = "inference")]
        Commands::Infer {
            input,
            output,
            model,
            list_models,
        } => {
            use birefnet::inference::{InferenceConfig, run_inference};

            if list_models {
                println!("Available pretrained models:");
                for model_name in ManagedModel::list_available_models() {
                    println!("  - {model_name}");
                }
                return Ok(());
            }

            let inference_config = InferenceConfig::new(input, output, model);

            // Stack overflow workaround for large models on platforms
            // with small default stack sizes (e.g. Windows).
            stacker::grow(4096 * 1024, || run_inference(&inference_config, &device))?;
            Ok(())
        }

        #[cfg(feature = "train")]
        Commands::Train { config, resume } => {
            use birefnet::training::{TrainingCliArgs, run_training};

            let args = TrainingCliArgs::new(config, resume.map(std::path::PathBuf::from));
            run_training(args)
        }

        Commands::Info => {
            println!("BiRefNet Information:");
            println!("  Backend: {NAME}");
            println!("  Device: {device:?}");
            Ok(())
        }
    }
}
