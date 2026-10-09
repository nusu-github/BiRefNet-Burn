use anyhow::Result;
use birefnet::device::{AVAILABLE_DEVICES, Precision, configure_precision, select_device};
use birefnet_util::ManagedModel;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "birefnet")]
#[command(
    about = "BiRefNet: Bilateral Reference Network for high-resolution dichotomous image segmentation"
)]
struct Cli {
    /// Device to run on (see `birefnet info` for the devices compiled into this build).
    /// `default` picks the first compiled-in backend, or the one named by `BURN_DEVICE`.
    #[arg(long, global = true, default_value = "default")]
    device: String,

    /// Float precision for every tensor on the device (defaults to the backend's, f32).
    #[arg(long, global = true, value_enum)]
    precision: Option<Precision>,

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

        /// Directory for checkpoints, metric logs and the final model
        #[arg(long, default_value = "./artifacts")]
        artifact_dir: String,

        /// Resume after this epoch from the checkpoints in the artifact directory
        #[arg(short, long)]
        resume: Option<usize>,
    },

    /// Show backend information
    Info,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    let mut device = select_device(&cli.device)?;
    // Device settings lock on the first tensor, so configure before anything else runs.
    if let Some(precision) = cli.precision {
        configure_precision(&mut device, precision)?;
    }
    tracing::info!(?device, "device selected");

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
        Commands::Train {
            config,
            artifact_dir,
            resume,
        } => {
            use birefnet::training::{TrainingCliArgs, run_training};

            let args = TrainingCliArgs::new(config, artifact_dir, resume);
            run_training(&args, &device)
        }

        Commands::Info => {
            println!("BiRefNet Information:");
            println!("  Available devices: {}", AVAILABLE_DEVICES.join(", "));
            println!("  Selected device: {device:?}");
            println!("  Float dtype: {:?}", device.settings().float_dtype);
            Ok(())
        }
    }
}
