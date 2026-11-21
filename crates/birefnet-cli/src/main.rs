//! Command-line entrypoint for `BiRefNet` inference, training, and backend diagnostics.

mod inference;
#[cfg(feature = "train")]
mod training;

use std::path::PathBuf;

use anyhow::Result;
use birefnet::{
    device::{AVAILABLE_DEVICES, Precision, configure_precision, select_device},
    weights::list_available_models,
};
use clap::{Parser, Subcommand};

const WORKER_STACK_BYTES: usize = 4 * 1024 * 1024;

#[derive(Debug, Parser)]
#[command(name = "birefnet")]
#[command(
    about = "BiRefNet: Bilateral Reference Network for high-resolution dichotomous image segmentation"
)]
struct Cli {
    /// Device to run on (`default` selects the highest-priority compiled backend or `BURN_DEVICE`).
    #[arg(long, global = true, default_value = "default")]
    device: String,

    /// Float precision for tensors created on the device (`f32`, `f16`, or `bf16`).
    #[arg(long, global = true)]
    precision: Option<Precision>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Run segmentation inference on an image file or directory.
    Infer {
        /// Input image path or directory.
        #[arg(short, long, required_unless_present = "list_models")]
        input: Option<PathBuf>,

        /// Output directory for RGBA segmentation results.
        #[arg(short, long, required_unless_present = "list_models")]
        output: Option<PathBuf>,

        /// Pretrained model preset name (e.g. `"General"`) or local checkpoint path.
        #[arg(short, long, required_unless_present = "list_models")]
        model: Option<String>,

        /// Print all registered pretrained model presets and exit.
        #[arg(long)]
        list_models: bool,

        /// Skip multi-pass blur-fusion foreground refinement.
        #[arg(long)]
        no_refine: bool,
    },

    /// Train a `BiRefNet` model from a JSON training configuration.
    #[cfg(feature = "train")]
    Train {
        /// Path to the JSON training configuration file.
        #[arg(short, long)]
        config: PathBuf,

        /// Directory for checkpoints, metric logs, and the final model.
        #[arg(long, default_value = "./artifacts")]
        artifact_dir: PathBuf,

        /// Resume after this epoch from `<artifact_dir>/checkpoint/*-<epoch>.bpk`.
        #[arg(short, long)]
        resume: Option<usize>,
    },

    /// Show available backends and the active device configuration.
    Info,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    if let Commands::Infer {
        list_models: true, ..
    } = cli.command
    {
        println!("Available pretrained models:");
        for model_name in list_available_models() {
            println!("  - {model_name}");
        }
        return Ok(());
    }

    let mut device = select_device(&cli.device)?;
    if let Some(precision) = cli.precision {
        configure_precision(&mut device, precision)?;
    }
    tracing::info!(?device, "device selected");

    match cli.command {
        Commands::Infer {
            input,
            output,
            model,
            no_refine,
            ..
        } => {
            let args = inference::InferenceCliArgs::new(
                input.ok_or_else(|| anyhow::anyhow!("missing `--input` argument"))?,
                output.ok_or_else(|| anyhow::anyhow!("missing `--output` argument"))?,
                model.ok_or_else(|| anyhow::anyhow!("missing `--model` argument"))?,
                !no_refine,
            );
            stacker::grow(WORKER_STACK_BYTES, || {
                inference::run_inference(&args, &device)
            })
        }

        #[cfg(feature = "train")]
        Commands::Train {
            config,
            artifact_dir,
            resume,
        } => {
            let args = training::TrainingCliArgs::new(config, artifact_dir, resume);
            stacker::grow(WORKER_STACK_BYTES, || {
                training::run_training(&args, &device)
            })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_allows_list_models_without_io_flags() {
        let cli = Cli::try_parse_from(["birefnet", "infer", "--list-models"]).unwrap();
        assert!(matches!(
            cli.command,
            Commands::Infer {
                input: None,
                output: None,
                model: None,
                list_models: true,
                no_refine: false,
            }
        ));
    }

    #[test]
    fn cli_rejects_infer_without_required_flags() {
        let _ = Cli::try_parse_from(["birefnet", "infer"]).unwrap_err();
    }

    #[test]
    fn cli_parses_precision_and_infer_flags() {
        let cli = Cli::try_parse_from([
            "birefnet",
            "--device",
            "flex",
            "--precision",
            "bf16",
            "infer",
            "-i",
            "photo.png",
            "-o",
            "out_dir",
            "-m",
            "General",
            "--no-refine",
        ])
        .unwrap();
        assert_eq!(cli.device, "flex");
        assert_eq!(cli.precision, Some(Precision::Bf16));
        assert!(matches!(
            cli.command,
            Commands::Infer {
                input: Some(_),
                output: Some(_),
                model: Some(_),
                list_models: false,
                no_refine: true,
            }
        ));

        let info_cli = Cli::try_parse_from(["birefnet", "info"]).unwrap();
        assert!(matches!(info_cli.command, Commands::Info));
    }
}
