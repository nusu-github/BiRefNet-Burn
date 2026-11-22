//! Command-line entrypoint for BiRefNet inference, training, and backend diagnostics.

mod inference;
#[cfg(feature = "train")]
mod training;

use std::path::PathBuf;

use anyhow::{Context, Result};
use birefnet::{
    device::{AVAILABLE_DEVICES, Precision, configure_precision, select_device},
    weights::list_available_models,
};
use burn::{
    backend::{Backend, Dispatch},
    tensor::{Device, FloatDType},
};
use clap::{Parser, Subcommand};
use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// Stack size for the inference / training worker.
///
/// Building, loading, and running deep Burn module trees recurses far beyond the 1 MiB
/// default main-thread stack on Windows; 64 MiB leaves ample headroom for Swin-L.
const WORKER_STACK_BYTES: usize = 64 * 1024 * 1024;

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

        /// Pretrained model preset name (e.g. `General`) or local checkpoint path.
        #[arg(short, long, required_unless_present = "list_models")]
        model: Option<String>,

        /// Architecture JSON (`ModelConfig`) for a local checkpoint; defaults to a
        /// `model_config.json` next to the checkpoint.
        #[arg(long)]
        config: Option<PathBuf>,

        /// Preset whose architecture and resolution a local checkpoint uses.
        #[arg(long, conflicts_with = "config")]
        preset: Option<String>,

        /// Print all registered pretrained model presets and exit.
        #[arg(long)]
        list_models: bool,

        /// Skip blur-fusion foreground refinement.
        #[arg(long)]
        no_refine: bool,
    },

    /// Train a BiRefNet model from a JSON training configuration.
    #[cfg(feature = "train")]
    Train {
        /// Path to the JSON training configuration file.
        #[arg(short, long)]
        config: PathBuf,

        /// Directory for checkpoints, metric logs, and the final model.
        #[arg(long, default_value = "./artifacts")]
        artifact_dir: PathBuf,

        /// Resume after this epoch from `<artifact_dir>/checkpoint`.
        #[arg(short, long)]
        resume: Option<usize>,
    },

    /// Show compiled backends and the selected device.
    Info,
}

fn print_info(device: &Device) {
    println!("Available devices: {}", AVAILABLE_DEVICES.join(", "));
    println!("Selected device:   {device:?}");
    println!(
        "Backend:           {}",
        <Dispatch as Backend>::name(device.as_dispatch())
    );
    let settings = device.settings();
    println!("Default float:     {:?}", settings.float_dtype);
    let supported: Vec<_> = [FloatDType::F32, FloatDType::F16, FloatDType::BF16]
        .into_iter()
        .filter(|&dtype| device.supports_dtype(dtype))
        .map(|dtype| format!("{dtype:?}"))
        .collect();
    println!("Supported floats:  {}", supported.join(", "));
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

    let mut device = select_device(&cli.device)
        .with_context(|| format!("cannot use device `{}`", cli.device))?;
    if let Some(precision) = cli.precision {
        configure_precision(&mut device, precision)?;
    }
    tracing::info!(name: "birefnet.cli.device", "using device {device:?}");

    match cli.command {
        Commands::Infer {
            input,
            output,
            model,
            config,
            preset,
            no_refine,
            ..
        } => {
            let args = inference::InferenceCliArgs {
                input_path: input.context("missing `--input`")?,
                output_path: output.context("missing `--output`")?,
                source: inference::ModelSource {
                    model: model.context("missing `--model`")?,
                    config,
                    preset,
                },
                refine_foreground: !no_refine,
            };
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
            let args = training::TrainingCliArgs {
                config_path: config,
                artifact_dir,
                resume_epoch: resume,
            };
            stacker::grow(WORKER_STACK_BYTES, || {
                training::run_training(&args, &device)
            })
        }

        Commands::Info => {
            print_info(&device);
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use std::assert_matches;

    use super::*;

    #[test]
    fn cli_allows_list_models_without_io_flags() {
        let cli = Cli::try_parse_from(["birefnet", "infer", "--list-models"]).unwrap();
        assert_matches!(
            cli.command,
            Commands::Infer {
                input: None,
                output: None,
                model: None,
                list_models: true,
                no_refine: false,
                ..
            }
        );
    }

    #[test]
    fn cli_rejects_infer_without_required_flags_or_with_conflicts() {
        Cli::try_parse_from(["birefnet", "infer"]).unwrap_err();
        Cli::try_parse_from([
            "birefnet", "infer", "-i", "a", "-o", "b", "-m", "c", "--config", "x.json", "--preset",
            "General",
        ])
        .unwrap_err();
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
            "weights.bpk",
            "--preset",
            "General-Lite",
            "--no-refine",
        ])
        .unwrap();
        assert_eq!(cli.device, "flex");
        assert_eq!(cli.precision, Some(Precision::Bf16));
        assert_matches!(
            cli.command,
            Commands::Infer {
                input: Some(_),
                output: Some(_),
                model: Some(_),
                preset: Some(_),
                config: None,
                list_models: false,
                no_refine: true,
            }
        );
        assert_matches!(
            Cli::try_parse_from(["birefnet", "info"]).unwrap().command,
            Commands::Info
        );
    }
}
