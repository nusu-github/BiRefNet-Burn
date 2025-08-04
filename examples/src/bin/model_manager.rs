//! Model Manager Example
//!
//! This example demonstrates the ManagedModel functionality,
//! showing how to list available models and load them.

use anyhow::Result;
use birefnet_examples::common::weights::ManagedModel;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "model_manager")]
#[command(about = "BiRefNet Model Manager")]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// List all available pre-trained models
    List,
    /// Show information about a specific model
    Info {
        /// Model name
        model_name: String,
    },
    /// Download and prepare a model for use
    Download {
        /// Model name
        model_name: String,
    },
}

fn main() -> Result<()> {
    let args = Args::parse();

    match args.command {
        Commands::List => {
            println!("Available BiRefNet models:");
            let models = ManagedModel::list_available_models();
            for model in models {
                println!("  - {model}");
            }
        }
        Commands::Info { model_name } => {
            if let Some(spec) = ManagedModel::get_model_spec(&model_name) {
                println!("Model: {model_name}");
                println!("HuggingFace Model ID: {}", spec.hf_model_id);
                println!(
                    "Default resolution: {}x{}",
                    spec.default_resolution.0, spec.default_resolution.1
                );
                println!(
                    "Supports dynamic resolution: {}",
                    spec.supports_dynamic_resolution
                );
            } else {
                println!(
                    "Model '{model_name}' not found. Use 'list' command to see available models."
                );
            }
        }
        Commands::Download { model_name } => {
            println!("Preparing model: {model_name}");
            match ManagedModel::from_pretrained(&model_name) {
                Ok(model) => {
                    println!("✓ Model loaded successfully!");
                    println!("Configuration: {:?}", model.get_config());
                    if let Some(weights_path) = model.get_weights_path() {
                        println!("Weights path: {}", weights_path.display());
                    }
                    if let Some(resolution) = model.get_resolution() {
                        println!("Resolution: {}x{}", resolution.0, resolution.1);
                    }
                    println!(
                        "Supports dynamic resolution: {}",
                        model.supports_dynamic_resolution()
                    );
                }
                Err(e) => {
                    eprintln!("Error loading model: {e}");
                }
            }
        }
    }

    Ok(())
}
