use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::Result;
use birefnet_model::{BiRefNetConfig, ModelConfig};
use birefnet_train::{BiRefNetDataset, dataset::BiRefNetBatcher};
use burn::{
    config::Config,
    data::dataloader::DataLoaderBuilder,
    optim::{AdamConfig, AdamWConfig, ModuleOptimizer, SgdConfig},
    prelude::*,
    train::{Learner, SupervisedTraining, metric::LossMetric},
};

/// CLI arguments for the training subcommand.
#[derive(Debug)]
pub struct TrainingCliArgs {
    /// Path to the training configuration file.
    pub config_path: PathBuf,
    /// Directory for checkpoints, metric logs and the final model.
    pub artifact_dir: PathBuf,
    /// Resume after this epoch from `<artifact_dir>/checkpoint/*-<epoch>.bpk`.
    pub resume_epoch: Option<usize>,
}

impl TrainingCliArgs {
    /// Creates a new set of training CLI arguments.
    pub fn new(
        config_path: impl Into<PathBuf>,
        artifact_dir: impl Into<PathBuf>,
        resume_epoch: Option<usize>,
    ) -> Self {
        Self {
            config_path: config_path.into(),
            artifact_dir: artifact_dir.into(),
            resume_epoch,
        }
    }
}

/// Comprehensive training configuration for BiRefNet.
///
/// Corresponds to the PyTorch implementation's training configuration,
/// covering model, optimizer, dataset, and checkpointing settings.
/// Loaded from a JSON file with [`Config::load`].
#[derive(Config, Debug)]
pub struct TrainingConfig {
    /// Model configuration.
    pub model: ModelConfig,

    /// Optimizer settings.
    pub optimizer: OptimizerConfig,

    #[config(default = 1e-4)]
    pub learning_rate: f64,

    #[config(default = 1e-2)]
    pub weight_decay: f64,

    /// Number of training epochs.
    #[config(default = 120)]
    pub num_epochs: usize,

    #[config(default = 1)]
    pub batch_size: usize,

    #[config(default = 4)]
    pub num_workers: usize,

    /// Dataset configuration.
    pub dataset: DatasetConfig,

    /// Checkpoint save frequency (in epochs).
    #[config(default = 5)]
    pub save_step: usize,

    /// Keep only the last N checkpoints.
    #[config(default = 20)]
    pub save_last: usize,

    /// Random seed for reproducibility.
    #[config(default = 42)]
    pub seed: u64,

    /// Number of epochs for fine-tuning the final layers.
    #[config(default = 0)]
    pub finetune_last_epochs: usize,
}

/// Dataset paths and preprocessing options.
#[derive(Config, Debug)]
pub struct DatasetConfig {
    /// Root directory containing dataset files.
    pub data_root_dir: String,

    /// List of training datasets to use.
    pub training_set: Vec<String>,

    /// Enable dynamic size processing.
    #[config(default = false)]
    pub dynamic_size: bool,

    /// Load all data into memory.
    #[config(default = false)]
    pub load_all: bool,

    /// Target image size for training.
    #[config(default = 1024)]
    pub size: u32,
}

/// Optimizer selection and learning-rate schedule.
#[derive(Config, Debug)]
pub struct OptimizerConfig {
    /// Type of optimizer (e.g. `"AdamW"`, `"Adam"`, `"SGD"`).
    pub optimizer_type: String,

    /// Learning rate decay epochs (negative values count from end).
    pub lr_decay_epochs: Vec<i32>,

    /// Learning rate decay rate.
    #[config(default = 0.1)]
    pub lr_decay_rate: f64,
}

impl OptimizerConfig {
    /// Builds the optimizer named by `optimizer_type`.
    ///
    /// # Errors
    ///
    /// Returns an error for an unsupported optimizer type.
    pub fn init(&self, weight_decay: f64) -> Result<ModuleOptimizer> {
        let optimizer = match self.optimizer_type.to_ascii_lowercase().as_str() {
            "adamw" => AdamWConfig::new()
                .with_weight_decay(weight_decay as f32)
                .init(),
            "adam" => AdamConfig::new().init(),
            "sgd" => SgdConfig::new().init(),
            other => {
                anyhow::bail!("unsupported optimizer type: {other} (expected AdamW, Adam or SGD)")
            }
        };
        Ok(optimizer)
    }
}

/// Runs the training loop on a specific device.
///
/// The model is built on the autodiff version of `device`; validation runs on the plain
/// device. Checkpoints (`model`, `optim` and `scheduler` records in Burnpack format) go to
/// `<artifact_dir>/checkpoint/` and the final weights to `<artifact_dir>/model.bpk`.
///
/// # Errors
///
/// Returns an error if model initialization, data loading, training, or
/// saving the final model fails.
pub fn run_training_on_device(
    device: &Device,
    config: &TrainingConfig,
    artifact_dir: &Path,
    resume_epoch: Option<usize>,
) -> Result<()> {
    tracing::info!(?device, "initializing BiRefNet training");

    fs::create_dir_all(artifact_dir)?;
    config.save(artifact_dir.join("config.json"))?;

    device.seed(config.seed);
    let autodiff_device = device.clone().autodiff();

    // Build the model on the autodiff device: moving a model there later does not make its
    // parameters trainable.
    let model_config = BiRefNetConfig::new(config.model.clone())
        .with_loss_config(Some(birefnet_loss::BiRefNetLossConfig::new()));
    let model = model_config.init(&autodiff_device)?;

    let optimizer = config.optimizer.init(config.weight_decay)?;
    tracing::info!(optimizer = %config.optimizer.optimizer_type, "optimizer created");

    let train_loader = DataLoaderBuilder::new(BiRefNetBatcher::new())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .set_device(autodiff_device)
        .build(BiRefNetDataset::new(&config.model, "train")?);
    let valid_loader = DataLoaderBuilder::new(BiRefNetBatcher::new())
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .set_device(device.clone())
        .build(BiRefNetDataset::new(&config.model, "val")?);

    let mut training = SupervisedTraining::new(artifact_dir, train_loader, valid_loader)
        .metrics((LossMetric::new(),))
        .with_default_checkpointers()
        .num_epochs(config.num_epochs)
        .summary();
    if let Some(epoch) = resume_epoch {
        tracing::info!(epoch, "resuming from checkpoint");
        training = training.checkpoint(epoch);
    }

    tracing::info!(epochs = config.num_epochs, "starting training");
    let result = training.launch(Learner::new(model, optimizer, config.learning_rate));

    if let Some(error) = result.error {
        anyhow::bail!("training failed: {error}");
    }
    if let Some(interruption) = result.interrupted {
        tracing::warn!(?interruption, "training was interrupted");
    }

    // `result.model` is already a validation snapshot (`Module::valid`).
    result
        .model
        .save_file(artifact_dir.join("model"))
        .map_err(|e| anyhow::anyhow!("failed to save final model: {e}"))?;

    tracing::info!(path = %artifact_dir.join("model.bpk").display(), "training completed successfully");
    Ok(())
}

/// Runs BiRefNet training from a CLI configuration.
///
/// Loads the JSON configuration file, validates paths, and launches the
/// training loop on `device`.
///
/// # Errors
///
/// Returns an error if the configuration file is missing, the
/// configuration cannot be parsed, or training fails.
pub fn run_training(args: &TrainingCliArgs, device: &Device) -> Result<()> {
    tracing::info!(config = %args.config_path.display(), "BiRefNet training system initialization");

    if !args.config_path.exists() {
        anyhow::bail!(
            "Configuration file not found: {}",
            args.config_path.display()
        );
    }

    tracing::info!("loading training configuration");
    let training_config = TrainingConfig::load(&args.config_path)?;

    tracing::info!(
        task = ?training_config.model.task.task,
        optimizer = %training_config.optimizer.optimizer_type,
        learning_rate = training_config.learning_rate,
        batch_size = training_config.batch_size,
        epochs = training_config.num_epochs,
        dataset = ?training_config.dataset.training_set,
        "configuration loaded",
    );

    run_training_on_device(
        device,
        &training_config,
        &args.artifact_dir,
        args.resume_epoch,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn optimizer(optimizer_type: &str) -> OptimizerConfig {
        OptimizerConfig::new(optimizer_type.to_owned(), vec![])
    }

    #[test]
    fn optimizer_type_is_case_insensitive() {
        for name in ["AdamW", "adam", "SGD"] {
            assert!(optimizer(name).init(1e-2).is_ok(), "{name}");
        }
    }

    #[test]
    fn unsupported_optimizer_type_is_an_error() {
        let Err(err) = optimizer("Lion").init(1e-2) else {
            panic!("`Lion` should be rejected");
        };
        let err = err.to_string();

        assert!(err.contains("unsupported optimizer type: lion"), "{err}");
    }
}
