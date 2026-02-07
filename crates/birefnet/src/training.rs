use std::{fs, path::Path, sync::Arc};

use anyhow::Result;
use birefnet_model::{BiRefNetConfig, InterpolationStrategy, ModelConfig, training::BiRefNetBatch};
use birefnet_train::{BiRefNetDataset, dataset::BiRefNetBatcher};
use burn::{
    backend::Autodiff,
    config::Config,
    data::dataloader::{DataLoader, DataLoaderBuilder},
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{LearnerBuilder, metric::LossMetric},
};

use crate::backend::burn_backend_types::{InferenceBackend, InferenceDevice, NAME};

/// CLI arguments for the training subcommand.
#[derive(Debug)]
pub struct TrainingCliArgs {
    /// Path to the training configuration file.
    pub config_path: std::path::PathBuf,
    /// Optional checkpoint file to resume training from.
    pub resume_checkpoint: Option<std::path::PathBuf>,
}

impl TrainingCliArgs {
    /// Creates a new set of training CLI arguments.
    pub fn new(
        config_path: impl Into<std::path::PathBuf>,
        resume_checkpoint: Option<std::path::PathBuf>,
    ) -> Self {
        Self {
            config_path: config_path.into(),
            resume_checkpoint,
        }
    }
}

/// Comprehensive training configuration for BiRefNet.
///
/// Corresponds to the PyTorch implementation's training configuration,
/// covering model, optimizer, dataset, and checkpointing settings.
/// Loaded from a JSON file via [`TrainingConfig::load`].
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

impl TrainingConfig {
    /// Loads a training configuration from a JSON file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let config_str = fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&config_str)?;
        Ok(config)
    }

    /// Saves this configuration to a JSON file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let config_str = serde_json::to_string_pretty(self)?;
        fs::write(path, config_str)?;
        Ok(())
    }
}

/// Simple training batch wrapper for Burn's learner.
#[derive(Debug)]
pub struct TrainingBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub targets: Tensor<B, 4>,
}

/// Runs the training loop on a specific device.
///
/// # Errors
///
/// Returns an error if model initialization, data loading, or
/// checkpoint saving fails.
pub fn run_training_on_device<B: AutodiffBackend>(
    device: B::Device,
    config: TrainingConfig,
    resume_checkpoint: Option<std::path::PathBuf>,
) -> Result<()>
where
    B::InnerBackend: Backend,
{
    tracing::info!(?device, "initializing BiRefNet training");

    B::seed(config.seed);

    let model_config = BiRefNetConfig::new(config.model.clone())
        .with_loss_config(Some(birefnet_loss::BiRefNetLossConfig::new()));
    let model = model_config.init::<B>(&device)?;

    let optimizer = AdamConfig::new().init();
    tracing::info!(optimizer = %config.optimizer.optimizer_type, "optimizer created");

    let learning_rate = config.learning_rate;

    let train_loader = create_train_dataloader::<B>(&config)?;
    let valid_loader = create_valid_dataloader::<B>(&config)?;

    let learner_builder = LearnerBuilder::new("./artifacts")
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device])
        .num_epochs(config.num_epochs)
        .summary();

    if let Some(_checkpoint_path) = resume_checkpoint {
        // TODO: Implement checkpoint loading
        tracing::warn!("checkpoint resume not yet implemented");
    }

    let learner = learner_builder.build(model, optimizer, learning_rate);

    tracing::info!(epochs = config.num_epochs, "starting training");
    let trained_model = learner.fit(train_loader, valid_loader);

    trained_model
        .save_file("./artifacts/final_model", &CompactRecorder::new())
        .map_err(|e| anyhow::anyhow!("failed to save final model: {e}"))?;

    tracing::info!("training completed successfully");
    Ok(())
}

/// Creates the training dataloader.
fn create_train_dataloader<B: AutodiffBackend>(
    config: &TrainingConfig,
) -> Result<Arc<dyn DataLoader<B, BiRefNetBatch<B>>>>
where
    B::InnerBackend: Backend,
{
    let model_config = ModelConfig::new(InterpolationStrategy::Bilinear);
    let dataset = BiRefNetDataset::new(&model_config, "train")?;
    let batcher = BiRefNetBatcher::<B>::new();

    let dataloader = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(dataset);

    Ok(dataloader)
}

/// Creates the validation dataloader.
fn create_valid_dataloader<B: AutodiffBackend>(
    config: &TrainingConfig,
) -> Result<Arc<dyn DataLoader<B::InnerBackend, BiRefNetBatch<B::InnerBackend>>>>
where
    B::InnerBackend: Backend,
{
    let model_config = ModelConfig::new(InterpolationStrategy::Bilinear);
    let dataset = BiRefNetDataset::new(&model_config, "val")?;
    let batcher = BiRefNetBatcher::<B::InnerBackend>::new();

    let dataloader = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(dataset);

    Ok(dataloader)
}

/// Runs BiRefNet training from a CLI configuration.
///
/// Loads the JSON configuration file, validates paths, selects the
/// backend, and launches the training loop.
///
/// # Errors
///
/// Returns an error if the configuration or checkpoint file is missing,
/// the configuration cannot be parsed, or training fails.
pub fn run_training(args: TrainingCliArgs) -> Result<()> {
    tracing::info!(config = %args.config_path.display(), "BiRefNet training system initialization");

    if let Some(checkpoint) = &args.resume_checkpoint {
        tracing::info!(checkpoint = %checkpoint.display(), "resuming from checkpoint");
    }

    if !args.config_path.exists() {
        anyhow::bail!(
            "Configuration file not found: {}",
            args.config_path.display()
        );
    }

    if let Some(checkpoint) = &args.resume_checkpoint {
        if !Path::new(checkpoint).exists() {
            anyhow::bail!("Checkpoint file not found: {}", checkpoint.display());
        }
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

    tracing::info!(backend = NAME, "starting training on backend");
    run_training_on_device::<Autodiff<InferenceBackend>>(
        InferenceDevice::default(),
        training_config,
        args.resume_checkpoint,
    )
}
