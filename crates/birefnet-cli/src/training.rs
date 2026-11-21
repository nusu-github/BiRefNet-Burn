//! CLI training runner configured via JSON files.

use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::Result;
use birefnet::{
    loss::BiRefNetLossConfig,
    model::{BiRefNetConfig, ModelConfig},
    train::{AugmentationConfig, BiRefNetBatcher, BiRefNetDataset, DatasetSplit},
};
use burn::{
    config::Config,
    data::dataloader::DataLoaderBuilder,
    optim::{AdamConfig, AdamWConfig, ModuleOptimizer, SgdConfig},
    prelude::*,
    train::{Learner, SupervisedTraining, metric::LossMetric},
};

pub(crate) struct TrainingCliArgs {
    pub(crate) config_path: PathBuf,
    pub(crate) artifact_dir: PathBuf,
    pub(crate) resume_epoch: Option<usize>,
}

impl TrainingCliArgs {
    pub(crate) const fn new(
        config_path: PathBuf,
        artifact_dir: PathBuf,
        resume_epoch: Option<usize>,
    ) -> Self {
        Self {
            config_path,
            artifact_dir,
            resume_epoch,
        }
    }
}

#[derive(Config, Debug)]
pub(crate) struct TrainingConfig {
    pub(crate) model: ModelConfig,
    pub(crate) optimizer: OptimizerConfig,
    #[config(default = 1e-4)]
    pub(crate) learning_rate: f64,
    #[config(default = 1e-2)]
    pub(crate) weight_decay: f64,
    #[config(default = 120)]
    pub(crate) num_epochs: usize,
    #[config(default = 1)]
    pub(crate) batch_size: usize,
    #[config(default = 4)]
    pub(crate) num_workers: usize,
    pub(crate) dataset: DatasetConfig,
    #[config(default = 5)]
    pub(crate) save_step: usize,
    #[config(default = 20)]
    pub(crate) save_last: usize,
    #[config(default = 42)]
    pub(crate) seed: u64,
    #[config(default = 0)]
    pub(crate) finetune_last_epochs: usize,
}

#[derive(Config, Debug)]
pub(crate) struct DatasetConfig {
    pub(crate) data_root_dir: String,
    pub(crate) training_set: Vec<String>,
    #[config(default = false)]
    pub(crate) dynamic_size: bool,
    #[config(default = false)]
    pub(crate) load_all: bool,
    #[config(default = 1024)]
    pub(crate) size: u32,
}

#[derive(Config, Debug)]
pub(crate) struct OptimizerConfig {
    pub(crate) optimizer_type: String,
    pub(crate) lr_decay_epochs: Vec<i32>,
    #[config(default = 0.1)]
    pub(crate) lr_decay_rate: f64,
}

impl OptimizerConfig {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "Burn AdamW weight_decay takes f32"
    )]
    pub(crate) fn init(&self, weight_decay: f64) -> Result<ModuleOptimizer> {
        let optimizer = match self.optimizer_type.to_ascii_lowercase().as_str() {
            "adamw" => AdamWConfig::new()
                .with_weight_decay(weight_decay as f32)
                .init(),
            "adam" => AdamConfig::new().init(),
            "sgd" => SgdConfig::new().init(),
            other => {
                anyhow::bail!("unsupported optimizer type `{other}` (expected AdamW, Adam, or SGD)")
            }
        };
        Ok(optimizer)
    }
}

fn build_augmentation_config(dataset: &DatasetConfig) -> AugmentationConfig {
    let target = (dataset.size, dataset.size);
    if dataset.dynamic_size {
        AugmentationConfig::with_dynamic_size(320, dataset.size.max(320), target)
    } else {
        AugmentationConfig {
            target_size: target,
            ..AugmentationConfig::default()
        }
    }
}

pub(crate) fn run_training_on_device(
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

    let model_config =
        BiRefNetConfig::new(config.model.clone()).with_loss_config(Some(BiRefNetLossConfig::new()));
    let model = model_config.init(&autodiff_device)?;

    let optimizer = config.optimizer.init(config.weight_decay)?;
    tracing::info!(optimizer = %config.optimizer.optimizer_type, "optimizer created");

    let aug_config = build_augmentation_config(&config.dataset);
    let train_dataset = BiRefNetDataset::with_augmentation(
        &config.dataset.data_root_dir,
        config.model.task.task,
        &DatasetSplit::Train,
        aug_config.clone(),
    )?;
    let valid_dataset = BiRefNetDataset::with_augmentation(
        &config.dataset.data_root_dir,
        config.model.task.task,
        &DatasetSplit::Val,
        aug_config,
    )?;

    let train_loader = DataLoaderBuilder::new(BiRefNetBatcher::new())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .set_device(autodiff_device)
        .build(train_dataset);
    let valid_loader = DataLoaderBuilder::new(BiRefNetBatcher::new())
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .set_device(device.clone())
        .build(valid_dataset);

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

    result
        .model
        .save_file(artifact_dir.join("model"))
        .map_err(|err| anyhow::anyhow!("failed to save final model: {err}"))?;

    tracing::info!(path = %artifact_dir.join("model.bpk").display(), "training completed successfully");
    Ok(())
}

pub(crate) fn run_training(args: &TrainingCliArgs, device: &Device) -> Result<()> {
    tracing::info!(config = %args.config_path.display(), "loading training configuration");

    if !args.config_path.exists() {
        anyhow::bail!(
            "configuration file not found: {}",
            args.config_path.display()
        );
    }

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
        assert!(
            err.to_string()
                .contains("unsupported optimizer type `lion`")
        );
    }
}
