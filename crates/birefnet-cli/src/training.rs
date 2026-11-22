//! `birefnet train`: supervised training configured by a JSON file.

use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use birefnet::{
    loss::BiRefNetLossConfig,
    metric::{Measure, SegmentationMetric},
    model::{BiRefNetConfig, ModelConfig},
    train::{
        AugmentationConfig, BiRefNetBatcher, BiRefNetDataset, DatasetSplit, DynamicSize,
        ImageAugmentor, Resolution,
    },
};
use burn::{
    config::Config,
    data::{dataloader::DataLoaderBuilder, dataset::transform::ComposedDataset},
    lr_scheduler::{LrScheduler, LrSchedulerRecord},
    optim::{AdamConfig, AdamWConfig, ModuleOptimizer, SgdConfig, decay::WeightDecayConfig},
    prelude::*,
    serde::{Deserialize, Serialize},
    train::{Learner, SupervisedTraining, checkpoint::KeepLastNCheckpoints, metric::LossMetric},
};

use crate::inference::MODEL_CONFIG_FILE;

pub(crate) struct TrainingCliArgs {
    pub(crate) config_path: PathBuf,
    pub(crate) artifact_dir: PathBuf,
    pub(crate) resume_epoch: Option<usize>,
}

/// Optimizer algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(crate = "burn::serde", rename_all = "lowercase")]
pub(crate) enum OptimizerKind {
    /// Adam with decoupled weight decay (upstream default).
    AdamW,
    /// Adam with L2 weight decay.
    Adam,
    /// Stochastic gradient descent with L2 weight decay.
    Sgd,
}

/// Optimizer and learning-rate schedule.
#[derive(Config, Debug)]
pub(crate) struct OptimizerConfig {
    /// Optimizer algorithm.
    #[config(default = "OptimizerKind::AdamW")]
    pub(crate) kind: OptimizerKind,
    /// Base learning rate.
    #[config(default = 1e-4)]
    pub(crate) learning_rate: f64,
    /// Weight decay (decoupled for `adamw`, L2 for `adam` and `sgd`).
    #[config(default = 1e-2)]
    pub(crate) weight_decay: f64,
    /// Epochs after which the learning rate is multiplied by `lr_decay_rate`; negative values
    /// count back from the last epoch (upstream `lr_decay_epochs`).
    #[config(default = "Vec::new()")]
    pub(crate) lr_decay_epochs: Vec<i64>,
    /// Multiplicative learning-rate decay at each milestone.
    #[config(default = 0.5)]
    pub(crate) lr_decay_rate: f64,
}

/// Dataset location and sampling.
#[derive(Config, Debug)]
pub(crate) struct DatasetConfig {
    /// Root directory containing `<task>/<split>/{im,gt}`.
    pub(crate) data_root_dir: PathBuf,
    /// Training split directories, concatenated (upstream `training_set`, joined by `+`).
    #[config(default = "vec![String::from(\"DIS-TR\")]")]
    pub(crate) training_sets: Vec<String>,
    /// Validation split directory.
    #[config(default = "String::from(\"DIS-VD\")")]
    pub(crate) validation_set: String,
    /// Training and validation `[width, height]`.
    #[config(default = "[1024, 1024]")]
    pub(crate) size: [u32; 2],
    /// Optional per-batch random training size: `[[min_w, max_w], [min_h, max_h]]`.
    #[config(default = "None")]
    pub(crate) dynamic_size: Option<[[u32; 2]; 2]>,
    /// Replace backgrounds with synthetic colors during training.
    #[config(default = false)]
    pub(crate) background_synthesis: bool,
}

/// Top-level `birefnet train` configuration.
#[derive(Config, Debug)]
pub(crate) struct TrainingConfig {
    /// Model architecture.
    pub(crate) model: ModelConfig,
    /// Dataset location and sampling.
    pub(crate) dataset: DatasetConfig,
    /// Optimizer and schedule.
    #[config(default = "OptimizerConfig::new()")]
    pub(crate) optimizer: OptimizerConfig,
    #[config(default = 120)]
    pub(crate) num_epochs: usize,
    #[config(default = 4)]
    pub(crate) batch_size: usize,
    #[config(default = 4)]
    pub(crate) num_workers: usize,
    /// Number of most recent checkpoints kept for resuming.
    #[config(default = 2)]
    pub(crate) keep_checkpoints: usize,
    /// Recompute activations in the backward pass to save memory.
    #[config(default = false)]
    pub(crate) gradient_checkpointing: bool,
    #[config(default = 42)]
    pub(crate) seed: u64,
}

impl OptimizerConfig {
    fn init(&self) -> Result<ModuleOptimizer> {
        if !(self.weight_decay.is_finite() && self.weight_decay >= 0.0) {
            bail!(
                "weight_decay must be finite and non-negative, got {}",
                self.weight_decay
            );
        }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "AdamW stores its weight decay as f32"
        )]
        let decay_f32 = self.weight_decay as f32;
        let l2 = (self.weight_decay > 0.0).then(|| WeightDecayConfig::new(decay_f32));
        Ok(match self.kind {
            OptimizerKind::AdamW => AdamWConfig::new().with_weight_decay(decay_f32).init(),
            OptimizerKind::Adam => AdamConfig::new().with_weight_decay(l2).init(),
            OptimizerKind::Sgd => SgdConfig::new().with_weight_decay(l2).init(),
        })
    }

    /// Converts epoch milestones to optimizer-step milestones.
    fn milestones(&self, num_epochs: usize, steps_per_epoch: usize) -> Result<Vec<usize>> {
        let epochs = i64::try_from(num_epochs)?;
        let mut milestones = self
            .lr_decay_epochs
            .iter()
            .map(|&epoch| {
                let epoch = if epoch > 0 { epoch } else { epochs + epoch + 1 };
                usize::try_from(epoch.clamp(0, epochs))
                    .map(|epoch| epoch * steps_per_epoch)
                    .map_err(anyhow::Error::from)
            })
            .collect::<Result<Vec<_>>>()?;
        milestones.sort_unstable();
        Ok(milestones)
    }
}

/// `torch.optim.lr_scheduler.MultiStepLR`, stepped per optimizer step.
#[derive(Debug, Clone)]
struct MultiStepLr {
    base_lr: f64,
    gamma: f64,
    milestones: Vec<usize>,
    step: usize,
}

impl LrScheduler for MultiStepLr {
    fn step(&mut self) -> f64 {
        let passed = self.milestones.partition_point(|&m| m < self.step);
        self.step += 1;
        let exponent = i32::try_from(passed).unwrap_or(i32::MAX);
        self.base_lr * self.gamma.powi(exponent)
    }

    fn to_record(&self) -> LrSchedulerRecord {
        LrSchedulerRecord::new().with_scalar("step", u64::try_from(self.step).unwrap_or(u64::MAX))
    }

    fn load_record(&mut self, record: LrSchedulerRecord) {
        if let Some(step) = record.scalar::<u64>("step") {
            self.step = usize::try_from(step).unwrap_or(usize::MAX);
        }
    }
}

fn training_resolution(dataset: &DatasetConfig) -> Result<Resolution> {
    Ok(match dataset.dynamic_size {
        Some([[min_w, max_w], [min_h, max_h]]) => {
            Resolution::Dynamic(DynamicSize::new(min_w..=max_w, min_h..=max_h)?)
        }
        None => Resolution::fixed(dataset.size[0], dataset.size[1])?,
    })
}

fn run_training_on_device(
    device: &Device,
    config: &TrainingConfig,
    artifact_dir: &Path,
    resume_epoch: Option<usize>,
) -> Result<()> {
    fs::create_dir_all(artifact_dir)?;
    config.save(artifact_dir.join("config.json"))?;
    config.model.save(artifact_dir.join(MODEL_CONFIG_FILE))?;

    device.seed(config.seed);
    let autodiff_device = device.clone().autodiff();
    let model = BiRefNetConfig::new(config.model.clone())
        .with_loss_config(Some(BiRefNetLossConfig::new()))
        .init(&autodiff_device)?;
    let optimizer = config.optimizer.init()?;

    let task = config.model.task.task;
    let root = &config.dataset.data_root_dir;
    let augmentation =
        AugmentationConfig::new().with_background_synthesis(config.dataset.background_synthesis);
    let train_sets = config
        .dataset
        .training_sets
        .iter()
        .enumerate()
        .map(|(idx, name)| {
            let seed = config
                .seed
                .wrapping_add(u64::try_from(idx).unwrap_or(u64::MAX));
            Ok(
                BiRefNetDataset::new(root, task, &DatasetSplit::Custom(name.clone()))?
                    .with_augmentor(ImageAugmentor::with_seed(augmentation.clone(), seed)),
            )
        })
        .collect::<Result<Vec<_>>>()?;
    if train_sets.is_empty() {
        bail!("dataset.training_sets must name at least one split");
    }
    let train_dataset = ComposedDataset::new(train_sets);
    // Validation sees no augmentation and a fixed size.
    let valid_dataset = BiRefNetDataset::new(
        root,
        task,
        &DatasetSplit::Custom(config.dataset.validation_set.clone()),
    )?;
    let train_len = burn::data::dataset::Dataset::len(&train_dataset);

    let train_loader = DataLoaderBuilder::new(BiRefNetBatcher::with_seed(
        training_resolution(&config.dataset)?,
        config.seed,
    ))
    .batch_size(config.batch_size)
    .shuffle(config.seed)
    .num_workers(config.num_workers)
    .set_device(autodiff_device)
    .build(train_dataset);
    let [width, height] = config.dataset.size;
    let valid_loader =
        DataLoaderBuilder::new(BiRefNetBatcher::new(Resolution::fixed(width, height)?))
            .batch_size(config.batch_size)
            .num_workers(config.num_workers)
            .set_device(device.clone())
            .build(valid_dataset);

    let steps_per_epoch = train_len.div_ceil(config.batch_size.max(1));
    let scheduler = MultiStepLr {
        base_lr: config.optimizer.learning_rate,
        gamma: config.optimizer.lr_decay_rate,
        milestones: config
            .optimizer
            .milestones(config.num_epochs, steps_per_epoch)?,
        step: 0,
    };

    let mut training = SupervisedTraining::new(artifact_dir, train_loader, valid_loader)
        .metrics((LossMetric::new(),))
        .metric_valid_numeric(SegmentationMetric::new(Measure::Mae)?)
        .metric_valid_numeric(SegmentationMetric::new(Measure::AdaptiveFMeasure {
            beta_sq: 0.3,
        })?)
        .with_default_checkpointers()
        .with_checkpointing_strategy(KeepLastNCheckpoints::new(config.keep_checkpoints.max(1)))
        .num_epochs(config.num_epochs)
        .summary();
    if config.gradient_checkpointing {
        training = training.gradient_checkpointing();
    }
    if let Some(epoch) = resume_epoch {
        tracing::info!(name: "birefnet.cli.train.resume", epoch, "resuming after epoch {epoch}");
        training = training.checkpoint(epoch);
    }

    tracing::info!(
        name: "birefnet.cli.train.start",
        epochs = config.num_epochs,
        samples = train_len,
        "training on {train_len} samples for {} epochs",
        config.num_epochs
    );
    let mut result = training.launch(Learner::new(model, optimizer, scheduler));
    result.renderer.manual_close();

    if let Some(error) = result.error {
        bail!("training failed: {error}");
    }
    if let Some(interruption) = result.interrupted {
        tracing::warn!(name: "birefnet.cli.train.interrupted", "training was interrupted: {interruption:?}");
    }

    let model_path = artifact_dir.join("model");
    result
        .model
        .save_file(&model_path)
        .with_context(|| format!("failed to save `{}`", model_path.display()))?;
    tracing::info!(
        name: "birefnet.cli.train.done",
        path = %artifact_dir.join("model.bpk").display(),
        "training finished"
    );
    Ok(())
}

pub(crate) fn run_training(args: &TrainingCliArgs, device: &Device) -> Result<()> {
    let config = TrainingConfig::load(&args.config_path).with_context(|| {
        format!(
            "failed to read training configuration `{}`",
            args.config_path.display()
        )
    })?;
    run_training_on_device(device, &config, &args.artifact_dir, args.resume_epoch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_optimizer_accepts_weight_decay() {
        for kind in [
            OptimizerKind::AdamW,
            OptimizerKind::Adam,
            OptimizerKind::Sgd,
        ] {
            let config = OptimizerConfig::new()
                .with_kind(kind)
                .with_weight_decay(1e-2);
            assert!(config.init().is_ok(), "{kind:?}");
        }
        assert!(
            OptimizerConfig::new()
                .with_weight_decay(-1.0)
                .init()
                .is_err()
        );
    }

    #[test]
    fn optimizer_kinds_deserialize_from_lowercase_names() {
        let config = OptimizerConfig::load_binary(br#"{"kind": "sgd"}"#).unwrap();
        assert_eq!(config.kind, OptimizerKind::Sgd);
        assert!(OptimizerConfig::load_binary(br#"{"kind": "lion"}"#).is_err());
    }

    #[test]
    fn multi_step_lr_decays_after_each_milestone_and_resumes() {
        // Two epochs of three steps; decay after epoch 1 and (counted from the end) epoch 2.
        let config = OptimizerConfig::new()
            .with_lr_decay_epochs(vec![1, -1])
            .with_lr_decay_rate(0.1);
        let milestones = config.milestones(2, 3).unwrap();
        assert_eq!(milestones, vec![3, 6]);

        let mut scheduler = MultiStepLr {
            base_lr: 1.0,
            gamma: 0.1,
            milestones,
            step: 0,
        };
        let rates: Vec<f64> = (0..7).map(|_| scheduler.step()).collect();
        assert_eq!(&rates[..3], &[1.0; 3]);
        assert!(rates[3..6].iter().all(|&lr| (lr - 0.1).abs() < 1e-12));
        assert!((rates[6] - 0.01).abs() < 1e-12);

        let mut resumed = MultiStepLr {
            step: 0,
            ..scheduler.clone()
        };
        resumed.load_record(scheduler.to_record());
        assert!((resumed.step() - scheduler.step()).abs() < 1e-15);
    }

    #[test]
    fn dynamic_size_ranges_map_to_a_dynamic_resolution() {
        let mut dataset = DatasetConfig::new(PathBuf::from("data"));
        assert_eq!(dataset.size, [1024, 1024]);
        assert!(matches!(
            training_resolution(&dataset).unwrap(),
            Resolution::Fixed([1024, 1024])
        ));
        dataset.dynamic_size = Some([[256, 2304], [256, 2304]]);
        assert!(matches!(
            training_resolution(&dataset).unwrap(),
            Resolution::Dynamic(_)
        ));
        dataset.dynamic_size = Some([[10, 20], [256, 512]]);
        assert!(training_resolution(&dataset).is_err());
    }
}
