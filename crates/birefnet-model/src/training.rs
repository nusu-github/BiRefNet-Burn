//! Training batch, step output, and Burn `TrainStep` / `InferenceStep` integration.

use burn::{
    prelude::*,
    tensor::ExecutionError,
    train::{
        InferenceStep, TrainOutput, TrainStep,
        metric::{Adaptor, ItemLazy, LossInput},
    },
};

use crate::{BiRefNet, error::ModelError};

/// Batched input images and segmentation target masks for `BiRefNet` training.
#[derive(Debug, Clone)]
pub struct BiRefNetBatch {
    /// Input RGB image batch with shape `[batch_size, 3, height, width]`.
    pub images: Tensor<4>,
    /// Ground-truth mask batch with shape `[batch_size, 1, height, width]`.
    pub masks: Tensor<4>,
}

/// Output emitted by a single `BiRefNet` training or validation step.
#[derive(Debug, Clone)]
pub struct BiRefNetOutput {
    /// Total scalar loss value `[1]`.
    pub loss: Tensor<1>,
    /// Final predicted segmentation logits `[batch_size, 1, height, width]`.
    pub output: Tensor<4>,
    /// Ground-truth segmentation masks `[batch_size, 1, height, width]`.
    pub targets: Tensor<4>,
}

impl ItemLazy for BiRefNetOutput {
    fn sync(self) -> Result<Self, ExecutionError> {
        // Flush queued work and detach every tensor from the autodiff graph.
        self.loss.device().flush()?;

        Ok(Self {
            loss: self.loss.without_autodiff(),
            output: self.output.without_autodiff(),
            targets: self.targets.without_autodiff(),
        })
    }
}

impl BiRefNetBatch {
    /// Creates a new training/validation batch.
    #[must_use]
    pub const fn new(images: Tensor<4>, masks: Tensor<4>) -> Self {
        Self { images, masks }
    }

    /// Returns the number of samples in this batch.
    #[must_use]
    pub fn batch_size(&self) -> usize {
        self.images.dims()[0]
    }
}

impl BiRefNetOutput {
    /// Creates a new training/validation step output.
    #[must_use]
    pub const fn new(loss: Tensor<1>, output: Tensor<4>, targets: Tensor<4>) -> Self {
        Self {
            loss,
            output,
            targets,
        }
    }
}

impl Adaptor<LossInput> for BiRefNetOutput {
    fn adapt(&self) -> LossInput {
        LossInput::new(self.loss.clone())
    }
}

impl BiRefNet {
    /// Runs a supervised training/validation step returning predictions, targets, and loss.
    ///
    /// Without a refiner, the loss supervises every multi-scale decoder output; with one, it
    /// supervises the refined prediction.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if `loss_config` is unset on [`BiRefNetConfig`](crate::BiRefNetConfig)
    /// or the forward pass or loss computation fails.
    pub fn forward_step(&self, batch: BiRefNetBatch) -> Result<BiRefNetOutput, ModelError> {
        let loss_fn = self.loss.as_ref().ok_or_else(|| {
            ModelError::invalid_config("loss_config is not set on BiRefNetConfig")
        })?;

        let (prediction, loss) = if self.has_refiner() {
            let prediction = self.forward(batch.images)?;
            let loss = loss_fn.forward(core::slice::from_ref(&prediction), &batch.masks)?;
            (prediction, loss)
        } else {
            let mut scaled_preds = self.forward_backbone_decoder(&batch.images, true)?;
            let loss = loss_fn.forward(&scaled_preds, &batch.masks)?;
            let prediction = scaled_preds
                .pop()
                .ok_or_else(|| ModelError::invalid_input("decoder emitted no predictions"))?;
            (prediction, loss)
        };

        Ok(BiRefNetOutput::new(loss, prediction, batch.masks))
    }
}

impl TrainStep for BiRefNet {
    type Input = BiRefNetBatch;
    type Output = BiRefNetOutput;

    fn step(&self, item: BiRefNetBatch) -> TrainOutput<BiRefNetOutput> {
        let output = self
            .forward_step(item)
            .unwrap_or_else(|err| panic!("BiRefNet training step failed: {err}"));
        TrainOutput::new(self, output.loss.backward(), output)
    }
}

impl InferenceStep for BiRefNet {
    type Input = BiRefNetBatch;
    type Output = BiRefNetOutput;

    fn step(&self, item: BiRefNetBatch) -> BiRefNetOutput {
        self.forward_step(item)
            .unwrap_or_else(|err| panic!("BiRefNet validation step failed: {err}"))
    }
}
