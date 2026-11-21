//! Training batch and step output structures for `BiRefNet`.

use burn::{
    prelude::*,
    tensor::ExecutionError,
    train::metric::{Adaptor, ItemLazy, LossInput},
};

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
