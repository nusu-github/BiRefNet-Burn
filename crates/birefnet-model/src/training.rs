//! Training data structures for BiRefNet.
//!
//! This module defines the batch and output structures used during training and validation.
//! By placing these structures in the model crate, we avoid circular dependencies while
//! maintaining clear separation of concerns.

use burn::prelude::*;
#[cfg(feature = "train")]
use burn::train::metric::{Adaptor, ItemLazy, LossInput};

/// Represents a batch of preprocessed data items from the BiRefNet dataset.
///
/// This struct contains batched image and mask tensors suitable for training
/// and validation with the Burn framework.
#[derive(Debug, Clone)]
pub struct BiRefNetBatch {
    /// Batched input image tensor with shape [B, C, H, W] where B=batch_size, C=3 for RGB
    pub images: Tensor<4>,
    /// Batched segmentation mask tensor with shape [B, C, H, W] where B=batch_size, C=1 for binary masks
    pub masks: Tensor<4>,
}

/// Output structure for BiRefNet training and validation steps.
///
/// Following Burn's best practices, this struct provides the essential training outputs
/// and implements proper metric adaptors for integration with the training framework.
#[derive(Debug, Clone)]
pub struct BiRefNetOutput {
    /// The computed loss value
    pub loss: Tensor<1>,
    /// Model prediction logits (segmentation masks)
    pub output: Tensor<4>,
    /// Ground truth target masks  
    pub targets: Tensor<4>,
}

#[cfg(feature = "train")]
impl ItemLazy for BiRefNetOutput {
    fn sync(self) -> Result<Self, burn::tensor::ExecutionError> {
        // No readback: metrics compute on the device and read back only final scalars.
        // Flushing dispatches buffered work; dropping autodiff keeps the tape off the metric thread.
        self.loss.device().flush()?;

        Ok(Self {
            loss: self.loss.without_autodiff(),
            output: self.output.without_autodiff(),
            targets: self.targets.without_autodiff(),
        })
    }
}

impl BiRefNetBatch {
    /// Create a new BiRefNet batch.
    pub const fn new(images: Tensor<4>, masks: Tensor<4>) -> Self {
        Self { images, masks }
    }

    /// Get the batch size.
    pub fn batch_size(&self) -> usize {
        self.images.dims()[0]
    }
}

impl BiRefNetOutput {
    /// Create a new BiRefNet output with proper field order.
    pub const fn new(loss: Tensor<1>, output: Tensor<4>, targets: Tensor<4>) -> Self {
        Self {
            loss,
            output,
            targets,
        }
    }
}

/// Adapter for Loss metric integration
#[cfg(feature = "train")]
impl Adaptor<LossInput> for BiRefNetOutput {
    fn adapt(&self) -> LossInput {
        LossInput::new(self.loss.clone())
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{Distribution, Tensor};

    use super::*;

    #[test]
    fn birefnet_batch_new_creates_correct_structure() {
        let device = burn::tensor::Device::cpu();

        let images = Tensor::<4>::random([4, 3, 64, 64], Distribution::Normal(0.0, 1.0), &device);
        let masks = Tensor::<4>::random([4, 1, 64, 64], Distribution::Normal(0.0, 1.0), &device);

        let batch = BiRefNetBatch::new(images, masks);

        assert_eq!(batch.images.shape().dims(), [4, 3, 64, 64]);
        assert_eq!(batch.masks.shape().dims(), [4, 1, 64, 64]);
        assert_eq!(batch.batch_size(), 4);
    }

    #[test]
    fn birefnet_output_new_creates_correct_structure() {
        let device = burn::tensor::Device::cpu();

        let logits = Tensor::<4>::random([2, 1, 32, 32], Distribution::Normal(0.0, 1.0), &device);
        let target = Tensor::<4>::random([2, 1, 32, 32], Distribution::Normal(0.0, 1.0), &device);
        let loss = Tensor::<1>::random([1], Distribution::Normal(0.0, 1.0), &device);

        let output = BiRefNetOutput::new(loss, logits, target);

        assert_eq!(output.output.shape().dims(), [2, 1, 32, 32]);
        assert_eq!(output.targets.shape().dims(), [2, 1, 32, 32]);
        assert_eq!(output.loss.shape().dims(), [1]);
    }
}
