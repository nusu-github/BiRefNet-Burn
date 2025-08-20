//! Training data structures for BiRefNet.
//!
//! This module defines the batch and output structures used during training and validation.
//! By placing these structures in the model crate, we avoid circular dependencies while
//! maintaining clear separation of concerns.

#[cfg(feature = "train")]
use burn::train::metric::{Adaptor, ItemLazy, LossInput};
use burn::{prelude::*, tensor::backend::Backend};

/// Represents a batch of preprocessed data items from the BiRefNet dataset.
///
/// This struct contains batched image and mask tensors suitable for training
/// and validation with the Burn framework.
#[derive(Debug, Clone)]
pub struct BiRefNetBatch<B: Backend> {
    /// Batched input image tensor with shape [B, C, H, W] where B=batch_size, C=3 for RGB
    pub images: Tensor<B, 4>,
    /// Batched segmentation mask tensor with shape [B, C, H, W] where B=batch_size, C=1 for binary masks
    pub masks: Tensor<B, 4>,
}

/// Output structure for BiRefNet training and validation steps.
///
/// Following Burn's best practices, this struct provides the essential training outputs
/// and implements proper metric adaptors for integration with the training framework.
#[derive(Debug, Clone)]
pub struct BiRefNetOutput<B: Backend> {
    /// The computed loss value
    pub loss: Tensor<B, 1>,
    /// Model prediction logits (segmentation masks)
    pub output: Tensor<B, 4>,
    /// Ground truth target masks  
    pub targets: Tensor<B, 4>,
}

#[cfg(all(feature = "train", feature = "ndarray"))]
impl<B: Backend> ItemLazy for BiRefNetOutput<B> {
    type ItemSync = BiRefNetOutput<burn::backend::ndarray::NdArray<f32>>;

    fn sync(self) -> Self::ItemSync {
        let [loss, output, targets] = burn::tensor::Transaction::default()
            .register(self.loss)
            .register(self.output)
            .register(self.targets)
            .execute()
            .try_into()
            .expect("Correct amount of tensor data");

        let device = &Default::default();

        BiRefNetOutput {
            loss: Tensor::from_data(loss, device),
            output: Tensor::from_data(output, device),
            targets: Tensor::from_data(targets, device),
        }
    }
}

#[cfg(all(feature = "train", not(feature = "ndarray")))]
impl<B: Backend> ItemLazy for BiRefNetOutput<B> {
    type ItemSync = Self;

    fn sync(self) -> Self::ItemSync {
        self
    }
}

#[cfg(not(feature = "train"))]
impl<B: Backend> ItemLazy for BiRefNetOutput<B> {
    type ItemSync = Self;

    fn sync(self) -> Self::ItemSync {
        self
    }
}

impl<B: Backend> BiRefNetBatch<B> {
    /// Create a new BiRefNet batch.
    pub const fn new(images: Tensor<B, 4>, masks: Tensor<B, 4>) -> Self {
        Self { images, masks }
    }

    /// Get the batch size.
    pub fn batch_size(&self) -> usize {
        self.images.dims()[0]
    }
}

impl<B: Backend> BiRefNetOutput<B> {
    /// Create a new BiRefNet output with proper field order.
    pub const fn new(loss: Tensor<B, 1>, output: Tensor<B, 4>, targets: Tensor<B, 4>) -> Self {
        Self {
            loss,
            output,
            targets,
        }
    }
}

/// Adapter for Loss metric integration
#[cfg(feature = "train")]
impl<B: Backend> Adaptor<LossInput<B>> for BiRefNetOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        backend::ndarray::NdArray,
        tensor::{Distribution, Tensor},
    };

    use super::*;

    type TestBackend = NdArray<f32>;

    #[test]
    fn birefnet_batch_new_creates_correct_structure() {
        let device = Default::default();

        let images = Tensor::<TestBackend, 4>::random(
            [4, 3, 64, 64],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let masks = Tensor::<TestBackend, 4>::random(
            [4, 1, 64, 64],
            Distribution::Normal(0.0, 1.0),
            &device,
        );

        let batch = BiRefNetBatch::new(images, masks);

        assert_eq!(batch.images.shape().dims, [4, 3, 64, 64]);
        assert_eq!(batch.masks.shape().dims, [4, 1, 64, 64]);
        assert_eq!(batch.batch_size(), 4);
    }

    #[test]
    fn birefnet_output_new_creates_correct_structure() {
        let device = Default::default();

        let logits = Tensor::<TestBackend, 4>::random(
            [2, 1, 32, 32],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let target = Tensor::<TestBackend, 4>::random(
            [2, 1, 32, 32],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let loss = Tensor::<TestBackend, 1>::random([1], Distribution::Normal(0.0, 1.0), &device);

        let output = BiRefNetOutput::new(loss, logits, target);

        assert_eq!(output.output.shape().dims, [2, 1, 32, 32]);
        assert_eq!(output.targets.shape().dims, [2, 1, 32, 32]);
        assert_eq!(output.loss.shape().dims, [1]);
    }
}
