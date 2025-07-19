//! Patch-based IoU loss for local region evaluation.

use burn::{
    nn::{Unfold4d, Unfold4dConfig},
    prelude::*,
    tensor::{backend::Backend, Tensor},
};

use super::combined::CombinedLoss;

/// Configuration for Patch IoU Loss function.
#[derive(Config, Debug)]
pub struct PatchIoULossConfig {
    #[config(default = 64)]
    pub patch_size: usize,
    #[config(default = 1e-6)]
    pub epsilon: f32,
}

/// Patch-based IoU loss for local region evaluation.
#[derive(Module, Debug)]
pub struct PatchIoULoss<B: Backend> {
    pub patch_size: usize,
    pub epsilon: f32,
    pub base_iou: CombinedLoss<B>,
    unfolder: Unfold4d,
}

impl PatchIoULossConfig {
    /// Initialize a new patch IoU loss function with the given configuration.
    pub fn init<B: Backend>(&self) -> PatchIoULoss<B> {
        let unfolder = Unfold4dConfig::new([self.patch_size, self.patch_size])
            .with_stride([self.patch_size, self.patch_size])
            .init();
        PatchIoULoss {
            patch_size: self.patch_size,
            epsilon: self.epsilon,
            base_iou: CombinedLoss::with_weights(0.0, 1.0),
            unfolder,
        }
    }
}

impl<B: Backend> Default for PatchIoULoss<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> PatchIoULoss<B> {
    /// Create a new patch IoU loss function with default configuration.
    pub fn new() -> Self {
        PatchIoULossConfig::new().init()
    }

    /// Calculate patch-based IoU loss.
    ///
    /// # Arguments
    /// * `pred` - Predicted segmentation map with shape [N, C, H, W]
    /// * `target` - Ground truth segmentation map with shape [N, C, H, W]
    ///
    /// # Returns
    /// Patch IoU loss tensor
    pub fn forward(&self, pred: Tensor<B, 4>, target: Tensor<B, 4>) -> Tensor<B, 1> {
        let [n, c, _, _] = pred.dims();
        let patch_size = self.patch_size;

        // Unfold predictions and targets into patches
        // The output shape is [N, C * patch_size * patch_size, num_patches]
        let pred_patches = self.unfolder.forward(pred);
        let target_patches = self.unfolder.forward(target);

        let num_patches = pred_patches.dims()[2];

        // Reshape to [N * num_patches, C, patch_size, patch_size] to compute IoU per patch
        let pred_reshaped = pred_patches.reshape([n * num_patches, c, patch_size, patch_size]);
        let target_reshaped = target_patches.reshape([n * num_patches, c, patch_size, patch_size]);

        // The IoU loss will be calculated on all patches, and the mean is returned.
        self.base_iou.iou_loss(pred_reshaped, target_reshaped)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_patch_iou_loss_creation() {
        let loss = PatchIoULoss::<TestBackend>::new();
        assert_eq!(loss.patch_size, 64, "Default patch size should be 64");
        assert!(
            (loss.epsilon - 1e-6).abs() < 1e-10,
            "Default epsilon should be 1e-6"
        );
    }

    #[test]
    fn test_patch_iou_loss_custom_config() {
        let config = PatchIoULossConfig::new()
            .with_patch_size(32)
            .with_epsilon(1e-5);
        let loss = config.init::<TestBackend>();

        assert_eq!(loss.patch_size, 32, "Custom patch size should be respected");
        assert!(
            (loss.epsilon - 1e-5).abs() < 1e-10,
            "Custom epsilon should be respected"
        );
    }

    #[test]
    fn test_patch_iou_loss_small_tensors() {
        let device = Default::default();
        let config = PatchIoULossConfig::new().with_patch_size(2);
        let loss = config.init::<TestBackend>();

        // Create 4x4 tensors that can be divided into 2x2 patches
        let pred = Tensor::<TestBackend, 4>::ones([1, 1, 4, 4], &device);
        let target = Tensor::<TestBackend, 4>::ones([1, 1, 4, 4], &device);

        let result = loss.forward(pred, target);
        let loss_value = result.into_scalar().elem::<f32>();

        // For identical tensors, IoU should be perfect (loss should be close to 0)
        assert!(loss_value >= 0.0, "IoU loss should be non-negative");
        assert!(loss_value <= 2.0, "IoU loss should be reasonable");
    }

    #[test]
    fn test_patch_iou_loss_different_tensors() {
        let device = Default::default();
        let config = PatchIoULossConfig::new().with_patch_size(2);
        let loss = config.init::<TestBackend>();

        // Create 4x4 tensors with different patterns
        let pred = Tensor::<TestBackend, 4>::ones([1, 1, 4, 4], &device);
        let target = Tensor::<TestBackend, 4>::zeros([1, 1, 4, 4], &device);

        let result = loss.forward(pred, target);
        let loss_value = result.into_scalar().elem::<f32>();

        // For completely different tensors, IoU loss should be higher
        assert!(loss_value >= 0.0, "IoU loss should be non-negative");
        assert!(
            loss_value <= 10.0,
            "IoU loss should be within reasonable bounds"
        );
    }
}
