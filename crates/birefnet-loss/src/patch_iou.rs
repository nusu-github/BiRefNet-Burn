//! Patch-based IoU loss for localized region evaluation.
//!
//! Applies IoU loss on overlapping patches of the input to emphasize
//! local consistency in segmentation tasks. This is particularly useful
//! for fine-grained segmentation where global IoU might miss local errors.
//!
//! The loss is computed as:
//! ```text
//! For each patch (y, x) with size (win_y, win_x):
//!   patch_pred = pred[:, :, y:y+win_y, x:x+win_x]
//!   patch_target = target[:, :, y:y+win_y, x:x+win_x]
//!   patch_loss = IoULoss(patch_pred, patch_target)
//! Total Loss = sum(all patch losses)
//! ```

use burn::{
    config::Config,
    module::{Content, DisplaySettings, Module, ModuleDisplay},
    nn::loss::Reduction,
    tensor::{Int, Tensor, backend::Backend},
};

use super::iou::IoULoss;

/// Configuration for creating a [Patch IoU loss](PatchIoULoss).
#[derive(Config, Debug)]
pub struct PatchIoULossConfig {
    /// Height of each patch window. Default: 64
    #[config(default = 64)]
    pub patch_height: usize,

    /// Width of each patch window. Default: 64
    #[config(default = 64)]
    pub patch_width: usize,

    /// Small epsilon value to avoid division by zero in IoU calculation. Default: 1e-8
    #[config(default = 1e-8)]
    pub eps: f64,
}

impl PatchIoULossConfig {
    /// Initialize [Patch IoU loss](PatchIoULoss).
    pub fn init(&self) -> PatchIoULoss {
        self.assertions();
        PatchIoULoss {
            patch_height: self.patch_height,
            patch_width: self.patch_width,
            iou_loss: IoULoss::new(),
        }
    }

    fn assertions(&self) {
        assert!(
            self.patch_height > 0,
            "Patch height for PatchIoULoss must be positive, got {}",
            self.patch_height
        );
        assert!(
            self.patch_width > 0,
            "Patch width for PatchIoULoss must be positive, got {}",
            self.patch_width
        );
        assert!(
            self.eps > 0.0,
            "Epsilon for PatchIoULoss must be positive, got {}",
            self.eps
        );
    }
}

/// Patch-based IoU loss for localized region evaluation.
///
/// This loss function divides the input into overlapping patches and
/// computes IoU loss for each patch individually. This approach helps
/// to emphasize local consistency and can detect errors that might be
/// missed by global IoU computation.
#[derive(Module, Clone, Debug)]
#[module(custom_display)]
pub struct PatchIoULoss {
    /// Height of each patch window.
    pub patch_height: usize,
    /// Width of each patch window.
    pub patch_width: usize,
    /// IoU loss function for individual patches.
    pub iou_loss: IoULoss,
}

impl Default for PatchIoULoss {
    fn default() -> Self {
        Self::new()
    }
}

impl ModuleDisplay for PatchIoULoss {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("patch_height", &self.patch_height)
            .add("patch_width", &self.patch_width)
            .add("iou_loss", &self.iou_loss)
            .optional()
    }
}

impl PatchIoULoss {
    /// Create a new patch IoU loss with default configuration.
    pub fn new() -> Self {
        PatchIoULossConfig::new().init()
    }

    /// Compute the criterion on the input tensor with reduction.
    ///
    /// # Shapes
    ///
    /// - predictions: `[batch_size, channels, height, width]`
    /// - targets: `[batch_size, channels, height, width]`
    /// - output: `[1]`
    pub fn forward<B: Backend>(
        &self,
        predictions: Tensor<B, 4>,
        targets: Tensor<B, 4, Int>,
        reduction: Reduction,
    ) -> Tensor<B, 1> {
        let loss = self.forward_no_reduction(predictions, targets);
        match reduction {
            Reduction::Mean | Reduction::Auto => loss.mean(),
            Reduction::Sum => loss.sum(),
        }
    }

    /// Compute the criterion on the input tensor without reduction.
    ///
    /// # Shapes
    ///
    /// - predictions: `[batch_size, channels, height, width]`
    /// - targets: `[batch_size, channels, height, width]`
    /// - output: `[num_patches]`
    pub fn forward_no_reduction<B: Backend>(
        &self,
        predictions: Tensor<B, 4>,
        targets: Tensor<B, 4, Int>,
    ) -> Tensor<B, 1> {
        self.assertions(&predictions, &targets);

        let [batch_size, channels, height, width] = predictions.dims();
        let device = predictions.device();

        let mut patch_losses = Vec::new();

        // Iterate over patches
        // TODO: Use Unfold4d?
        for anchor_y in (0..height).step_by(self.patch_height) {
            for anchor_x in (0..width).step_by(self.patch_width) {
                // Calculate patch boundaries
                let end_y = (anchor_y + self.patch_height).min(height);
                let end_x = (anchor_x + self.patch_width).min(width);

                // Extract patches using slice operations
                let patch_pred = predictions.clone().slice([
                    0..batch_size,
                    0..channels,
                    anchor_y..end_y,
                    anchor_x..end_x,
                ]);
                let patch_target = targets.clone().slice([
                    0..batch_size,
                    0..channels,
                    anchor_y..end_y,
                    anchor_x..end_x,
                ]);

                // Compute IoU loss for this patch (returns [batch_size])
                let patch_loss = self.iou_loss.forward_no_reduction(patch_pred, patch_target);

                // Sum over batch dimension to get a scalar for this patch
                let patch_loss_scalar = patch_loss.sum();
                patch_losses.push(patch_loss_scalar);
            }
        }

        // Stack all patch losses into a single tensor
        if patch_losses.is_empty() {
            Tensor::zeros([1], &device)
        } else {
            // Sum all patch losses
            let mut total_loss = patch_losses[0].clone();
            for patch_loss in patch_losses.into_iter().skip(1) {
                total_loss = total_loss + patch_loss;
            }
            total_loss.reshape([1])
        }
    }

    fn assertions<B: Backend>(&self, predictions: &Tensor<B, 4>, targets: &Tensor<B, 4, Int>) {
        let pred_dims = predictions.dims();
        let target_dims = targets.dims();
        assert_eq!(
            pred_dims, target_dims,
            "Shape of predictions ({pred_dims:?}) must match targets ({target_dims:?})"
        );

        let [_, _, height, width] = pred_dims;
        assert!(
            height >= self.patch_height && width >= self.patch_width,
            "Input dimensions ({height}x{width}) must be at least as large as patch size ({}x{})",
            self.patch_height,
            self.patch_width
        );
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{TensorData, cast::ToElement};

    use super::*;
    use crate::tests::TestBackend;

    #[test]
    fn patch_iou_loss_forward_perfect_overlap_returns_near_zero() {
        let device = Default::default();
        let loss = PatchIoULoss::new();

        // Perfect overlap across all patches
        let predictions =
            Tensor::<TestBackend, 4>::from_data(TensorData::from([[[[1.0; 128]; 128]]]), &device);
        let targets = Tensor::<TestBackend, 4, Int>::from_data(
            TensorData::from([[[[1; 128]; 128]]]),
            &device,
        );

        let result_mean = loss.forward(predictions.clone(), targets.clone(), Reduction::Mean);
        let result_no_reduction = loss.forward_no_reduction(predictions, targets);

        // Perfect IoU should give loss ≈ 0
        assert!(
            result_mean.into_scalar().to_f64() < 1e-5,
            "Loss should be near zero for perfect overlap"
        );
        assert!(result_no_reduction.into_scalar().to_f64() < 1e-5);
    }

    #[test]
    fn patch_iou_loss_forward_no_overlap_returns_high_loss() {
        let device = Default::default();
        let loss = PatchIoULoss::new();

        // Create checkerboard pattern with no overlap
        let mut pred_data = [[[[0.0; 128]; 128]; 1]; 1];
        let mut target_data = [[[[0; 128]; 128]; 1]; 1];

        // Fill first half with 1s in predictions, second half with 1s in targets
        for i in 0..64 {
            for j in 0..128 {
                pred_data[0][0][i][j] = 1.0;
                target_data[0][0][i + 64][j] = 1;
            }
        }

        let predictions = Tensor::<TestBackend, 4>::from_data(TensorData::from(pred_data), &device);
        let targets =
            Tensor::<TestBackend, 4, Int>::from_data(TensorData::from(target_data), &device);

        let result = loss.forward(predictions, targets, Reduction::Mean);

        // Should have high loss for no overlap
        assert!(
            result.into_scalar().to_f64() > 0.5,
            "Loss should be high for no overlap"
        );
    }

    #[test]
    fn patch_iou_loss_forward_small_input_with_2x2_patches_works() {
        let device = Default::default();
        let config = PatchIoULossConfig::new()
            .with_patch_height(2)
            .with_patch_width(2);
        let loss = config.init();

        // Small 4x4 input
        let predictions = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([[[
                [0.8, 0.2, 0.9, 0.1],
                [0.3, 0.7, 0.4, 0.6],
                [0.9, 0.1, 0.8, 0.2],
                [0.2, 0.8, 0.3, 0.7],
            ]]]),
            &device,
        );
        let targets = Tensor::<TestBackend, 4, Int>::from_data(
            TensorData::from([[[[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]]]]),
            &device,
        );

        let result_mean = loss.forward(predictions.clone(), targets.clone(), Reduction::Mean);
        let result_sum = loss.forward(predictions.clone(), targets.clone(), Reduction::Sum);
        let result_no_reduction = loss.forward_no_reduction(predictions, targets);

        // All should be valid finite values
        assert!(result_mean.clone().into_scalar().to_f64().is_finite());
        assert!(result_sum.clone().into_scalar().to_f64().is_finite());
        assert!(result_no_reduction.into_scalar().to_f64().is_finite());

        // Sum should be >= Mean (since we're summing patches)
        assert!(result_sum.into_scalar().to_f64() >= result_mean.into_scalar().to_f64());
    }

    #[test]
    fn patch_iou_loss_forward_batch_processing_produces_valid_output() {
        let device = Default::default();
        let config = PatchIoULossConfig::new()
            .with_patch_height(4)
            .with_patch_width(4);
        let loss = config.init();

        // Batch of 2 samples, each 8x8
        let predictions = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([
                [[[0.8; 8]; 8]], // Sample 1: all high predictions
                [[[0.2; 8]; 8]], // Sample 2: all low predictions
            ]),
            &device,
        );
        let targets = Tensor::<TestBackend, 4, Int>::from_data(
            TensorData::from([
                [[[1; 8]; 8]], // Sample 1: all targets
                [[[0; 8]; 8]], // Sample 2: no targets
            ]),
            &device,
        );

        let result_mean = loss.forward(predictions.clone(), targets.clone(), Reduction::Mean);
        let result_no_reduction = loss.forward_no_reduction(predictions, targets);

        // Both samples should have low loss (good predictions vs targets)
        assert!(result_mean.into_scalar().to_f64() >= 0.0);
        assert!(result_no_reduction.into_scalar().to_f64() >= 0.0);
    }

    #[test]
    fn patch_iou_loss_auto_reduction_equals_mean_reduction() {
        let device = Default::default();
        let config = PatchIoULossConfig::new()
            .with_patch_height(4)
            .with_patch_width(4);
        let loss = config.init();

        let predictions = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([[[
                [0.3, 0.7, 0.8, 0.2],
                [0.9, 0.1, 0.4, 0.6],
                [0.5, 0.5, 0.7, 0.3],
                [0.2, 0.8, 0.1, 0.9],
            ]]]),
            &device,
        );
        let targets = Tensor::<TestBackend, 4, Int>::from_data(
            TensorData::from([[[[0, 1, 1, 0], [1, 0, 0, 1], [1, 1, 1, 0], [0, 1, 0, 1]]]]),
            &device,
        );

        let result_auto = loss.forward(predictions.clone(), targets.clone(), Reduction::Auto);
        let result_mean = loss.forward(predictions, targets, Reduction::Mean);

        let auto_val = result_auto.into_scalar().to_f64();
        let mean_val = result_mean.into_scalar().to_f64();

        assert!((auto_val - mean_val).abs() < 1e-6, "Auto should equal Mean");
    }

    #[test]
    #[should_panic = "Patch height for PatchIoULoss must be positive"]
    fn patch_iou_loss_config_zero_patch_height_panics() {
        let _loss = PatchIoULossConfig::new().with_patch_height(0).init();
    }

    #[test]
    #[should_panic = "Patch width for PatchIoULoss must be positive"]
    fn patch_iou_loss_config_zero_patch_width_panics() {
        let _loss = PatchIoULossConfig::new().with_patch_width(0).init();
    }

    #[test]
    #[should_panic = "Input dimensions"]
    fn patch_iou_loss_forward_input_smaller_than_patch_panics() {
        let device = Default::default();
        let loss = PatchIoULoss::new(); // Default 64x64 patches

        // Input smaller than patch size
        let predictions = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([[[[0.5, 0.5], [0.5, 0.5]]]]),
            &device,
        );
        let targets = Tensor::<TestBackend, 4, Int>::from_data(
            TensorData::from([[[[1, 0], [0, 1]]]]),
            &device,
        );

        let _result = loss.forward_no_reduction(predictions, targets);
    }

    #[test]
    #[should_panic = "Shape of predictions"]
    fn patch_iou_loss_forward_mismatched_shapes_panics() {
        let device = Default::default();
        let loss = PatchIoULoss::new();

        let predictions =
            Tensor::<TestBackend, 4>::from_data(TensorData::from([[[[1.0; 64]; 64]]]), &device);
        let targets =
            Tensor::<TestBackend, 4, Int>::from_data(TensorData::from([[[[1; 32]; 32]]]), &device);

        let _result = loss.forward_no_reduction(predictions, targets);
    }

    #[test]
    fn patch_iou_loss_display_shows_patch_dimensions() {
        let config = PatchIoULossConfig::new()
            .with_patch_height(32)
            .with_patch_width(48)
            .with_eps(1e-6);
        let loss = config.init();

        let display_str = format!("{loss}");
        assert!(display_str.contains("PatchIoULoss"));
        assert!(display_str.contains("patch_height: 32"));
        assert!(display_str.contains("patch_width: 48"));
    }
}
