//! Patch-wise Intersection over Union loss for local segmentation consistency.

use burn::{
    module::{Content, DisplaySettings, ModuleDisplay},
    nn::loss::Reduction,
    prelude::*,
};

use crate::{
    iou::{IoULoss, IoULossConfig},
    ops::{accumulate_tensor, assert_same_shape, reduce_loss},
};

/// Configuration for [`PatchIoULoss`].
#[derive(Config, Debug, PartialEq)]
pub struct PatchIoULossConfig {
    /// Height of each spatial patch window.
    #[config(default = 64)]
    pub patch_height: usize,
    /// Width of each spatial patch window.
    #[config(default = 64)]
    pub patch_width: usize,
    /// Small epsilon passed to each patch [`IoULoss`].
    #[config(default = 1e-8)]
    pub eps: f64,
}

impl PatchIoULossConfig {
    /// Initializes a [`PatchIoULoss`] module.
    ///
    /// # Panics
    ///
    /// Panics if `patch_height == 0`, `patch_width == 0`, or `eps` is not finite and positive.
    #[must_use]
    pub fn init(&self) -> PatchIoULoss {
        assert!(
            self.patch_height > 0 && self.patch_width > 0,
            "PatchIoULoss patch dimensions must be positive, got {}x{}",
            self.patch_height,
            self.patch_width
        );
        PatchIoULoss {
            patch_height: self.patch_height,
            patch_width: self.patch_width,
            iou_loss: IoULossConfig::new().with_eps(self.eps).init(),
        }
    }
}

/// Patch-wise IoU loss that partitions predictions into spatial patches and sums per-patch IoU losses.
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct PatchIoULoss {
    patch_height: usize,
    patch_width: usize,
    iou_loss: IoULoss,
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
    /// Creates a [`PatchIoULoss`] with default configuration (`64x64` patches).
    #[must_use]
    pub fn new() -> Self {
        PatchIoULossConfig::new().init()
    }

    /// Computes reduced patch IoU loss `[1]` over probabilities in `[0, 1]`.
    #[must_use]
    pub fn forward(
        &self,
        predictions: &Tensor<4>,
        targets: &Tensor<4>,
        reduction: Reduction,
    ) -> Tensor<1> {
        let loss = self.forward_no_reduction(predictions, targets);
        reduce_loss(loss, reduction)
    }

    /// Computes total patch IoU loss summed across spatial patches and batch items as `[1]`.
    ///
    /// # Panics
    ///
    /// Panics if shapes differ or spatial dimensions are smaller than `(patch_height, patch_width)`.
    #[must_use]
    pub fn forward_no_reduction(&self, predictions: &Tensor<4>, targets: &Tensor<4>) -> Tensor<1> {
        assert_same_shape(predictions, targets);

        let [batch_size, channels, height, width] = predictions.dims();
        assert!(
            height >= self.patch_height && width >= self.patch_width,
            "input dimensions ({height}x{width}) must be at least patch size ({}x{})",
            self.patch_height,
            self.patch_width
        );

        let mut total_loss: Option<Tensor<1>> = None;
        for anchor_y in (0..height).step_by(self.patch_height) {
            let end_y = (anchor_y + self.patch_height).min(height);
            for anchor_x in (0..width).step_by(self.patch_width) {
                let end_x = (anchor_x + self.patch_width).min(width);
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
                let patch_loss = self
                    .iou_loss
                    .forward_no_reduction(patch_pred, patch_target)
                    .sum();
                accumulate_tensor(&mut total_loss, patch_loss);
            }
        }

        total_loss
            .unwrap_or_else(|| Tensor::zeros([1], &predictions.device()))
            .reshape([1])
    }
}
