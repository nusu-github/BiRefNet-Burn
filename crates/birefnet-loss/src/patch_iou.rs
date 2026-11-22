//! Patch-wise Intersection over Union loss for local segmentation consistency.

use burn::{
    module::{Content, DisplaySettings, ModuleDisplay},
    nn::loss::Reduction,
    prelude::*,
    tensor::ops::PadMode,
};

use crate::ops::{assert_same_shape, reduce_loss};

/// Configuration for [`PatchIouLoss`].
#[derive(Config, Debug, PartialEq)]
pub struct PatchIouLossConfig {
    /// Height of each spatial patch window.
    #[config(default = 64)]
    pub patch_height: usize,
    /// Width of each spatial patch window.
    #[config(default = 64)]
    pub patch_width: usize,
    /// Smoothing term added to each patch's intersection and union.
    #[config(default = 1e-8)]
    pub eps: f64,
}

impl PatchIouLossConfig {
    /// Initializes a [`PatchIouLoss`] module.
    ///
    /// # Panics
    ///
    /// Panics if `patch_height == 0`, `patch_width == 0`, or `eps` is not finite and positive.
    #[must_use]
    pub fn init(&self) -> PatchIouLoss {
        assert!(
            self.patch_height > 0 && self.patch_width > 0,
            "PatchIouLoss patch dimensions must be positive, got {}x{}",
            self.patch_height,
            self.patch_width
        );
        assert!(
            self.eps.is_finite() && self.eps > 0.0,
            "PatchIouLoss eps must be finite and positive, got {}",
            self.eps
        );
        PatchIouLoss {
            patch_height: self.patch_height,
            patch_width: self.patch_width,
            eps: self.eps,
        }
    }
}

/// Patch-wise IoU loss: the sum of per-patch soft IoU losses over a non-overlapping grid.
///
/// Edge patches that extend past the image are clipped, matching a sliding-window loop.
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct PatchIouLoss {
    patch_height: usize,
    patch_width: usize,
    eps: f64,
}

impl Default for PatchIouLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl ModuleDisplay for PatchIouLoss {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("patch_height", &self.patch_height)
            .add("patch_width", &self.patch_width)
            .add("eps", &self.eps)
            .optional()
    }
}

impl PatchIouLoss {
    /// Creates a [`PatchIouLoss`] with default configuration (`64x64` patches).
    #[must_use]
    pub fn new() -> Self {
        PatchIouLossConfig::new().init()
    }

    /// Computes reduced patch IoU loss `[1]` over probabilities in `[0, 1]`.
    ///
    /// `Reduction::Mean` averages the per-sample patch sums over the batch.
    ///
    /// # Panics
    ///
    /// Panics if `predictions` and `targets` have different shapes.
    #[must_use]
    pub fn forward(
        &self,
        predictions: Tensor<4>,
        targets: Tensor<4>,
        reduction: Reduction,
    ) -> Tensor<1> {
        let loss = self.forward_no_reduction(predictions, targets);
        reduce_loss(loss, reduction)
    }

    /// Computes the per-sample sum of patch IoU losses as `[batch_size]`.
    ///
    /// # Panics
    ///
    /// Panics if `predictions` and `targets` have different shapes.
    #[must_use]
    pub fn forward_no_reduction(&self, predictions: Tensor<4>, targets: Tensor<4>) -> Tensor<1> {
        assert_same_shape(&predictions, &targets);

        let pred_patches = self.to_patches(predictions);
        let target_patches = self.to_patches(targets);

        let intersection = (pred_patches.clone() * target_patches.clone()).sum_dim(2);
        let union = pred_patches.sum_dim(2) + target_patches.sum_dim(2) - intersection.clone();
        let iou = intersection.add_scalar(self.eps) / union.add_scalar(self.eps);

        iou.neg()
            .add_scalar(1.0)
            .sum_dim(1)
            .squeeze_dims::<1>(&[1, 2])
    }

    /// Rearranges `[B, C, H, W]` into `[B, patches, C * patch_h * patch_w]`.
    ///
    /// Inputs are zero-padded up to a multiple of the patch size; zeros add nothing to
    /// intersections or unions, so padded patches equal the clipped edge patches.
    fn to_patches(&self, tensor: Tensor<4>) -> Tensor<3> {
        let [batch, channels, height, width] = tensor.dims();
        let (ph, pw) = (self.patch_height, self.patch_width);
        let rows = height.div_ceil(ph);
        let cols = width.div_ceil(pw);
        let pad_h = rows * ph - height;
        let pad_w = cols * pw - width;

        let padded = if pad_h == 0 && pad_w == 0 {
            tensor
        } else {
            tensor.pad((0, pad_w, 0, pad_h), PadMode::Constant(0.0))
        };

        padded
            .reshape([batch, channels, rows, ph, cols, pw])
            .permute([0, 2, 4, 1, 3, 5])
            .reshape([batch, rows * cols, channels * ph * pw])
    }
}
