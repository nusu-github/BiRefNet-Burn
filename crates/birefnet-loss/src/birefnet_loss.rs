//! Integrated multi-task BiRefNet loss combining pixel and auxiliary classification terms.

use burn::{nn::loss::Reduction, prelude::*};

use crate::{
    breakdown::{Component, LossBreakdown, read_breakdown},
    classification::{ClassificationLoss, ClassificationLossConfig},
    error::LossError,
    pixel::{LossWeightsConfig, PixLoss, PixLossConfig},
};

/// Configuration for [`BiRefNetLoss`].
#[derive(Config, Debug, PartialEq)]
pub struct BiRefNetLossConfig {
    /// Pixel-level loss component weights.
    #[config(default = "default_pixel_weights()")]
    pub pixel_weights: LossWeightsConfig,
    /// Weight for auxiliary classification loss (disabled when `<= 0.0`).
    #[config(default = 5.0)]
    pub classification_weight: f64,
    /// Global scalar multiplier applied across pixel and classification losses.
    #[config(default = 1.0)]
    pub global_scale: f64,
}

/// Upstream `General` task weights: BCE 30, IoU 0.5, MAE 100, SSIM 10.
fn default_pixel_weights() -> LossWeightsConfig {
    LossWeightsConfig::new()
        .with_bce(30.0)
        .with_iou(0.5)
        .with_iou_patch(0.0)
        .with_mae(100.0)
        .with_mse(0.0)
        .with_reg(0.0)
        .with_ssim(10.0)
        .with_cnt(0.0)
        .with_structure(0.0)
}

impl BiRefNetLossConfig {
    /// Configuration preset tuned for Dichotomous Image Segmentation (DIS).
    #[must_use]
    pub fn dis_optimized() -> Self {
        Self::new().with_pixel_weights(default_pixel_weights().with_mae(0.0))
    }

    /// Configuration preset tuned for general foreground/salient object segmentation.
    #[must_use]
    pub fn general_optimized() -> Self {
        Self::new().with_pixel_weights(default_pixel_weights())
    }

    /// Configuration preset tuned for alpha matting with continuous targets.
    #[must_use]
    pub fn matting_optimized() -> Self {
        Self::new().with_pixel_weights(default_pixel_weights().with_iou(0.0))
    }

    /// Sets the BCE loss weight.
    #[must_use]
    pub const fn with_bce_weight(mut self, weight: f64) -> Self {
        self.pixel_weights.bce = weight;
        self
    }

    /// Sets the IoU loss weight.
    #[must_use]
    pub const fn with_iou_weight(mut self, weight: f64) -> Self {
        self.pixel_weights.iou = weight;
        self
    }

    /// Sets the MAE loss weight.
    #[must_use]
    pub const fn with_mae_weight(mut self, weight: f64) -> Self {
        self.pixel_weights.mae = weight;
        self
    }

    /// Sets the SSIM loss weight.
    #[must_use]
    pub const fn with_ssim_weight(mut self, weight: f64) -> Self {
        self.pixel_weights.ssim = weight;
        self
    }

    /// Disables the auxiliary classification loss branch.
    #[must_use]
    pub const fn without_classification(mut self) -> Self {
        self.classification_weight = 0.0;
        self
    }

    /// Initializes a [`BiRefNetLoss`] module.
    #[must_use]
    pub fn init(&self) -> BiRefNetLoss {
        let pixel_loss = PixLossConfig::new(self.pixel_weights.clone()).init();
        let classification_loss =
            (self.classification_weight > 0.0).then(|| ClassificationLossConfig::new().init());

        BiRefNetLoss {
            pixel_loss,
            classification_loss,
            classification_weight: self.classification_weight,
            global_scale: self.global_scale,
        }
    }
}

/// Combined pixel-level and optional auxiliary classification loss for BiRefNet training.
#[derive(Module, Debug)]
pub struct BiRefNetLoss {
    pixel_loss: PixLoss,
    classification_loss: Option<ClassificationLoss>,
    classification_weight: f64,
    global_scale: f64,
}

impl BiRefNetLoss {
    /// Returns the global scaling factor configured for this loss.
    #[must_use]
    pub const fn global_scale(&self) -> f64 {
        self.global_scale
    }

    /// Computes total pixel loss without host scalar synchronization.
    ///
    /// # Errors
    ///
    /// Returns [`LossError`] if `scaled_preds` is empty or shapes mismatch `targets`.
    pub fn forward(
        &self,
        scaled_preds: &[Tensor<4>],
        targets: &Tensor<4>,
    ) -> Result<Tensor<1>, LossError> {
        self.forward_with_classification(scaled_preds, targets, None, None)
    }

    /// Computes combined pixel and classification loss without host scalar synchronization.
    ///
    /// The classification term is added only when this loss has a classification branch and
    /// both `class_preds` and `class_targets` are provided.
    ///
    /// # Errors
    ///
    /// Returns [`LossError`] if `scaled_preds` is empty or shapes mismatch `targets`.
    pub fn forward_with_classification(
        &self,
        scaled_preds: &[Tensor<4>],
        targets: &Tensor<4>,
        class_preds: Option<&[Tensor<2>]>,
        class_targets: Option<&Tensor<1, Int>>,
    ) -> Result<Tensor<1>, LossError> {
        let pixel = self
            .pixel_loss
            .forward(scaled_preds, targets, self.global_scale)?;
        Ok(match self.classification_term(class_preds, class_targets) {
            Some(cls_val) => pixel + cls_val,
            None => pixel,
        })
    }

    /// Computes pixel loss and reads the per-component [`LossBreakdown`] in one transaction.
    ///
    /// # Errors
    ///
    /// Returns [`LossError`] if `scaled_preds` is empty, shapes mismatch `targets`, or the
    /// component values cannot be read back from the device.
    pub fn forward_detailed(
        &self,
        scaled_preds: &[Tensor<4>],
        targets: &Tensor<4>,
    ) -> Result<(Tensor<1>, LossBreakdown), LossError> {
        self.forward_detailed_with_classification(scaled_preds, targets, None, None)
    }

    /// Computes pixel + classification losses alongside a [`LossBreakdown`] for logging.
    ///
    /// # Errors
    ///
    /// Returns [`LossError`] if `scaled_preds` is empty, shapes mismatch `targets`, or the
    /// component values cannot be read back from the device.
    pub fn forward_detailed_with_classification(
        &self,
        scaled_preds: &[Tensor<4>],
        targets: &Tensor<4>,
        class_preds: Option<&[Tensor<2>]>,
        class_targets: Option<&Tensor<1, Int>>,
    ) -> Result<(Tensor<1>, LossBreakdown), LossError> {
        let terms = self
            .pixel_loss
            .compute(scaled_preds, targets, self.global_scale, true)?;
        let mut total_loss = terms.total;
        let mut components = terms.components;

        if let Some(cls_val) = self.classification_term(class_preds, class_targets) {
            total_loss = total_loss + cls_val.clone();
            components.push((Component::Classification, cls_val));
        }

        let breakdown = read_breakdown(total_loss.clone(), components)?;
        Ok((total_loss, breakdown))
    }

    fn classification_term(
        &self,
        class_preds: Option<&[Tensor<2>]>,
        class_targets: Option<&Tensor<1, Int>>,
    ) -> Option<Tensor<1>> {
        let cls_loss = self.classification_loss.as_ref()?;
        let (preds, cls_targets) = (class_preds?, class_targets?);
        Some(
            cls_loss
                .forward(preds, cls_targets, Reduction::Mean)
                .mul_scalar(self.classification_weight * self.global_scale),
        )
    }
}
