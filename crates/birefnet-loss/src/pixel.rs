//! Pixel loss function that combines multiple loss types for segmentation training.
//!
//! This module implements the PixLoss class equivalent to the original PyTorch implementation,
//! providing configurable weighted combinations of different loss functions.

use std::collections::HashMap;

use burn::{
    nn::loss::{BinaryCrossEntropyLoss, BinaryCrossEntropyLossConfig, MseLoss, Reduction},
    prelude::*,
    tensor::{
        Tensor, activation,
        backend::Backend,
        cast::ToElement,
        module::interpolate,
        ops::{InterpolateMode, InterpolateOptions},
    },
};

use super::{
    ContourLossConfig, IoULossConfig, MaeLossConfig, SSIMLossConfig, StructureLossConfig,
    ThresholdRegularizationLossConfig,
    contour::ContourLoss,
    iou::IoULoss,
    mae::MaeLoss,
    patch_iou::{PatchIoULoss, PatchIoULossConfig},
    ssim::SSIMLoss,
    structure::StructureLoss,
    threshold_regularization::ThresholdRegularizationLoss,
};

/// Configuration for different loss components in PixLoss.
#[derive(Config, Debug)]
pub struct LossWeightsConfig {
    #[config(default = 30.0)]
    pub bce: f64,
    #[config(default = 0.5)]
    pub iou: f64,
    #[config(default = 0.5)]
    pub iou_patch: f64,
    #[config(default = 30.0)]
    pub mae: f64,
    #[config(default = 30.0)]
    pub mse: f64,
    #[config(default = 3.0)]
    pub triplet: f64,
    #[config(default = 100.0)]
    pub reg: f64,
    #[config(default = 10.0)]
    pub ssim: f64,
    #[config(default = 5.0)]
    pub cnt: f64,
    #[config(default = 5.0)]
    pub structure: f64,
}

/// Configuration for PixLoss.
#[derive(Config, Debug)]
pub struct PixLossConfig {
    pub loss_weights: LossWeightsConfig,
}

/// Pixel loss function that combines multiple loss types.
///
/// This class replicates the functionality of the original PyTorch PixLoss class,
/// combining various loss functions with configurable weights.
#[derive(Module, Debug)]
pub struct PixLoss<B: Backend> {
    // Store loss weights as separate values (not in Module field)
    bce_weight: f64,
    iou_weight: f64,
    iou_patch_weight: f64,
    mae_weight: f64,
    mse_weight: f64,
    triplet_weight: f64,
    reg_weight: f64,
    ssim_weight: f64,
    cnt_weight: f64,
    structure_weight: f64,

    // Loss components (optional based on weights)
    pub bce_loss: Option<BinaryCrossEntropyLoss<B>>,
    pub iou_loss: Option<IoULoss>,
    pub iou_patch_loss: Option<PatchIoULoss>,
    pub mae_loss: Option<MaeLoss>,
    pub mse_loss: Option<MseLoss>,
    pub reg_loss: Option<ThresholdRegularizationLoss>,
    pub ssim_loss: Option<SSIMLoss>,
    pub cnt_loss: Option<ContourLoss>,
    pub structure_loss: Option<StructureLoss>,
}

impl PixLossConfig {
    /// Initialize a new PixLoss with the given configuration.
    pub fn init<B: Backend>(&self, device: &B::Device) -> PixLoss<B> {
        let weights = &self.loss_weights;

        PixLoss {
            // Store weights as individual fields
            bce_weight: weights.bce,
            iou_weight: weights.iou,
            iou_patch_weight: weights.iou_patch,
            mae_weight: weights.mae,
            mse_weight: weights.mse,
            triplet_weight: weights.triplet,
            reg_weight: weights.reg,
            ssim_weight: weights.ssim,
            cnt_weight: weights.cnt,
            structure_weight: weights.structure,

            // Initialize loss components only if their weights are non-zero
            bce_loss: (weights.bce > 0.0).then(|| BinaryCrossEntropyLossConfig::new().init(device)),
            iou_loss: (weights.iou > 0.0).then(|| IoULossConfig::new().init()),
            iou_patch_loss: (weights.iou_patch > 0.0).then(|| PatchIoULossConfig::new().init()),
            mae_loss: (weights.mae > 0.0).then(|| MaeLossConfig::new().init()),
            mse_loss: (weights.mse > 0.0).then(MseLoss::new),
            reg_loss: (weights.reg > 0.0).then(|| ThresholdRegularizationLossConfig::new().init()),
            ssim_loss: (weights.ssim > 0.0).then(|| SSIMLossConfig::new().init()),
            cnt_loss: (weights.cnt > 0.0).then(|| ContourLossConfig::new().init()),
            structure_loss: (weights.structure > 0.0).then(|| StructureLossConfig::new().init()),
        }
    }
}

impl<B: Backend> PixLoss<B> {
    /// Get a copy of the current loss weights for compatibility.
    pub const fn loss_weights(&self) -> LossWeightsConfig {
        LossWeightsConfig {
            bce: self.bce_weight,
            iou: self.iou_weight,
            iou_patch: self.iou_patch_weight,
            mae: self.mae_weight,
            mse: self.mse_weight,
            triplet: self.triplet_weight,
            reg: self.reg_weight,
            ssim: self.ssim_weight,
            cnt: self.cnt_weight,
            structure: self.structure_weight,
        }
    }

    /// Calculate pixel loss for multi-scale predictions.
    ///
    /// # Arguments
    /// * `scaled_preds` - List of predictions at different scales
    /// * `gt` - Ground truth segmentation map
    /// * `pix_loss_lambda` - Global scaling factor for pixel loss
    ///
    /// # Returns
    /// A tuple of (total_loss, loss_dict) where loss_dict contains individual loss values
    pub fn forward(
        &self,
        scaled_preds: Vec<Tensor<B, 4>>,
        gt: Tensor<B, 4, Int>,
        pix_loss_lambda: f64,
    ) -> (Tensor<B, 1>, HashMap<String, f64>) {
        let mut total_loss: Option<Tensor<B, 1>> = None;
        let mut loss_dict: HashMap<String, f64> = HashMap::new();
        let device = gt.device();

        for pred_lvl in &scaled_preds {
            // Resize prediction to match ground truth if necessary
            let pred_resized = if pred_lvl.dims() == gt.dims() {
                pred_lvl.clone()
            } else {
                let [_, _, h, w] = gt.dims();
                let options = InterpolateOptions::new(InterpolateMode::Nearest);
                interpolate(pred_lvl.clone(), [h, w], options)
            };

            // Apply sigmoid to predictions
            let pred_sigmoid = activation::sigmoid(pred_resized.clone());

            // Calculate individual losses
            let mut level_loss: Option<Tensor<B, 1>> = None;

            // BCE Loss
            if let Some(ref bce_loss) = self.bce_loss {
                let loss = bce_loss
                    .forward(pred_sigmoid.clone(), gt.clone())
                    .mul_scalar(self.bce_weight)
                    .mul_scalar(pix_loss_lambda);
                level_loss = Some(level_loss.map_or_else(|| loss.clone(), |l| l + loss.clone()));
                *loss_dict.entry("bce".to_owned()).or_insert(0.0) +=
                    loss.into_scalar().to_f64() / scaled_preds.len() as f64;
            }

            // IoU Loss (using combined loss which includes IoU)
            if let Some(ref iou_loss) = self.iou_loss {
                let loss = iou_loss
                    .forward(pred_resized.clone(), gt.clone(), Reduction::Mean)
                    .mul_scalar(self.iou_weight)
                    .mul_scalar(pix_loss_lambda);
                level_loss = Some(level_loss.map_or_else(|| loss.clone(), |l| l + loss.clone()));
                *loss_dict.entry("iou".to_owned()).or_insert(0.0) +=
                    loss.into_scalar().to_f64() / scaled_preds.len() as f64;
            }

            // Patch IoU Loss
            if let Some(ref iou_patch_loss) = self.iou_patch_loss {
                let loss = iou_patch_loss
                    .forward_no_reduction(pred_sigmoid.clone(), gt.clone())
                    .mul_scalar(self.iou_patch_weight)
                    .mul_scalar(pix_loss_lambda);
                level_loss = Some(level_loss.map_or_else(|| loss.clone(), |l| l + loss.clone()));
                *loss_dict.entry("iou_patch".to_owned()).or_insert(0.0) +=
                    loss.into_scalar().to_f64() / scaled_preds.len() as f64;
            }

            // MAE Loss
            if let Some(ref mae_loss) = self.mae_loss {
                let loss = mae_loss
                    .forward(pred_sigmoid.clone(), gt.clone().float(), Reduction::Mean)
                    .mul_scalar(self.mae_weight)
                    .mul_scalar(pix_loss_lambda);
                level_loss = Some(level_loss.map_or_else(|| loss.clone(), |l| l + loss.clone()));
                *loss_dict.entry("mae".to_owned()).or_insert(0.0) +=
                    loss.into_scalar().to_f64() / scaled_preds.len() as f64;
            }

            // MSE Loss
            if let Some(ref mse_loss) = self.mse_loss {
                let loss: Tensor<B, 1> = mse_loss
                    .forward(pred_sigmoid.clone(), gt.clone().float(), Reduction::Mean)
                    .mul_scalar(self.mse_weight)
                    .mul_scalar(pix_loss_lambda);
                level_loss = Some(level_loss.map_or_else(|| loss.clone(), |l| l + loss.clone()));
                *loss_dict.entry("mse".to_owned()).or_insert(0.0) +=
                    loss.into_scalar().to_f64() / scaled_preds.len() as f64;
            }

            // Threshold Regularization Loss
            if let Some(ref reg_loss) = self.reg_loss {
                let loss = reg_loss
                    .forward(pred_sigmoid.clone(), Reduction::Mean)
                    .mul_scalar(self.reg_weight)
                    .mul_scalar(pix_loss_lambda);
                level_loss = Some(level_loss.map_or_else(|| loss.clone(), |l| l + loss.clone()));
                *loss_dict.entry("reg".to_owned()).or_insert(0.0) +=
                    loss.into_scalar().to_f64() / scaled_preds.len() as f64;
            }

            // SSIM Loss
            if let Some(ref ssim_loss) = self.ssim_loss {
                let loss = ssim_loss
                    .forward(pred_sigmoid.clone(), gt.clone().float(), Reduction::Mean)
                    .mul_scalar(self.ssim_weight)
                    .mul_scalar(pix_loss_lambda);
                level_loss = Some(level_loss.map_or_else(|| loss.clone(), |l| l + loss.clone()));
                *loss_dict.entry("ssim".to_owned()).or_insert(0.0) +=
                    loss.into_scalar().to_f64() / scaled_preds.len() as f64;
            }

            // Contour Loss
            if let Some(ref cnt_loss) = self.cnt_loss {
                let loss = cnt_loss
                    .forward_no_reduction(pred_resized.clone(), gt.clone())
                    .mul_scalar(self.cnt_weight)
                    .mul_scalar(pix_loss_lambda);
                level_loss = Some(level_loss.map_or_else(|| loss.clone(), |l| l + loss.clone()));
                *loss_dict.entry("cnt".to_owned()).or_insert(0.0) +=
                    loss.into_scalar().to_f64() / scaled_preds.len() as f64;
            }

            // Structure Loss
            if let Some(ref structure_loss) = self.structure_loss {
                let loss = structure_loss
                    .forward(pred_resized, gt.clone(), Reduction::Mean)
                    .mul_scalar(self.structure_weight)
                    .mul_scalar(pix_loss_lambda);
                level_loss = Some(level_loss.map_or_else(|| loss.clone(), |l| l + loss.clone()));
                *loss_dict.entry("structure".to_owned()).or_insert(0.0) +=
                    loss.into_scalar().to_f64() / scaled_preds.len() as f64;
            }

            // Add level loss to total loss
            if let Some(level_loss) = level_loss {
                total_loss = Some(match total_loss {
                    Some(t) => t + level_loss,
                    None => level_loss,
                });
            }
        }

        // Return total loss and loss dictionary
        let final_loss = total_loss.unwrap_or_else(|| Tensor::zeros([1], &device));

        (final_loss, loss_dict)
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::cast::ToElement;

    use super::*;
    use crate::tests::TestBackend;

    #[test]
    fn pixel_loss_forward_produces_finite_values() {
        let device = Default::default();
        let config = PixLossConfig::new(LossWeightsConfig::new());
        let loss = config.init::<TestBackend>(&device);

        // Single prediction level (64x64 minimum size for patch IoU)
        let pred = Tensor::<TestBackend, 4>::from_floats([[[[0.8; 64]; 64]]], &device);
        let gt = Tensor::<TestBackend, 4, Int>::from_ints([[[[1; 64]; 64]]], &device);

        let scaled_preds = vec![pred];
        let (total_loss, loss_dict) = loss.forward(scaled_preds, gt, 1.0);

        // Should return valid loss and dictionary
        assert!(total_loss.into_scalar().to_f64().is_finite());
        assert!(!loss_dict.is_empty());

        // Should have entries for enabled losses
        if loss.bce_weight > 0.0 {
            assert!(loss_dict.contains_key("bce"));
        }
        if loss.iou_weight > 0.0 {
            assert!(loss_dict.contains_key("iou"));
        }
    }

    #[test]
    fn pixel_loss_handles_multiple_prediction_scales() {
        let device = Default::default();
        // Disable patch IoU to allow smaller tensor sizes
        let weights = LossWeightsConfig::new().with_iou_patch(0.0);
        let config = PixLossConfig::new(weights);
        let loss = config.init::<TestBackend>(&device);

        // Multiple prediction levels (64x64 sizes to avoid stack overflow)
        let pred1 = Tensor::<TestBackend, 4>::from_floats([[[[0.8; 64]; 64]]], &device);
        let pred2 = Tensor::<TestBackend, 4>::from_floats([[[[0.5; 32]; 32]]], &device);
        let gt = Tensor::<TestBackend, 4, Int>::from_ints([[[[1; 64]; 64]]], &device);

        let scaled_preds = vec![pred1, pred2];
        let (total_loss, loss_dict) = loss.forward(scaled_preds, gt, 1.0);

        // Should handle multiple scales correctly
        assert!(total_loss.into_scalar().to_f64().is_finite());
        assert!(!loss_dict.is_empty());
    }

    #[test]
    fn pixel_loss_scales_correctly_with_lambda() {
        let device = Default::default();
        let config = PixLossConfig::new(LossWeightsConfig::new());
        let loss = config.init::<TestBackend>(&device);

        let pred = Tensor::<TestBackend, 4>::from_floats([[[[0.5; 64]; 64]]], &device);
        let gt = Tensor::<TestBackend, 4, Int>::from_ints([[[[1; 64]; 64]]], &device);

        let scaled_preds = vec![pred];

        // Test with different lambda values
        let (loss1, _) = loss.forward(scaled_preds.clone(), gt.clone(), 1.0);
        let (loss2, _) = loss.forward(scaled_preds, gt, 2.0);

        let val1 = loss1.into_scalar().to_f64();
        let val2 = loss2.into_scalar().to_f64();

        // Second loss should be approximately 2x the first
        assert!(
            2.0f64.mul_add(-val1, val2).abs() < 1e-4,
            "Loss scaling failed: {val1} vs {val2}"
        );
    }

    #[test]
    fn pixel_loss_returns_zero_for_empty_predictions() {
        let device = Default::default();
        let config = PixLossConfig::new(LossWeightsConfig::new());
        let loss = config.init::<TestBackend>(&device);

        let gt = Tensor::<TestBackend, 4, Int>::from_ints([[[[1; 64]; 64]]], &device);

        let scaled_preds: Vec<Tensor<TestBackend, 4>> = vec![];
        let (total_loss, loss_dict) = loss.forward(scaled_preds, gt, 1.0);

        // Should return zero loss for empty predictions
        assert_eq!(total_loss.into_scalar().to_f64(), 0.0);
        assert!(loss_dict.is_empty());
    }
}
