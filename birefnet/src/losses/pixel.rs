//! Pixel loss function that combines multiple loss types for segmentation training.
//!
//! This module implements the PixLoss class equivalent to the original PyTorch implementation,
//! providing configurable weighted combinations of different loss functions.

use std::collections::HashMap;

use super::{
    bce::{BCELoss, BCELossConfig},
    combined::CombinedLoss,
    contour::ContourLoss,
    iou::PatchIoULoss,
    mae::MaeLoss,
    ssim::SSIMLoss,
    structure::StructureLoss,
    threshold_regularization::ThrRegLoss,
    triplet::TripletLoss,
};
use burn::nn::loss::Reduction;
use burn::tensor::activation;
use burn::tensor::cast::ToElement;
use burn::{
    nn::loss::MseLoss,
    prelude::*,
    tensor::{
        backend::Backend,
        module::interpolate,
        ops::{InterpolateMode, InterpolateOptions},
        ElementConversion, Tensor,
    },
};

/// Configuration for different loss components in PixLoss.
#[derive(Config, Debug)]
pub struct LossWeightsConfig {
    #[config(default = 30.0)]
    pub bce: f32,
    #[config(default = 0.5)]
    pub iou: f32,
    #[config(default = 0.5)]
    pub iou_patch: f32,
    #[config(default = 30.0)]
    pub mae: f32,
    #[config(default = 30.0)]
    pub mse: f32,
    #[config(default = 3.0)]
    pub triplet: f32,
    #[config(default = 100.0)]
    pub reg: f32,
    #[config(default = 10.0)]
    pub ssim: f32,
    #[config(default = 5.0)]
    pub cnt: f32,
    #[config(default = 5.0)]
    pub structure: f32,
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
    bce_weight: f32,
    iou_weight: f32,
    iou_patch_weight: f32,
    mae_weight: f32,
    mse_weight: f32,
    triplet_weight: f32,
    reg_weight: f32,
    ssim_weight: f32,
    cnt_weight: f32,
    structure_weight: f32,

    // Loss components (optional based on weights)
    pub bce_loss: Option<BCELoss<B>>,
    pub iou_loss: Option<CombinedLoss<B>>,
    pub iou_patch_loss: Option<PatchIoULoss<B>>,
    pub mae_loss: Option<MaeLoss<B>>,
    pub mse_loss: Option<MseLoss>,
    pub triplet_loss: Option<TripletLoss<B>>,
    pub reg_loss: Option<ThrRegLoss<B>>,
    pub ssim_loss: Option<SSIMLoss<B>>,
    pub cnt_loss: Option<ContourLoss<B>>,
    pub structure_loss: Option<StructureLoss<B>>,
}

impl PixLossConfig {
    /// Initialize a new PixLoss with the given configuration.
    pub fn init<B: Backend>(&self, _device: &B::Device) -> PixLoss<B> {
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
            bce_loss: if weights.bce > 0.0 {
                Some(BCELossConfig::new().init())
            } else {
                None
            },
            iou_loss: if weights.iou > 0.0 {
                Some(CombinedLoss::new())
            } else {
                None
            },
            iou_patch_loss: if weights.iou_patch > 0.0 {
                Some(PatchIoULoss::new())
            } else {
                None
            },
            mae_loss: if weights.mae > 0.0 {
                Some(MaeLoss::new())
            } else {
                None
            },
            mse_loss: if weights.mse > 0.0 {
                Some(MseLoss::new())
            } else {
                None
            },
            triplet_loss: if weights.triplet > 0.0 {
                Some(TripletLoss::new())
            } else {
                None
            },
            reg_loss: if weights.reg > 0.0 {
                Some(ThrRegLoss::new())
            } else {
                None
            },
            ssim_loss: if weights.ssim > 0.0 {
                Some(SSIMLoss::new())
            } else {
                None
            },
            cnt_loss: if weights.cnt > 0.0 {
                Some(ContourLoss::new())
            } else {
                None
            },
            structure_loss: if weights.structure > 0.0 {
                Some(StructureLoss::new())
            } else {
                None
            },
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
        gt: Tensor<B, 4>,
        pix_loss_lambda: f32,
    ) -> (Tensor<B, 1>, HashMap<String, f64>) {
        let mut total_loss: Option<Tensor<B, 1>> = None;
        let mut loss_dict: HashMap<String, f64> = HashMap::new();
        let device = gt.device();

        for pred_lvl in scaled_preds.iter() {
            // Resize prediction to match ground truth if necessary
            let pred_resized = if pred_lvl.dims() != gt.dims() {
                let [_, _, h, w] = gt.dims();
                let options = InterpolateOptions::new(InterpolateMode::Bilinear);
                interpolate(pred_lvl.clone(), [h, w], options)
            } else {
                pred_lvl.clone()
            };

            // Apply sigmoid to predictions
            let pred_sigmoid = activation::sigmoid(pred_resized.clone());

            // Calculate individual losses
            let mut level_loss: Option<Tensor<B, 1>> = None;

            // BCE Loss
            if let Some(ref bce_loss) = self.bce_loss {
                let loss = bce_loss.forward(pred_resized.clone(), gt.clone())
                    * self.bce_weight.elem::<B::FloatElem>()
                    * pix_loss_lambda.elem::<B::FloatElem>();
                level_loss = Some(level_loss.map_or_else(|| loss.clone(), |l| l + loss.clone()));
                *loss_dict.entry("bce".to_string()).or_insert(0.0) +=
                    loss.into_scalar().to_f64() / scaled_preds.len() as f64;
            }

            // IoU Loss (using combined loss which includes IoU)
            if let Some(ref iou_loss) = self.iou_loss {
                let loss = iou_loss.iou_loss(pred_resized.clone(), gt.clone())
                    * self.iou_weight.elem::<B::FloatElem>()
                    * pix_loss_lambda.elem::<B::FloatElem>();
                level_loss = Some(level_loss.map_or_else(|| loss.clone(), |l| l + loss.clone()));
                *loss_dict.entry("iou".to_string()).or_insert(0.0) +=
                    loss.into_scalar().to_f64() / scaled_preds.len() as f64;
            }

            // Patch IoU Loss
            if let Some(ref iou_patch_loss) = self.iou_patch_loss {
                let loss = iou_patch_loss.forward(pred_sigmoid.clone(), gt.clone())
                    * self.iou_patch_weight.elem::<B::FloatElem>()
                    * pix_loss_lambda.elem::<B::FloatElem>();
                level_loss = Some(level_loss.map_or_else(|| loss.clone(), |l| l + loss.clone()));
                *loss_dict.entry("iou_patch".to_string()).or_insert(0.0) +=
                    loss.into_scalar().to_f64() / scaled_preds.len() as f64;
            }

            // MAE Loss
            if let Some(ref mae_loss) = self.mae_loss {
                let loss = mae_loss.forward(pred_sigmoid.clone(), gt.clone())
                    * self.mae_weight.elem::<B::FloatElem>()
                    * pix_loss_lambda.elem::<B::FloatElem>();
                level_loss = Some(level_loss.map_or_else(|| loss.clone(), |l| l + loss.clone()));
                *loss_dict.entry("mae".to_string()).or_insert(0.0) +=
                    loss.into_scalar().to_f64() / scaled_preds.len() as f64;
            }

            // MSE Loss
            if let Some(ref mse_loss) = self.mse_loss {
                let loss: Tensor<B, 1> =
                    mse_loss.forward(pred_sigmoid.clone(), gt.clone(), Reduction::Mean)
                        * self.mse_weight.elem::<B::FloatElem>()
                        * pix_loss_lambda.elem::<B::FloatElem>();
                level_loss = Some(level_loss.map_or_else(|| loss.clone(), |l| l + loss.clone()));
                *loss_dict.entry("mse".to_string()).or_insert(0.0) +=
                    loss.into_scalar().to_f64() / scaled_preds.len() as f64;
            }

            // Triplet Loss
            if let Some(ref triplet_loss) = self.triplet_loss {
                let loss = triplet_loss.forward(pred_resized.clone(), gt.clone())
                    * self.triplet_weight.elem::<B::FloatElem>()
                    * pix_loss_lambda.elem::<B::FloatElem>();
                level_loss = Some(level_loss.map_or_else(|| loss.clone(), |l| l + loss.clone()));
                *loss_dict.entry("triplet".to_string()).or_insert(0.0) +=
                    loss.into_scalar().to_f64() / scaled_preds.len() as f64;
            }

            // Threshold Regularization Loss
            if let Some(ref reg_loss) = self.reg_loss {
                let loss = reg_loss.forward(pred_sigmoid.clone(), gt.clone())
                    * self.reg_weight.elem::<B::FloatElem>()
                    * pix_loss_lambda.elem::<B::FloatElem>();
                level_loss = Some(level_loss.map_or_else(|| loss.clone(), |l| l + loss.clone()));
                *loss_dict.entry("reg".to_string()).or_insert(0.0) +=
                    loss.into_scalar().to_f64() / scaled_preds.len() as f64;
            }

            // SSIM Loss
            if let Some(ref ssim_loss) = self.ssim_loss {
                let loss = ssim_loss.forward(pred_sigmoid.clone(), gt.clone())
                    * self.ssim_weight.elem::<B::FloatElem>()
                    * pix_loss_lambda.elem::<B::FloatElem>();
                level_loss = Some(level_loss.map_or_else(|| loss.clone(), |l| l + loss.clone()));
                *loss_dict.entry("ssim".to_string()).or_insert(0.0) +=
                    loss.into_scalar().to_f64() / scaled_preds.len() as f64;
            }

            // Contour Loss
            if let Some(ref cnt_loss) = self.cnt_loss {
                let loss = cnt_loss.forward(pred_resized.clone(), gt.clone())
                    * self.cnt_weight.elem::<B::FloatElem>()
                    * pix_loss_lambda.elem::<B::FloatElem>();
                level_loss = Some(level_loss.map_or_else(|| loss.clone(), |l| l + loss.clone()));
                *loss_dict.entry("cnt".to_string()).or_insert(0.0) +=
                    loss.into_scalar().to_f64() / scaled_preds.len() as f64;
            }

            // Structure Loss
            if let Some(ref structure_loss) = self.structure_loss {
                let loss = structure_loss.forward(pred_resized, gt.clone())
                    * self.structure_weight.elem::<B::FloatElem>()
                    * pix_loss_lambda.elem::<B::FloatElem>();
                level_loss = Some(level_loss.map_or_else(|| loss.clone(), |l| l + loss.clone()));
                *loss_dict.entry("structure".to_string()).or_insert(0.0) +=
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
