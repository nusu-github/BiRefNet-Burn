//! Combined multi-component pixel loss for BiRefNet supervision.

use std::collections::HashMap;

use burn::{
    nn::loss::{MseLoss, Reduction},
    prelude::*,
    tensor::{
        activation,
        module::interpolate,
        ops::{InterpolateMode, InterpolateOptions},
    },
};

use crate::{
    contour::{ContourLoss, ContourLossConfig},
    error::LossError,
    iou::{IoULoss, IoULossConfig},
    mae::{MaeLoss, MaeLossConfig},
    ops::{accumulate_tensor, bce_with_logits, usize_to_f64},
    patch_iou::{PatchIoULoss, PatchIoULossConfig},
    ssim::{SSIMLoss, SSIMLossConfig},
    structure::{StructureLoss, StructureLossConfig},
    threshold_regularization::{ThresholdRegularizationLoss, ThresholdRegularizationLossConfig},
};

/// Component weights for [`PixLoss`]. Components with weight `<= 0.0` are disabled.
#[derive(Config, Debug, PartialEq)]
pub struct LossWeightsConfig {
    /// Binary cross-entropy with logits weight.
    #[config(default = 30.0)]
    pub bce: f64,
    /// Soft IoU loss weight.
    #[config(default = 0.5)]
    pub iou: f64,
    /// Patch-wise IoU loss weight.
    #[config(default = 0.5)]
    pub iou_patch: f64,
    /// Mean absolute error weight.
    #[config(default = 30.0)]
    pub mae: f64,
    /// Mean squared error weight.
    #[config(default = 30.0)]
    pub mse: f64,
    /// Threshold regularization weight.
    #[config(default = 100.0)]
    pub reg: f64,
    /// Structural similarity loss weight.
    #[config(default = 10.0)]
    pub ssim: f64,
    /// Boundary contour loss weight.
    #[config(default = 5.0)]
    pub cnt: f64,
    /// Structure loss weight.
    #[config(default = 5.0)]
    pub structure: f64,
}

/// Configuration for [`PixLoss`].
#[derive(Config, Debug, PartialEq)]
pub struct PixLossConfig {
    /// Component loss weights.
    pub loss_weights: LossWeightsConfig,
}

impl PixLossConfig {
    /// Initializes a [`PixLoss`] module from `self.loss_weights`.
    #[must_use]
    pub fn init(&self, _device: &Device) -> PixLoss {
        let w = &self.loss_weights;
        PixLoss {
            bce_weight: w.bce,
            iou_weight: w.iou,
            iou_patch_weight: w.iou_patch,
            mae_weight: w.mae,
            mse_weight: w.mse,
            reg_weight: w.reg,
            ssim_weight: w.ssim,
            cnt_weight: w.cnt,
            structure_weight: w.structure,
            iou_loss: (w.iou > 0.0).then(|| IoULossConfig::new().init()),
            iou_patch_loss: (w.iou_patch > 0.0).then(|| PatchIoULossConfig::new().init()),
            mae_loss: (w.mae > 0.0).then(|| MaeLossConfig::new().init()),
            mse_loss: (w.mse > 0.0).then(MseLoss::new),
            reg_loss: (w.reg > 0.0).then(|| ThresholdRegularizationLossConfig::new().init()),
            ssim_loss: (w.ssim > 0.0).then(|| SSIMLossConfig::new().init()),
            cnt_loss: (w.cnt > 0.0).then(|| ContourLossConfig::new().init()),
            structure_loss: (w.structure > 0.0).then(|| StructureLossConfig::new().init()),
        }
    }
}

/// Multi-component pixel supervision loss summed across multi-scale prediction maps.
#[derive(Module, Debug)]
pub struct PixLoss {
    bce_weight: f64,
    iou_weight: f64,
    iou_patch_weight: f64,
    mae_weight: f64,
    mse_weight: f64,
    reg_weight: f64,
    ssim_weight: f64,
    cnt_weight: f64,
    structure_weight: f64,
    iou_loss: Option<IoULoss>,
    iou_patch_loss: Option<PatchIoULoss>,
    mae_loss: Option<MaeLoss>,
    mse_loss: Option<MseLoss>,
    reg_loss: Option<ThresholdRegularizationLoss>,
    ssim_loss: Option<SSIMLoss>,
    cnt_loss: Option<ContourLoss>,
    structure_loss: Option<StructureLoss>,
}

impl PixLoss {
    /// Returns a copy of the configured component weights.
    #[must_use]
    pub const fn loss_weights(&self) -> LossWeightsConfig {
        LossWeightsConfig {
            bce: self.bce_weight,
            iou: self.iou_weight,
            iou_patch: self.iou_patch_weight,
            mae: self.mae_weight,
            mse: self.mse_weight,
            reg: self.reg_weight,
            ssim: self.ssim_weight,
            cnt: self.cnt_weight,
            structure: self.structure_weight,
        }
    }

    /// Computes total multi-scale pixel loss without synchronizing scalar metrics to the host.
    ///
    /// # Errors
    ///
    /// Returns [`LossError`] if `scaled_preds` is empty or batch/channel dimensions mismatch `gt`.
    pub fn forward(
        &self,
        scaled_preds: &[Tensor<4>],
        gt: &Tensor<4>,
        pix_loss_lambda: f64,
    ) -> Result<Tensor<1>, LossError> {
        let (total, _) = self.compute_loss(scaled_preds, gt, pix_loss_lambda, false)?;
        Ok(total)
    }

    /// Computes total multi-scale pixel loss alongside per-component average scalars for logging.
    ///
    /// # Errors
    ///
    /// Returns [`LossError`] if `scaled_preds` is empty or batch/channel dimensions mismatch `gt`.
    pub fn forward_detailed(
        &self,
        scaled_preds: &[Tensor<4>],
        gt: &Tensor<4>,
        pix_loss_lambda: f64,
    ) -> Result<(Tensor<1>, HashMap<String, f64>), LossError> {
        self.compute_loss(scaled_preds, gt, pix_loss_lambda, true)
    }

    fn compute_loss(
        &self,
        scaled_preds: &[Tensor<4>],
        gt: &Tensor<4>,
        pix_loss_lambda: f64,
        collect_breakdown: bool,
    ) -> Result<(Tensor<1>, HashMap<String, f64>), LossError> {
        if scaled_preds.is_empty() {
            return Err(LossError::empty_predictions());
        }

        let gt_dims = gt.dims();
        for (idx, pred) in scaled_preds.iter().enumerate() {
            let pred_dims = pred.dims();
            if pred_dims[..2] != gt_dims[..2] {
                return Err(LossError::invalid_dimensions(format!(
                    "prediction[{idx}] [B, C] ({:?}) does not match targets ({:?})",
                    &pred_dims[..2],
                    &gt_dims[..2]
                )));
            }
        }

        let scale_count = usize_to_f64(scaled_preds.len());
        let [_, _, target_h, target_w] = gt_dims;
        let device = gt.device();

        let mut total_loss: Option<Tensor<1>> = None;
        let mut breakdown = HashMap::new();

        for pred_lvl in scaled_preds {
            let pred_logits = if pred_lvl.dims() == gt_dims {
                pred_lvl.clone()
            } else {
                interpolate(
                    pred_lvl.clone(),
                    InterpolateOptions::new(InterpolateMode::Nearest)
                        .with_output_size([target_h, target_w]),
                )
            };
            let pred_probs = activation::sigmoid(pred_logits.clone());

            let mut level_loss: Option<Tensor<1>> = None;
            let mut add_component = |name: &str, raw: Tensor<1>, weight: f64| {
                let weighted = raw.mul_scalar(weight * pix_loss_lambda);
                if collect_breakdown {
                    let scalar = weighted.clone().into_scalar::<f64>() / scale_count;
                    *breakdown.entry(name.to_owned()).or_insert(0.0) += scalar;
                }
                accumulate_tensor(&mut level_loss, weighted);
            };

            if self.bce_weight > 0.0 {
                let bce = bce_with_logits(pred_logits.clone(), gt.clone()).mean();
                add_component("bce", bce, self.bce_weight);
            }
            if let Some(ref iou_loss) = self.iou_loss {
                let iou = iou_loss.forward(pred_probs.clone(), gt.clone(), Reduction::Mean);
                add_component("iou", iou, self.iou_weight);
            }
            if let Some(ref iou_patch_loss) = self.iou_patch_loss {
                let patch = iou_patch_loss.forward(&pred_probs, gt, Reduction::Mean);
                add_component("iou_patch", patch, self.iou_patch_weight);
            }
            if let Some(ref mae_loss) = self.mae_loss {
                let mae = mae_loss.forward(pred_probs.clone(), gt.clone(), Reduction::Mean);
                add_component("mae", mae, self.mae_weight);
            }
            if let Some(ref mse_loss) = self.mse_loss {
                let mse = mse_loss.forward(pred_probs.clone(), gt.clone(), Reduction::Mean);
                add_component("mse", mse, self.mse_weight);
            }
            if let Some(ref reg_loss) = self.reg_loss {
                let reg = reg_loss.forward(pred_probs.clone(), Reduction::Mean);
                add_component("reg", reg, self.reg_weight);
            }
            if let Some(ref ssim_loss) = self.ssim_loss {
                let ssim = ssim_loss.forward(pred_probs.clone(), gt.clone(), Reduction::Mean);
                add_component("ssim", ssim, self.ssim_weight);
            }
            if let Some(ref cnt_loss) = self.cnt_loss {
                let cnt = cnt_loss.forward(pred_probs, gt.clone(), Reduction::Mean);
                add_component("cnt", cnt, self.cnt_weight);
            }
            if let Some(ref structure_loss) = self.structure_loss {
                let structure = structure_loss.forward(pred_logits, gt.clone(), Reduction::Mean);
                add_component("structure", structure, self.structure_weight);
            }

            if let Some(level) = level_loss {
                accumulate_tensor(&mut total_loss, level);
            }
        }

        Ok((
            total_loss.unwrap_or_else(|| Tensor::zeros([1], &device)),
            breakdown,
        ))
    }
}
