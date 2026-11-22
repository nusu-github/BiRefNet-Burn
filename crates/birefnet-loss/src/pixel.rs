//! Combined multi-component pixel loss for BiRefNet supervision.

use std::cell::OnceCell;

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
    breakdown::{Component, LossBreakdown, read_breakdown},
    contour::{ContourLoss, ContourLossConfig},
    error::LossError,
    iou::{IouLoss, IouLossConfig},
    mae::{MaeLoss, MaeLossConfig},
    ops::{accumulate_tensor, bce_with_logits},
    patch_iou::{PatchIouLoss, PatchIouLossConfig},
    ssim::{SsimLoss, SsimLossConfig},
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
    pub fn init(&self) -> PixLoss {
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
            iou_loss: (w.iou > 0.0).then(|| IouLossConfig::new().init()),
            iou_patch_loss: (w.iou_patch > 0.0).then(|| PatchIouLossConfig::new().init()),
            mae_loss: (w.mae > 0.0).then(|| MaeLossConfig::new().init()),
            mse_loss: (w.mse > 0.0).then(MseLoss::new),
            reg_loss: (w.reg > 0.0).then(|| ThresholdRegularizationLossConfig::new().init()),
            ssim_loss: (w.ssim > 0.0).then(|| SsimLossConfig::new().init()),
            cnt_loss: (w.cnt > 0.0).then(|| ContourLossConfig::new().init()),
            structure_loss: (w.structure > 0.0).then(|| StructureLossConfig::new().init()),
        }
    }
}

/// Multi-component pixel supervision loss summed across multi-scale prediction maps.
///
/// Every component is batch-averaged (`Reduction::Mean`). Predictions whose spatial size
/// differs from the targets are resized with nearest-neighbour interpolation, the only mode
/// whose backward pass every Burn backend implements.
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
    iou_loss: Option<IouLoss>,
    iou_patch_loss: Option<PatchIouLoss>,
    mae_loss: Option<MaeLoss>,
    mse_loss: Option<MseLoss>,
    reg_loss: Option<ThresholdRegularizationLoss>,
    ssim_loss: Option<SsimLoss>,
    cnt_loss: Option<ContourLoss>,
    structure_loss: Option<StructureLoss>,
}

/// Device-side result of [`PixLoss::compute`].
pub(crate) struct PixTerms {
    pub(crate) total: Tensor<1>,
    pub(crate) components: Vec<(Component, Tensor<1>)>,
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
        Ok(self
            .compute(scaled_preds, gt, pix_loss_lambda, false)?
            .total)
    }

    /// Computes total multi-scale pixel loss and reads per-component values in one transaction.
    ///
    /// # Errors
    ///
    /// Returns [`LossError`] if `scaled_preds` is empty, batch/channel dimensions mismatch `gt`,
    /// or the component values cannot be read back from the device.
    pub fn forward_detailed(
        &self,
        scaled_preds: &[Tensor<4>],
        gt: &Tensor<4>,
        pix_loss_lambda: f64,
    ) -> Result<(Tensor<1>, LossBreakdown), LossError> {
        let terms = self.compute(scaled_preds, gt, pix_loss_lambda, true)?;
        let breakdown = read_breakdown(terms.total.clone(), terms.components)?;
        Ok((terms.total, breakdown))
    }

    pub(crate) fn compute(
        &self,
        scaled_preds: &[Tensor<4>],
        gt: &Tensor<4>,
        pix_loss_lambda: f64,
        collect_components: bool,
    ) -> Result<PixTerms, LossError> {
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

        let [_, _, target_h, target_w] = gt_dims;
        let mut total_loss: Option<Tensor<1>> = None;
        let mut components: Vec<(Component, Tensor<1>)> = Vec::new();
        let mut add_component = |component: Component, raw: Tensor<1>, weight: f64| {
            let weighted = raw.mul_scalar(weight * pix_loss_lambda);
            if collect_components {
                match components.iter_mut().find(|(c, _)| *c == component) {
                    Some((_, acc)) => *acc = acc.clone() + weighted.clone(),
                    None => components.push((component, weighted.clone())),
                }
            }
            accumulate_tensor(&mut total_loss, weighted);
        };

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
            // Only probability-space components need the sigmoid; compute it at most once.
            let probs_cell = OnceCell::new();
            let probs = || {
                probs_cell
                    .get_or_init(|| activation::sigmoid(pred_logits.clone()))
                    .clone()
            };

            if self.bce_weight > 0.0 {
                let bce = bce_with_logits(pred_logits.clone(), gt.clone()).mean();
                add_component(Component::Bce, bce, self.bce_weight);
            }
            if let Some(ref iou_loss) = self.iou_loss {
                let iou = iou_loss.forward(probs(), gt.clone(), Reduction::Mean);
                add_component(Component::Iou, iou, self.iou_weight);
            }
            if let Some(ref iou_patch_loss) = self.iou_patch_loss {
                let patch = iou_patch_loss.forward(probs(), gt.clone(), Reduction::Mean);
                add_component(Component::IouPatch, patch, self.iou_patch_weight);
            }
            if let Some(ref mae_loss) = self.mae_loss {
                let mae = mae_loss.forward(probs(), gt.clone(), Reduction::Mean);
                add_component(Component::Mae, mae, self.mae_weight);
            }
            if let Some(ref mse_loss) = self.mse_loss {
                let mse = mse_loss.forward(probs(), gt.clone(), Reduction::Mean);
                add_component(Component::Mse, mse, self.mse_weight);
            }
            if let Some(ref reg_loss) = self.reg_loss {
                let reg = reg_loss.forward(probs(), Reduction::Mean);
                add_component(Component::Reg, reg, self.reg_weight);
            }
            if let Some(ref ssim_loss) = self.ssim_loss {
                let ssim = ssim_loss.forward(probs(), gt.clone(), Reduction::Mean);
                add_component(Component::Ssim, ssim, self.ssim_weight);
            }
            if let Some(ref cnt_loss) = self.cnt_loss {
                let cnt = cnt_loss.forward(probs(), gt.clone(), Reduction::Mean);
                add_component(Component::Cnt, cnt, self.cnt_weight);
            }
            if let Some(ref structure_loss) = self.structure_loss {
                let structure = structure_loss.forward(pred_logits, gt.clone(), Reduction::Mean);
                add_component(Component::Structure, structure, self.structure_weight);
            }
        }

        let total = total_loss.unwrap_or_else(|| Tensor::zeros([1], (&gt.device(), gt.dtype())));
        Ok(PixTerms { total, components })
    }
}
