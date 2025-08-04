//! BiRefNet training loss system.
//!
//! This module provides the complete BiRefNet training loss system that integrates
//! pixel loss, classification loss, and guidance distillation loss as used
//! in the original BiRefNet training implementation.
//!
//! This represents Level 3 of the hierarchical loss architecture:
//! - Level 1: Basic segmentation losses (combined.rs)
//! - Level 2: Multi-loss pixel system (pixel.rs)
//! - Level 3: Complete BiRefNet training system (this module)

use std::collections::HashMap;

use super::{classification::ClsLoss, pixel::PixLoss, BCELoss, BCELossConfig, PixLossConfig};
use burn::tensor::cast::ToElement;
use burn::{
    prelude::*,
    tensor::{
        backend::Backend,
        module::interpolate,
        ops::{InterpolateMode, InterpolateOptions},
        ElementConversion, Tensor,
    },
};

/// Configuration for the BiRefNet training loss system.
#[derive(Config, Debug)]
pub struct BiRefNetLossConfig {
    #[config(default = 1.0)]
    pub gdt_loss_weight: f32,
    #[config(default = true)]
    pub enable_out_ref: bool,
    pub pix_loss_config: PixLossConfig,
}

#[derive(Module, Debug)]
pub struct BiRefNetLoss<B: Backend> {
    pub gdt_loss_weight: f32,
    pub enable_out_ref: bool,

    pub pixel_loss: PixLoss<B>,
    pub cls_loss: ClsLoss<B>,
    pub gdt_loss: Option<BCELoss<B>>,
}

impl BiRefNetLossConfig {
    /// Initialize a new BiRefNet training loss system with the given configuration.
    pub fn init<B: Backend>(&self, device: &B::Device) -> BiRefNetLoss<B> {
        // Initialize pixel loss based on task
        BiRefNetLoss {
            gdt_loss_weight: self.gdt_loss_weight,
            enable_out_ref: self.enable_out_ref,

            pixel_loss: self.pix_loss_config.init(device),
            cls_loss: ClsLoss::new(device),
            gdt_loss: if self.enable_out_ref {
                Some(BCELossConfig::new().init())
            } else {
                None
            },
        }
    }
}

impl<B: Backend> BiRefNetLoss<B> {
    /// Calculate the total training loss.
    ///
    /// # Arguments
    /// * `scaled_preds` - Multi-scale predictions from the model
    /// * `class_preds` - Classification predictions (optional)
    /// * `gdt_outputs` - Guidance distillation outputs (optional)
    /// * `gt` - Ground truth segmentation map
    /// * `class_labels` - Ground truth class labels
    /// * `pix_loss_lambda` - Global scaling factor for pixel loss
    ///
    /// # Returns
    /// A tuple of (total_loss, detailed_loss_dict)
    pub fn forward(
        &self,
        scaled_preds: Vec<Tensor<B, 4>>,
        class_preds: Option<Vec<Option<Tensor<B, 2>>>>,
        gdt_outputs: Option<(Vec<Tensor<B, 4>>, Vec<Tensor<B, 4>>)>,
        gt: Tensor<B, 4>,
        class_labels: Option<Tensor<B, 1, Int>>,
        pix_loss_lambda: f32,
    ) -> (Tensor<B, 1>, HashMap<String, f64>) {
        let device = gt.device();
        let mut loss_dict = HashMap::new();
        let mut total_loss = Tensor::zeros([1], &device);

        // 1. Calculate pixel loss
        let (pixel_loss, pixel_loss_dict) =
            self.pixel_loss.forward(scaled_preds, gt, pix_loss_lambda);
        total_loss = total_loss + pixel_loss.clone();
        loss_dict.insert("loss_pix".to_string(), pixel_loss.into_scalar().to_f64());

        // Merge individual pixel loss components
        for (key, value) in pixel_loss_dict {
            loss_dict.insert(key, value);
        }

        // 2. Calculate classification loss
        if let (Some(class_preds), Some(class_labels)) = (class_preds, class_labels) {
            // Filter out None predictions
            let valid_preds: Vec<Tensor<B, 2>> = class_preds.into_iter().flatten().collect();

            if !valid_preds.is_empty() {
                let cls_loss = self.cls_loss.forward(valid_preds, class_labels);
                total_loss = total_loss + cls_loss.clone();
                loss_dict.insert("loss_cls".to_string(), cls_loss.into_scalar().to_f64());
            } else {
                loss_dict.insert("loss_cls".to_string(), 0.0);
            }
        } else {
            loss_dict.insert("loss_cls".to_string(), 0.0);
        }

        // 3. Calculate guidance distillation loss (reference loss)
        if let (Some(ref gdt_loss), Some((gdt_preds, gdt_labels))) = (&self.gdt_loss, gdt_outputs) {
            let mut total_gdt_loss = Tensor::zeros([1], &device);

            for (gdt_pred, gdt_label) in gdt_preds.iter().zip(gdt_labels.iter()) {
                // Resize gdt_pred to match gdt_label if necessary
                let gdt_pred_resized = if gdt_pred.dims() != gdt_label.dims() {
                    let [_, _, h, w] = gdt_label.dims();
                    let options = InterpolateOptions::new(InterpolateMode::Bilinear);
                    interpolate(gdt_pred.clone(), [h, w], options)
                } else {
                    gdt_pred.clone()
                };

                // Apply sigmoid to both prediction and label
                let gdt_pred_sigmoid = burn::tensor::activation::sigmoid(gdt_pred_resized);
                let gdt_label_sigmoid = burn::tensor::activation::sigmoid(gdt_label.clone());

                // Calculate BCE loss
                let gdt_loss_val = gdt_loss.forward(gdt_pred_sigmoid, gdt_label_sigmoid);
                total_gdt_loss = total_gdt_loss + gdt_loss_val;
            }

            // Apply weight to total GDT loss
            total_gdt_loss = total_gdt_loss * self.gdt_loss_weight.elem::<B::FloatElem>();
            total_loss = total_loss + total_gdt_loss.clone();
            loss_dict.insert(
                "loss_gdt".to_string(),
                total_gdt_loss.into_scalar().to_f64(),
            );
        } else {
            loss_dict.insert("loss_gdt".to_string(), 0.0);
        }

        (total_loss, loss_dict)
    }

    /// TODO: Calculate the total training loss with simplified interface.
    /// This should include:
    /// - Multi-scale pixel losses (BCE, IoU, SSIM, MAE)
    /// - Classification loss for auxiliary outputs
    /// - Gradient descent loss (GDT) for refinement
    /// - Proper loss weighting according to BiRefNet paper
    ///
    /// This is a simplified version of the forward method for easier use.
    ///
    /// # Arguments
    /// * `scaled_preds` - Multi-scale predictions from the model
    /// * `gt` - Ground truth segmentation map
    ///
    /// # Returns
    /// Total loss tensor
    pub fn forward_simple(
        &self,
        scaled_preds: Vec<Tensor<B, 4>>,
        gt: Tensor<B, 4>,
    ) -> Tensor<B, 1> {
        let (loss, _) = self.forward(
            scaled_preds,
            None, // No classification predictions
            None, // No GDT outputs
            gt,
            None,    // No class labels
            1.0_f32, // Default pixel loss lambda
        );
        loss
    }

    /// Get current loss configuration as a HashMap for logging.
    pub fn get_loss_config(&self) -> HashMap<String, f32> {
        let mut config = HashMap::new();
        let weights = self.pixel_loss.loss_weights();
        config.insert("bce".to_string(), weights.bce);
        config.insert("iou".to_string(), weights.iou);
        config.insert("iou_patch".to_string(), weights.iou_patch);
        config.insert("mae".to_string(), weights.mae);
        config.insert("mse".to_string(), weights.mse);
        config.insert("triplet".to_string(), weights.triplet);
        config.insert("reg".to_string(), weights.reg);
        config.insert("ssim".to_string(), weights.ssim);
        config.insert("cnt".to_string(), weights.cnt);
        config.insert("structure".to_string(), weights.structure);
        config.insert("gdt_weight".to_string(), self.gdt_loss_weight);
        config
    }
}
