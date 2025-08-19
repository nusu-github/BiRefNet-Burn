//! Integrated BiRefNet loss system combining multiple loss functions.
//!
//! This module provides a unified loss system specifically designed for BiRefNet training,
//! combining pixel-wise and classification losses with default weights optimized for
//! segmentation tasks.

use std::collections::HashMap;

use burn::{
    nn::loss::Reduction,
    prelude::*,
    tensor::{backend::Backend, cast::ToElement, Tensor},
};
use thiserror::Error;

use crate::{
    classification::{ClassificationLoss, ClassificationLossConfig},
    pixel::{LossWeightsConfig, PixLoss, PixLossConfig},
};

/// Errors that can occur during BiRefNet loss computation.
#[derive(Debug, Error)]
pub enum BiRefNetLossError {
    /// Empty predictions provided to loss function
    #[error("predictions cannot be empty - at least one prediction tensor is required")]
    EmptyPredictions,

    /// Invalid tensor dimensions
    #[error("tensor dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: String, actual: String },

    /// Invalid parameter values
    #[error("invalid parameter '{parameter}': {reason}")]
    InvalidParameter { parameter: String, reason: String },

    /// Numerical computation errors
    #[error("numerical computation failed: {reason}")]
    ComputationError { reason: String },

    /// Tensor shape incompatibility
    #[error("incompatible tensor shapes: predictions shape {pred_shape:?} does not match targets shape {target_shape:?}")]
    IncompatibleShapes {
        pred_shape: Vec<usize>,
        target_shape: Vec<usize>,
    },
}

/// Configuration for the integrated BiRefNet loss system.
///
/// This configuration combines pixel-level losses (BCE, IoU, SSIM, MAE, etc.)
/// with optional classification losses, using weights optimized for BiRefNet training.
#[derive(Config, Debug)]
pub struct BiRefNetLossConfig {
    /// Weights for pixel-level losses
    #[config(default = "default_pixel_weights()")]
    pub pixel_weights: LossWeightsConfig,

    /// Weight for classification loss (set to 0.0 to disable)
    #[config(default = "5.0")]
    pub classification_weight: f64,

    /// Global scaling factor for all losses
    #[config(default = "1.0")]
    pub global_scale: f64,
}

/// Default pixel loss weights optimized for BiRefNet training.
/// These values are derived from the original PyTorch implementation.
fn default_pixel_weights() -> LossWeightsConfig {
    LossWeightsConfig::new()
        .with_bce(30.0) // Binary Cross Entropy - primary pixel loss
        .with_iou(0.5) // Intersection over Union - shape awareness
        .with_iou_patch(0.0) // Patch IoU - disabled by default for performance
        .with_mae(100.0) // Mean Absolute Error - pixel accuracy
        .with_mse(0.0) // Mean Squared Error - disabled by default
        .with_triplet(0.0) // Triplet loss - disabled by default
        .with_reg(0.0) // Threshold regularization - disabled by default
        .with_ssim(10.0) // Structural Similarity - perceptual quality
        .with_cnt(0.0) // Contour loss - disabled by default
        .with_structure(0.0) // Structure loss - disabled by default
}

impl BiRefNetLossConfig {
    /// Create a configuration optimized for DIS (Dichotomous Image Segmentation) tasks.

    pub fn dis_optimized() -> Self {
        Self::new().with_pixel_weights(
            LossWeightsConfig::new()
                .with_bce(30.0)
                .with_iou(0.5)
                .with_mae(0.0) // Disabled for DIS
                .with_ssim(10.0),
        )
    }

    /// Create a configuration optimized for General segmentation tasks.

    pub fn general_optimized() -> Self {
        Self::new().with_pixel_weights(
            LossWeightsConfig::new()
                .with_bce(30.0)
                .with_iou(0.5)
                .with_mae(100.0)
                .with_ssim(10.0),
        )
    }

    /// Create a configuration optimized for Matting tasks.

    pub fn matting_optimized() -> Self {
        Self::new().with_pixel_weights(
            LossWeightsConfig::new()
                .with_bce(30.0)
                .with_iou(0.0) // Disabled for matting
                .with_mae(100.0)
                .with_ssim(10.0),
        )
    }

    /// Builder method to set BCE weight

    pub const fn with_bce_weight(mut self, weight: f64) -> Self {
        self.pixel_weights.bce = weight;
        self
    }

    /// Builder method to set IoU weight

    pub const fn with_iou_weight(mut self, weight: f64) -> Self {
        self.pixel_weights.iou = weight;
        self
    }

    /// Builder method to set MAE weight

    pub const fn with_mae_weight(mut self, weight: f64) -> Self {
        self.pixel_weights.mae = weight;
        self
    }

    /// Builder method to set SSIM weight

    pub const fn with_ssim_weight(mut self, weight: f64) -> Self {
        self.pixel_weights.ssim = weight;
        self
    }

    /// Builder method to disable classification loss

    pub const fn without_classification(mut self) -> Self {
        self.classification_weight = 0.0;
        self
    }
}

/// Integrated BiRefNet loss system.
///
/// This loss system combines multiple loss functions optimized for segmentation tasks,
/// providing both pixel-level and optional classification losses with configurable weights.
#[derive(Module, Debug)]
pub struct BiRefNetLoss<B: Backend> {
    /// Pixel-level loss component
    pixel_loss: PixLoss<B>,

    /// Classification loss component (optional)
    classification_loss: Option<ClassificationLoss<B>>,

    /// Classification weight
    classification_weight: f64,

    /// Global scaling factor
    global_scale: f64,
}

impl BiRefNetLossConfig {
    /// Initialize the BiRefNet loss system with the given configuration.
    pub fn init<B: Backend>(&self, device: &B::Device) -> BiRefNetLoss<B> {
        let pixel_config = PixLossConfig::new(self.pixel_weights.clone());
        let pixel_loss = pixel_config.init(device);

        let classification_loss = (self.classification_weight > 0.0)
            .then(|| ClassificationLossConfig::new().init(device));

        BiRefNetLoss {
            pixel_loss,
            classification_loss,
            classification_weight: self.classification_weight,
            global_scale: self.global_scale,
        }
    }
}

impl<B: Backend> BiRefNetLoss<B> {
    /// Create a new BiRefNet loss with default configuration.

    pub fn new(config: BiRefNetLossConfig) -> Self {
        let device = &B::Device::default();
        config.init(device)
    }

    /// Compute the forward pass of the integrated loss system.
    ///
    /// # Arguments
    /// * `scaled_preds` - Multi-scale predictions from the model
    /// * `targets` - Ground truth segmentation maps
    ///
    /// # Returns
    /// Total loss (pixel-level only, for backward compatibility)
    ///
    /// # Errors
    /// Returns `BiRefNetLossError::EmptyPredictions` if no predictions are provided.
    pub fn forward(
        &self,
        scaled_preds: Vec<Tensor<B, 4>>,
        targets: Tensor<B, 4>,
    ) -> Result<Tensor<B, 1>, BiRefNetLossError> {
        self.forward_with_classification(scaled_preds, targets, None, None)
    }

    /// Compute the forward pass with optional classification loss.
    ///
    /// # Arguments
    /// * `scaled_preds` - Multi-scale predictions from the model
    /// * `targets` - Ground truth segmentation maps
    /// * `class_preds` - Optional classification predictions `[Tensor<B, 2>]`
    /// * `class_targets` - Optional classification targets `Tensor<B, 1, Int>`
    ///
    /// # Returns
    /// Total loss combining pixel and classification components
    ///
    /// # Errors
    /// Returns `BiRefNetLossError::EmptyPredictions` if no predictions are provided.
    pub fn forward_with_classification(
        &self,
        scaled_preds: Vec<Tensor<B, 4>>,
        targets: Tensor<B, 4>,
        class_preds: Option<&[Tensor<B, 2>]>,
        class_targets: Option<&Tensor<B, 1, Int>>,
    ) -> Result<Tensor<B, 1>, BiRefNetLossError> {
        if scaled_preds.is_empty() {
            return Err(BiRefNetLossError::EmptyPredictions);
        }

        // Convert targets to Int for pixel loss computation
        let targets_int = targets.int();

        // Compute pixel-level loss
        let (pixel_loss, _loss_dict) =
            self.pixel_loss
                .forward(scaled_preds, targets_int, self.global_scale);

        // Total loss starts with pixel loss
        let mut total_loss = pixel_loss;

        // Add classification loss if enabled and predictions are provided
        if let Some(ref cls_loss) = self.classification_loss {
            if let (Some(class_preds), Some(class_targets)) = (class_preds, class_targets) {
                let cls_loss_value = cls_loss
                    .forward(class_preds, class_targets, Reduction::Mean)
                    .mul_scalar(self.classification_weight)
                    .mul_scalar(self.global_scale);

                total_loss = total_loss + cls_loss_value;
            }
        }

        Ok(total_loss)
    }

    /// Compute forward pass with detailed loss breakdown.
    ///
    /// # Arguments
    /// * `scaled_preds` - Multi-scale predictions from the model
    /// * `targets` - Ground truth segmentation maps
    ///
    /// # Returns
    /// Tuple of (total_loss, detailed_loss_dict) for monitoring training progress
    ///
    /// # Errors
    /// Returns `BiRefNetLossError::EmptyPredictions` if no predictions are provided.
    pub fn forward_detailed(
        &self,
        scaled_preds: Vec<Tensor<B, 4>>,
        targets: Tensor<B, 4>,
    ) -> Result<(Tensor<B, 1>, HashMap<String, f64>), BiRefNetLossError> {
        self.forward_detailed_with_classification(scaled_preds, targets, None, None)
    }

    /// Compute forward pass with detailed loss breakdown and optional classification.
    ///
    /// # Arguments
    /// * `scaled_preds` - Multi-scale predictions from the model
    /// * `targets` - Ground truth segmentation maps
    /// * `class_preds` - Optional classification predictions `[Tensor<B, 2>]`
    /// * `class_targets` - Optional classification targets `Tensor<B, 1, Int>`
    ///
    /// # Returns
    /// Tuple of (total_loss, detailed_loss_dict) for monitoring training progress
    ///
    /// # Errors
    /// Returns `BiRefNetLossError::EmptyPredictions` if no predictions are provided.
    pub fn forward_detailed_with_classification(
        &self,
        scaled_preds: Vec<Tensor<B, 4>>,
        targets: Tensor<B, 4>,
        class_preds: Option<&[Tensor<B, 2>]>,
        class_targets: Option<&Tensor<B, 1, Int>>,
    ) -> Result<(Tensor<B, 1>, HashMap<String, f64>), BiRefNetLossError> {
        if scaled_preds.is_empty() {
            return Err(BiRefNetLossError::EmptyPredictions);
        }

        // Convert targets to Int for pixel loss computation
        let targets_int = targets.int();

        // Compute pixel-level loss with detailed breakdown
        let (pixel_loss, mut loss_dict) =
            self.pixel_loss
                .forward(scaled_preds, targets_int, self.global_scale);

        let mut total_loss = pixel_loss;

        // Add classification loss if enabled and predictions are provided
        if let Some(ref cls_loss) = self.classification_loss {
            if let (Some(class_preds), Some(class_targets)) = (class_preds, class_targets) {
                let cls_loss_value = cls_loss
                    .forward(class_preds, class_targets, Reduction::Mean)
                    .mul_scalar(self.classification_weight)
                    .mul_scalar(self.global_scale);

                total_loss = total_loss.clone() + cls_loss_value.clone();

                // Add classification loss to dictionary
                loss_dict.insert(
                    "classification".to_owned(),
                    cls_loss_value.into_scalar().to_f64(),
                );
            }
        }

        // Add total loss to dictionary
        loss_dict.insert(
            "total".to_owned(),
            total_loss.clone().into_scalar().to_f64(),
        );

        Ok((total_loss, loss_dict))
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        backend::NdArray,
        tensor::{Distribution, Tensor},
    };
    pub type TestBackend = NdArray<f32>;
    use super::*;

    #[test]
    fn birefnet_loss_config_creates_instance_with_positive_global_scale() {
        let config = BiRefNetLossConfig::new();
        let device = Default::default();
        let loss = config.init::<TestBackend>(&device);

        // Should create loss without errors
        assert!(loss.global_scale > 0.0);
    }

    #[test]
    fn birefnet_loss_forward_computes_finite_total_loss() {
        let config = BiRefNetLossConfig::new();
        let device = Default::default();
        let loss = config.init::<TestBackend>(&device);

        // Create test tensors with appropriate ranges
        // Use reasonable logit values (sigmoid will map these to [0,1])
        let pred = Tensor::<TestBackend, 4>::random(
            [2, 1, 64, 64],
            Distribution::Normal(0.0, 2.0), // Logits range: mostly [-6, 6]
            &device,
        );
        let target = Tensor::<TestBackend, 4>::random(
            [2, 1, 64, 64],
            Distribution::Uniform(0.0, 1.0), // Random values 0-1
            &device,
        )
        .round(); // Round to 0.0 or 1.0 (binary targets)

        let scaled_preds = vec![pred];
        let result = loss.forward(scaled_preds, target);

        assert!(result.is_ok());
        let total_loss = result.unwrap();
        assert!(total_loss.into_scalar().to_f64().is_finite());
    }

    #[test]
    fn birefnet_loss_forward_detailed_returns_total_and_component_losses() {
        let config = BiRefNetLossConfig::new();
        let device = Default::default();
        let loss = config.init::<TestBackend>(&device);

        // Create test tensors with appropriate ranges
        let pred = Tensor::<TestBackend, 4>::random(
            [2, 1, 64, 64],
            Distribution::Normal(0.0, 2.0), // Reasonable logits
            &device,
        );
        let target = Tensor::<TestBackend, 4>::random(
            [2, 1, 64, 64],
            Distribution::Uniform(0.0, 1.0), // Random values 0-1
            &device,
        )
        .round(); // Round to 0.0 or 1.0 (binary targets)

        let scaled_preds = vec![pred];
        let result = loss.forward_detailed(scaled_preds, target);

        assert!(result.is_ok());
        let (_total_loss, loss_dict) = result.unwrap();

        // Should contain individual loss components
        assert!(loss_dict.contains_key("total"));
        if config.pixel_weights.bce > 0.0 {
            assert!(loss_dict.contains_key("bce"));
        }
    }

    #[test]
    fn birefnet_loss_preset_configs_apply_correct_weights() {
        let dis_config = BiRefNetLossConfig::dis_optimized();
        let general_config = BiRefNetLossConfig::general_optimized();
        let matting_config = BiRefNetLossConfig::matting_optimized();

        // DIS should have MAE disabled
        assert_eq!(dis_config.pixel_weights.mae, 0.0);

        // General should have MAE enabled
        assert!(general_config.pixel_weights.mae > 0.0);

        // Matting should have IoU disabled
        assert_eq!(matting_config.pixel_weights.iou, 0.0);
    }

    #[test]
    fn birefnet_loss_forward_with_empty_predictions_returns_error() {
        let config = BiRefNetLossConfig::new();
        let device = Default::default();
        let loss = config.init::<TestBackend>(&device);

        let target = Tensor::<TestBackend, 4>::zeros([2, 1, 64, 64], &device);
        let scaled_preds: Vec<Tensor<TestBackend, 4>> = vec![];

        let result = loss.forward(scaled_preds, target);
        assert!(result.is_err());

        // Verify it's the correct error variant
        match result.unwrap_err() {
            BiRefNetLossError::EmptyPredictions => (),
            other => panic!("Expected EmptyPredictions error, got: {:?}", other),
        }
    }

    #[test]
    fn birefnet_loss_forward_with_classification_computes_combined_loss() {
        let config = BiRefNetLossConfig::new().with_classification_weight(5.0);
        let device = Default::default();
        let loss = config.init::<TestBackend>(&device);

        // Create segmentation data
        let pred = Tensor::<TestBackend, 4>::random(
            [2, 1, 64, 64],
            Distribution::Normal(0.0, 2.0),
            &device,
        );
        let target = Tensor::<TestBackend, 4>::random(
            [2, 1, 64, 64],
            Distribution::Uniform(0.0, 1.0),
            &device,
        )
        .round();

        // Create classification data
        let class_pred1 = Tensor::<TestBackend, 2>::random(
            [2, 10], // batch_size=2, num_classes=10
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let class_pred2 =
            Tensor::<TestBackend, 2>::random([2, 10], Distribution::Normal(0.0, 1.0), &device);
        let class_preds = vec![class_pred1, class_pred2];
        let class_targets =
            Tensor::<TestBackend, 1>::random([2], Distribution::Uniform(0.0, 10.0), &device).int();

        let scaled_preds = vec![pred];
        let result = loss.forward_with_classification(
            scaled_preds,
            target,
            Some(&class_preds),
            Some(&class_targets),
        );

        assert!(result.is_ok());
        let total_loss = result.unwrap();
        assert!(total_loss.into_scalar().to_f64().is_finite());
    }

    #[test]
    fn birefnet_loss_forward_detailed_with_classification_returns_loss_dict() {
        let config = BiRefNetLossConfig::new().with_classification_weight(3.0);
        let device = Default::default();
        let loss = config.init::<TestBackend>(&device);

        // Create data
        let pred = Tensor::<TestBackend, 4>::random(
            [2, 1, 32, 32],
            Distribution::Normal(0.0, 2.0),
            &device,
        );
        let target = Tensor::<TestBackend, 4>::random(
            [2, 1, 32, 32],
            Distribution::Uniform(0.0, 1.0),
            &device,
        )
        .round();

        let class_pred =
            Tensor::<TestBackend, 2>::random([2, 5], Distribution::Normal(0.0, 1.0), &device);
        let class_preds = vec![class_pred];
        let class_targets =
            Tensor::<TestBackend, 1>::random([2], Distribution::Uniform(0.0, 5.0), &device).int();

        let scaled_preds = vec![pred];
        let result = loss.forward_detailed_with_classification(
            scaled_preds,
            target,
            Some(&class_preds),
            Some(&class_targets),
        );

        assert!(result.is_ok());
        let (total_loss, loss_dict) = result.unwrap();

        // Check that all expected keys are present
        assert!(loss_dict.contains_key("total"));
        assert!(loss_dict.contains_key("classification"));
        assert!(total_loss.into_scalar().to_f64().is_finite());
    }

    #[test]
    fn birefnet_loss_without_classification_weight_ignores_classification() {
        let config = BiRefNetLossConfig::new().without_classification();
        let device = Default::default();
        let loss = config.init::<TestBackend>(&device);

        // Create data
        let pred = Tensor::<TestBackend, 4>::random(
            [2, 1, 32, 32],
            Distribution::Normal(0.0, 2.0),
            &device,
        );
        let target = Tensor::<TestBackend, 4>::random(
            [2, 1, 32, 32],
            Distribution::Uniform(0.0, 1.0),
            &device,
        )
        .round();

        let class_pred =
            Tensor::<TestBackend, 2>::random([2, 5], Distribution::Normal(0.0, 1.0), &device);
        let class_preds = vec![class_pred];
        let class_targets =
            Tensor::<TestBackend, 1>::random([2], Distribution::Uniform(0.0, 5.0), &device).int();

        let scaled_preds = vec![pred];

        // Both calls should return same result (classification should be ignored)
        let result1 = loss.forward(scaled_preds.clone(), target.clone());
        let result2 = loss.forward_with_classification(
            scaled_preds,
            target,
            Some(&class_preds),
            Some(&class_targets),
        );

        assert!(result1.is_ok());
        assert!(result2.is_ok());

        let loss1 = result1.unwrap().into_scalar().to_f64();
        let loss2 = result2.unwrap().into_scalar().to_f64();

        // Should be very close (considering floating point precision)
        assert!((loss1 - loss2).abs() < 1e-6);
    }
}
