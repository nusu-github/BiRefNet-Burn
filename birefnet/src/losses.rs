//! Loss functions for BiRefNet training.
//!
//! This module implements the loss functions used in BiRefNet training,
//! including Binary Cross-Entropy (BCE) loss and Intersection over Union (IoU) loss.
//!
//! The implementation follows the original PyTorch BiRefNet loss.py structure.

use burn::prelude::*;
use burn::tensor::{backend::Backend, Tensor};

/// Configuration for Combined Loss function.
#[derive(Config, Debug)]
pub struct CombinedLossConfig {
    #[config(default = 1.0)]
    pub bce_weight: f32,
    #[config(default = 1.0)]
    pub iou_weight: f32,
    #[config(default = 1e-6)]
    pub epsilon: f32,
}

/// Combined loss function for BiRefNet training.
///
/// This combines BCE loss and IoU loss as used in the original implementation.
#[derive(Module, Debug)]
pub struct CombinedLoss<B: Backend> {
    pub bce_weight: f32,
    pub iou_weight: f32,
    pub epsilon: f32,
    _phantom: std::marker::PhantomData<B>,
}

impl CombinedLossConfig {
    /// Initialize a new combined loss function with the given configuration.
    pub const fn init<B: Backend>(&self) -> CombinedLoss<B> {
        CombinedLoss {
            bce_weight: self.bce_weight,
            iou_weight: self.iou_weight,
            epsilon: self.epsilon,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B: Backend> Default for CombinedLoss<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> CombinedLoss<B> {
    /// Create a new combined loss function with default configuration.
    pub fn new() -> Self {
        CombinedLossConfig::new().init()
    }

    /// Create a new combined loss function with custom weights.
    pub fn with_weights(bce_weight: f32, iou_weight: f32) -> Self {
        CombinedLossConfig::new()
            .with_bce_weight(bce_weight)
            .with_iou_weight(iou_weight)
            .init()
    }

    /// Calculate the combined loss for binary segmentation.
    ///
    /// # Arguments
    /// * `pred` - Predicted segmentation map with shape [N, C, H, W]
    /// * `target` - Ground truth segmentation map with shape [N, C, H, W]
    ///
    /// # Returns
    /// Combined loss tensor
    pub fn forward(&self, pred: Tensor<B, 4>, target: Tensor<B, 4>) -> Tensor<B, 1> {
        let bce_loss_value = self.bce_loss(pred.clone(), target.clone());
        let iou_loss_value = self.iou_loss(pred, target);

        bce_loss_value * self.bce_weight + iou_loss_value * self.iou_weight
    }

    /// Calculate Binary Cross-Entropy loss with logits.
    ///
    /// # Arguments
    /// * `pred` - Predicted logits with shape [N, C, H, W]
    /// * `target` - Ground truth binary masks with shape [N, C, H, W]
    ///
    /// # Returns
    /// BCE loss tensor
    pub fn bce_loss(&self, pred: Tensor<B, 4>, target: Tensor<B, 4>) -> Tensor<B, 1> {
        // Manual implementation of binary cross entropy with logits
        // BCE = -[y*log(sigmoid(x)) + (1-y)*log(1-sigmoid(x))]
        // Using the numerically stable formulation: BCE = max(x, 0) - x*y + log(1 + exp(-abs(x)))

        let max_val = pred.clone().clamp_max(0.0);
        let term1 = max_val - pred.clone() * target;
        let term2 = (-pred.abs()).exp().add_scalar(1.0).log();

        (term1 + term2).mean()
    }

    /// Calculate IoU (Intersection over Union) loss.
    ///
    /// This implements the IoU loss as:
    /// IoU = (intersection + epsilon) / (union + epsilon)
    /// Loss = 1 - IoU
    ///
    /// # Arguments
    /// * `pred` - Predicted logits with shape [N, C, H, W]
    /// * `target` - Ground truth binary masks with shape [N, C, H, W]
    ///
    /// # Returns
    /// IoU loss tensor
    pub fn iou_loss(&self, pred: Tensor<B, 4>, target: Tensor<B, 4>) -> Tensor<B, 1> {
        // Apply sigmoid to get probabilities
        let pred_sigmoid = burn::tensor::activation::sigmoid(pred);

        // Calculate intersection: sum of element-wise multiplication
        let intersection = (pred_sigmoid.clone() * target.clone()).sum();

        // Calculate union: sum of both tensors minus intersection
        let union = pred_sigmoid.sum() + target.sum() - intersection.clone();

        // Calculate IoU with epsilon for numerical stability
        let iou = (intersection + self.epsilon) / (union + self.epsilon);

        // Return IoU loss (1 - IoU)
        let one = Tensor::ones_like(&iou);
        one - iou
    }
}

/// Configuration for Multi-scale Loss function.
#[derive(Config, Debug)]
pub struct MultiScaleLossConfig {
    #[config(default = 1.0)]
    pub bce_weight: f32,
    #[config(default = 1.0)]
    pub iou_weight: f32,
    #[config(default = "vec![1.0, 0.8, 0.6, 0.4]")]
    pub scale_weights: Vec<f32>,
    #[config(default = 1e-6)]
    pub epsilon: f32,
}

/// Multi-scale loss function for hierarchical supervision.
///
/// This applies the combined loss at multiple scales as used in the original implementation.
#[derive(Module, Debug)]
pub struct MultiScaleLoss<B: Backend> {
    pub base_loss: CombinedLoss<B>,
    pub scale_weights: Vec<f32>,
}

impl MultiScaleLossConfig {
    /// Initialize a new multi-scale loss function with the given configuration.
    pub fn init<B: Backend>(&self) -> MultiScaleLoss<B> {
        MultiScaleLoss {
            base_loss: CombinedLossConfig::new()
                .with_bce_weight(self.bce_weight)
                .with_iou_weight(self.iou_weight)
                .with_epsilon(self.epsilon)
                .init(),
            scale_weights: self.scale_weights.clone(),
        }
    }
}

impl<B: Backend> Default for MultiScaleLoss<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> MultiScaleLoss<B> {
    /// Create a new multi-scale loss function with default configuration.
    pub fn new() -> Self {
        MultiScaleLossConfig::new().init()
    }

    /// Create a new multi-scale loss function with custom weights.
    pub fn with_weights(bce_weight: f32, iou_weight: f32, scale_weights: Vec<f32>) -> Self {
        MultiScaleLossConfig::new()
            .with_bce_weight(bce_weight)
            .with_iou_weight(iou_weight)
            .with_scale_weights(scale_weights)
            .init()
    }

    /// Calculate multi-scale loss.
    ///
    /// # Arguments
    /// * `preds` - List of predicted segmentation maps at different scales
    /// * `targets` - List of ground truth segmentation maps at different scales
    ///
    /// # Returns
    /// Combined multi-scale loss tensor
    pub fn forward(&self, preds: Vec<Tensor<B, 4>>, targets: Vec<Tensor<B, 4>>) -> Tensor<B, 1> {
        assert_eq!(preds.len(), targets.len());
        assert_eq!(preds.len(), self.scale_weights.len());

        let mut total_loss = None;

        for ((pred, target), &weight) in preds
            .iter()
            .zip(targets.iter())
            .zip(self.scale_weights.iter())
        {
            let scale_loss = self.base_loss.forward(pred.clone(), target.clone());
            let weighted_loss = scale_loss * weight;
            total_loss = match total_loss {
                Some(acc) => Some(acc + weighted_loss),
                None => Some(weighted_loss),
            };
        }

        total_loss.unwrap_or_else(|| panic!("No predictions provided"))
    }
}

/// Configuration for Structure Loss function.
#[derive(Config, Debug)]
pub struct StructureLossConfig {
    #[config(default = 1.0)]
    pub weight: f32,
}

/// Structure loss implementation for edge-aware training.
///
/// This is a placeholder for the structure loss used in the original implementation.
/// The full implementation would include SSIM and other structural similarities.
#[derive(Module, Debug)]
pub struct StructureLoss<B: Backend> {
    pub weight: f32,
    _phantom: std::marker::PhantomData<B>,
}

impl StructureLossConfig {
    /// Initialize a new structure loss function with the given configuration.
    pub const fn init<B: Backend>(&self) -> StructureLoss<B> {
        StructureLoss {
            weight: self.weight,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B: Backend> Default for StructureLoss<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> StructureLoss<B> {
    /// Create a new structure loss function with default configuration.
    pub fn new() -> Self {
        StructureLossConfig::new().init()
    }

    /// Create a new structure loss function with custom weight.
    pub fn with_weight(weight: f32) -> Self {
        StructureLossConfig::new().with_weight(weight).init()
    }

    /// Calculate structure loss (simplified version).
    ///
    /// # Arguments
    /// * `pred` - Predicted segmentation map with shape [N, C, H, W]
    /// * `target` - Ground truth segmentation map with shape [N, C, H, W]
    ///
    /// # Returns
    /// Structure loss tensor
    pub fn forward(&self, pred: Tensor<B, 4>, target: Tensor<B, 4>) -> Tensor<B, 1> {
        // For now, return a simple L1 loss as a placeholder
        // This should be replaced with proper SSIM or other structural loss
        let l1_loss = (pred - target).abs().mean();
        l1_loss * self.weight
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_combined_loss_config() {
        let config = CombinedLossConfig::new()
            .with_bce_weight(2.0)
            .with_iou_weight(0.5)
            .with_epsilon(1e-8);
        assert_eq!(config.bce_weight, 2.0);
        assert_eq!(config.iou_weight, 0.5);
        assert_eq!(config.epsilon, 1e-8);
    }

    #[test]
    fn test_combined_loss() {
        let loss_fn = CombinedLoss::<TestBackend>::new();

        let pred = Tensor::random(
            [1, 1, 4, 4],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &Default::default(),
        );
        let target = Tensor::random(
            [1, 1, 4, 4],
            burn::tensor::Distribution::Uniform(0.0, 1.0),
            &Default::default(),
        );

        let loss = loss_fn.forward(pred, target);

        // Check that loss is a scalar tensor
        assert_eq!(loss.shape().dims, [1]);
    }

    #[test]
    fn test_bce_loss() {
        let loss_fn = CombinedLoss::<TestBackend>::with_weights(1.0, 0.0);

        let pred = Tensor::random(
            [1, 1, 4, 4],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &Default::default(),
        );
        let target = Tensor::random(
            [1, 1, 4, 4],
            burn::tensor::Distribution::Uniform(0.0, 1.0),
            &Default::default(),
        );

        let loss = loss_fn.bce_loss(pred, target);

        // Check that loss is a scalar tensor
        assert_eq!(loss.shape().dims, [1]);
    }

    #[test]
    fn test_iou_loss() {
        let loss_fn = CombinedLoss::<TestBackend>::with_weights(0.0, 1.0);

        let pred = Tensor::random(
            [1, 1, 4, 4],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &Default::default(),
        );
        let target = Tensor::random(
            [1, 1, 4, 4],
            burn::tensor::Distribution::Uniform(0.0, 1.0),
            &Default::default(),
        );

        let loss = loss_fn.iou_loss(pred, target);

        // Check that loss is a scalar tensor
        assert_eq!(loss.shape().dims, [1]);
    }

    #[test]
    fn test_multi_scale_loss_config() {
        let config = MultiScaleLossConfig::new()
            .with_bce_weight(1.5)
            .with_iou_weight(0.5)
            .with_scale_weights(vec![1.0, 0.8, 0.6]);
        assert_eq!(config.bce_weight, 1.5);
        assert_eq!(config.iou_weight, 0.5);
        assert_eq!(config.scale_weights, vec![1.0, 0.8, 0.6]);
    }

    #[test]
    fn test_multi_scale_loss() {
        let loss_fn = MultiScaleLoss::<TestBackend>::new();

        let preds = vec![
            Tensor::random(
                [1, 1, 4, 4],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &Default::default(),
            ),
            Tensor::random(
                [1, 1, 4, 4],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &Default::default(),
            ),
        ];
        let targets = vec![
            Tensor::random(
                [1, 1, 4, 4],
                burn::tensor::Distribution::Uniform(0.0, 1.0),
                &Default::default(),
            ),
            Tensor::random(
                [1, 1, 4, 4],
                burn::tensor::Distribution::Uniform(0.0, 1.0),
                &Default::default(),
            ),
        ];

        let loss_fn = MultiScaleLoss::<TestBackend>::with_weights(1.0, 1.0, vec![1.0, 0.8]);
        let loss = loss_fn.forward(preds, targets);

        // Check that loss is a scalar tensor
        assert_eq!(loss.shape().dims, [1]);
    }

    #[test]
    fn test_structure_loss_config() {
        let config = StructureLossConfig::new().with_weight(2.0);
        assert_eq!(config.weight, 2.0);
    }

    #[test]
    fn test_structure_loss() {
        let loss_fn = StructureLoss::<TestBackend>::with_weight(1.5);

        let pred = Tensor::random(
            [1, 1, 4, 4],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &Default::default(),
        );
        let target = Tensor::random(
            [1, 1, 4, 4],
            burn::tensor::Distribution::Uniform(0.0, 1.0),
            &Default::default(),
        );

        let loss = loss_fn.forward(pred, target);

        // Check that loss is a scalar tensor
        assert_eq!(loss.shape().dims, [1]);
    }
}
