//! Loss functions for BiRefNet training.
//!
//! This module implements the loss functions used in BiRefNet training,
//! including Binary Cross-Entropy (BCE) loss and Intersection over Union (IoU) loss.
//!
//! The implementation follows the original PyTorch BiRefNet loss.py structure.

use burn::{
    module::Param,
    nn::{
        conv::Conv2dConfig,
        loss::{CrossEntropyLoss, CrossEntropyLossConfig},
        pool::AvgPool2dConfig,
        PaddingConfig2d, Unfold4d, Unfold4dConfig,
    },
    prelude::*,
    tensor::{backend::Backend, Tensor},
};

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
        assert_eq!(
            preds.len(),
            targets.len(),
            "Number of predictions and targets must be equal."
        );
        assert_eq!(
            preds.len(),
            self.scale_weights.len(),
            "Number of predictions and scale_weights must be equal."
        );

        let total_loss = preds
            .iter()
            .zip(targets.iter())
            .zip(self.scale_weights.iter())
            .fold(None, |acc, ((pred, target), &weight)| {
                let scale_loss = self.base_loss.forward(pred.clone(), target.clone());
                let weighted_loss = scale_loss * weight;
                match acc {
                    Some(total) => Some(total + weighted_loss),
                    None => Some(weighted_loss),
                }
            });

        total_loss.unwrap_or_else(|| Tensor::zeros([1], &preds[0].device()))
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

    /// Calculate structure loss.
    ///
    /// # Arguments
    /// * `pred` - Predicted segmentation map with shape [N, C, H, W]
    /// * `target` - Ground truth segmentation map with shape [N, C, H, W]
    ///
    /// # Returns
    /// Structure loss tensor
    pub fn forward(&self, pred: Tensor<B, 4>, target: Tensor<B, 4>) -> Tensor<B, 1> {
        // Calculate edge-aware weight using average pooling
        let [n, c, h, w] = target.dims();

        // Create average pooling layer with kernel size 31
        let avg_pool = AvgPool2dConfig::new([31, 31])
            .with_strides([1, 1])
            .with_padding(PaddingConfig2d::Explicit(15, 15))
            .init();

        // Apply average pooling to get smoothed target
        let pooled = avg_pool.forward(target.clone());

        // Calculate edge weight: weit = 1 + 5 * |avg_pool(target) - target|
        let weit = (pooled - target.clone())
            .abs()
            .mul_scalar(5.0)
            .add_scalar(1.0);

        // Weighted BCE loss
        // Using the numerically stable formulation: `max(x, 0) - x*y + log(1 + exp(-abs(x)))`
        let bce_term1 = pred.clone().clamp_min(0.0) - pred.clone() * target.clone();
        let bce_term2 = (-pred.clone().abs()).exp().add_scalar(1.0).log();
        let wbce = (bce_term1 + bce_term2) * weit.clone();
        let wbce_loss = wbce.sum_dim(2).sum_dim(2) / weit.clone().sum_dim(2).sum_dim(2);

        // Weighted IoU loss
        let pred_sigmoid = burn::tensor::activation::sigmoid(pred);
        let inter = (pred_sigmoid.clone() * target.clone() * weit.clone())
            .sum_dim(2)
            .sum_dim(2);
        let union = ((pred_sigmoid + target) * weit).sum_dim(2).sum_dim(2);
        let wiou = (inter.clone().add_scalar(1.0) / (union - inter).add_scalar(1.0))
            .neg()
            .add_scalar(1.0);

        // Combine losses
        (wbce_loss + wiou).mean() * self.weight
    }
}

/// Configuration for Contour Loss function.
#[derive(Config, Debug)]
pub struct ContourLossConfig {
    #[config(default = 10.0)]
    pub length_weight: f32,
    #[config(default = 1e-8)]
    pub epsilon: f32,
}

/// Contour loss for boundary refinement.
#[derive(Module, Debug)]
pub struct ContourLoss<B: Backend> {
    pub length_weight: f32,
    pub epsilon: f32,
    _phantom: std::marker::PhantomData<B>,
}

impl ContourLossConfig {
    /// Initialize a new contour loss function with the given configuration.
    pub const fn init<B: Backend>(&self) -> ContourLoss<B> {
        ContourLoss {
            length_weight: self.length_weight,
            epsilon: self.epsilon,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B: Backend> Default for ContourLoss<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> ContourLoss<B> {
    /// Create a new contour loss function with default configuration.
    pub fn new() -> Self {
        ContourLossConfig::new().init()
    }

    /// Create a new contour loss function with custom weight.
    pub fn with_weight(length_weight: f32) -> Self {
        ContourLossConfig::new()
            .with_length_weight(length_weight)
            .init()
    }

    /// Calculate contour loss.
    ///
    /// # Arguments
    /// * `pred` - Predicted segmentation map with shape [N, C, H, W]
    /// * `target` - Ground truth segmentation map with shape [N, C, H, W]
    ///
    /// # Returns
    /// Contour loss tensor
    pub fn forward(&self, pred: Tensor<B, 4>, target: Tensor<B, 4>) -> Tensor<B, 1> {
        // The original PyTorch implementation computes loss on probabilities
        let pred_prob = burn::tensor::activation::sigmoid(pred);

        let [n, c, h, w] = pred_prob.dims();

        // length term - replicating the specific slicing from the original PyTorch code
        // delta_r = pred[:,:,1:,:] - pred[:,:,:-1,:]
        let delta_r = pred_prob.clone().slice([0..n, 0..c, 1..h, 0..w])
            - pred_prob.clone().slice([0..n, 0..c, 0..h - 1, 0..w]);

        // delta_c = pred[:,:,:,1:] - pred[:,:,:,:-1]
        let delta_c = pred_prob.clone().slice([0..n, 0..c, 0..h, 1..w])
            - pred_prob.clone().slice([0..n, 0..c, 0..h, 0..w - 1]);

        // These specific slices require the input to be at least 3x3.
        // Panics will occur on smaller inputs, which matches the original PyTorch behavior.
        // delta_r    = delta_r[:,:,1:,:-2]**2
        let delta_r_sq = delta_r
            .slice([0..n, 0..c, 1..(h - 1), 0..(w - 2)])
            .powf_scalar(2.0);

        // delta_c    = delta_c[:,:,:-2,1:]**2
        let delta_c_sq = delta_c
            .slice([0..n, 0..c, 0..(h - 2), 1..(w - 1)])
            .powf_scalar(2.0);

        let delta_pred = (delta_r_sq + delta_c_sq).abs();
        let length = (delta_pred + self.epsilon).sqrt().mean();

        // Region terms
        let c_in = Tensor::ones_like(&pred_prob);
        let c_out = Tensor::zeros_like(&pred_prob);

        // region_in = mean(pred * (target - c_in)^2)
        let region_in = (pred_prob.clone() * (target.clone() - c_in).powf_scalar(2.0)).mean();

        // region_out = mean((1-pred) * (target - c_out)^2)
        let region_out = ((Tensor::ones_like(&pred_prob) - pred_prob)
            * (target - c_out).powf_scalar(2.0))
        .mean();

        let region = region_in + region_out;

        // Total loss = weight * length + region
        length * self.length_weight + region
    }
}

/// Configuration for SSIM Loss function.
#[derive(Config, Debug)]
pub struct SSIMLossConfig {
    #[config(default = 11)]
    pub window_size: usize,
    #[config(default = true)]
    pub size_average: bool,
}

/// SSIM (Structural Similarity Index) loss.
#[derive(Module, Debug)]
pub struct SSIMLoss<B: Backend> {
    pub window_size: usize,
    pub size_average: bool,
    _phantom: std::marker::PhantomData<B>,
}

impl SSIMLossConfig {
    /// Initialize a new SSIM loss function with the given configuration.
    pub const fn init<B: Backend>(&self) -> SSIMLoss<B> {
        SSIMLoss {
            window_size: self.window_size,
            size_average: self.size_average,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B: Backend> Default for SSIMLoss<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> SSIMLoss<B> {
    /// Create a new SSIM loss function with default configuration.
    pub fn new() -> Self {
        SSIMLossConfig::new().init()
    }

    /// Create Gaussian window for SSIM calculation.
    fn create_window(&self, channels: usize, device: &B::Device) -> Tensor<B, 4> {
        let sigma = 1.5;
        let window_size = self.window_size as i32;
        let mean = window_size / 2;

        // Create 1D Gaussian window
        let mut gauss_1d = vec![0.0; self.window_size];
        let mut sum = 0.0;
        for (i, val) in gauss_1d.iter_mut().enumerate() {
            let x = i as i32 - mean;
            *val = (-(x * x) as f32 / (2.0 * sigma * sigma)).exp();
            sum += *val;
        }

        // Normalize
        for val in &mut gauss_1d {
            *val /= sum;
        }

        // Convert to tensor and create 2D window
        let window_1d = Tensor::<B, 1>::from_floats(gauss_1d.as_slice(), device);
        let window_1d = window_1d.unsqueeze::<2>();
        let window_2d = window_1d.clone().matmul(window_1d.transpose());

        // Expand to match channels
        window_2d
            .unsqueeze::<4>()
            .unsqueeze::<4>()
            .repeat(&[channels, 1, 1, 1])
    }

    /// Calculate SSIM loss.
    ///
    /// # Arguments
    /// * `img1` - First image tensor with shape [N, C, H, W]
    /// * `img2` - Second image tensor with shape [N, C, H, W]
    ///
    /// # Returns
    /// SSIM loss tensor (1 - SSIM)
    pub fn forward(&self, img1: Tensor<B, 4>, img2: Tensor<B, 4>) -> Tensor<B, 1> {
        let [_, channels, _, _] = img1.dims();
        let device = img1.device();

        // Create Gaussian window
        let window = self.create_window(channels, &device);

        // Constants for numerical stability
        let c1 = 0.01_f32.powi(2);
        let c2 = 0.03_f32.powi(2);

        let padding = self.window_size / 2;

        // Calculate local means using convolution with Gaussian window
        let mut conv = Conv2dConfig::new([channels, 1], [self.window_size, self.window_size])
            .with_padding(PaddingConfig2d::Explicit(padding, padding))
            .with_groups(channels)
            .with_bias(false)
            .init(&device);
        conv.weight = Param::from_tensor(window);

        let mu1 = conv.forward(img1.clone());
        let mu2 = conv.forward(img2.clone());

        let mu1_sq = mu1.clone().powf_scalar(2.0);
        let mu2_sq = mu2.clone().powf_scalar(2.0);
        let mu1_mu2 = mu1 * mu2;

        let sigma1_sq = conv.forward(img1.clone().powf_scalar(2.0)) - mu1_sq.clone();
        let sigma2_sq = conv.forward(img2.clone().powf_scalar(2.0)) - mu2_sq.clone();
        let sigma12 = conv.forward(img1 * img2) - mu1_mu2.clone();

        // SSIM calculation
        let ssim_n = (mu1_mu2 * 2.0 + c1) * (sigma12 * 2.0 + c2);
        let ssim_d = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2);
        let ssim = ssim_n / ssim_d;

        // Return 1 - SSIM as loss
        let one = Tensor::ones_like(&ssim);
        let loss = one - (ssim + 1.0) / 2.0;

        if self.size_average {
            loss.mean()
        } else {
            loss.mean_dim(1).mean_dim(1).mean_dim(1).mean()
        }
    }
}

/// Configuration for Patch IoU Loss function.
#[derive(Config, Debug)]
pub struct PatchIoULossConfig {
    #[config(default = 64)]
    pub patch_size: usize,
    #[config(default = 1e-6)]
    pub epsilon: f32,
}

/// Patch-based IoU loss for local region evaluation.
#[derive(Module, Debug)]
pub struct PatchIoULoss<B: Backend> {
    pub patch_size: usize,
    pub epsilon: f32,
    pub base_iou: CombinedLoss<B>,
    unfolder: Unfold4d,
}

impl PatchIoULossConfig {
    /// Initialize a new patch IoU loss function with the given configuration.
    pub fn init<B: Backend>(&self) -> PatchIoULoss<B> {
        let unfolder = Unfold4dConfig::new([self.patch_size, self.patch_size])
            .with_stride([self.patch_size, self.patch_size])
            .init();
        PatchIoULoss {
            patch_size: self.patch_size,
            epsilon: self.epsilon,
            base_iou: CombinedLoss::with_weights(0.0, 1.0),
            unfolder,
        }
    }
}

impl<B: Backend> Default for PatchIoULoss<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> PatchIoULoss<B> {
    /// Create a new patch IoU loss function with default configuration.
    pub fn new() -> Self {
        PatchIoULossConfig::new().init()
    }

    /// Calculate patch-based IoU loss.
    ///
    /// # Arguments
    /// * `pred` - Predicted segmentation map with shape [N, C, H, W]
    /// * `target` - Ground truth segmentation map with shape [N, C, H, W]
    ///
    /// # Returns
    /// Patch IoU loss tensor
    pub fn forward(&self, pred: Tensor<B, 4>, target: Tensor<B, 4>) -> Tensor<B, 1> {
        let [n, c, _, _] = pred.dims();
        let patch_size = self.patch_size;

        // Unfold predictions and targets into patches
        // The output shape is [N, C * patch_size * patch_size, num_patches]
        let pred_patches = self.unfolder.forward(pred);
        let target_patches = self.unfolder.forward(target);

        let num_patches = pred_patches.dims()[2];

        // Reshape to [N * num_patches, C, patch_size, patch_size] to compute IoU per patch
        let pred_reshaped = pred_patches.reshape([n * num_patches, c, patch_size, patch_size]);
        let target_reshaped = target_patches.reshape([n * num_patches, c, patch_size, patch_size]);

        // The IoU loss will be calculated on all patches, and the mean is returned.
        self.base_iou.iou_loss(pred_reshaped, target_reshaped)
    }
}

/// Configuration for Threshold Regularization Loss.
#[derive(Config, Debug)]
pub struct ThrRegLossConfig {
    #[config(default = 1.0)]
    pub weight: f32,
}

/// Threshold regularization loss to push predictions towards 0 or 1.
#[derive(Module, Debug)]
pub struct ThrRegLoss<B: Backend> {
    pub weight: f32,
    _phantom: std::marker::PhantomData<B>,
}

impl ThrRegLossConfig {
    /// Initialize a new threshold regularization loss with the given configuration.
    pub const fn init<B: Backend>(&self) -> ThrRegLoss<B> {
        ThrRegLoss {
            weight: self.weight,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B: Backend> Default for ThrRegLoss<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> ThrRegLoss<B> {
    /// Create a new threshold regularization loss with default configuration.
    pub fn new() -> Self {
        ThrRegLossConfig::new().init()
    }

    /// Calculate threshold regularization loss.
    ///
    /// # Arguments
    /// * `pred` - Predicted segmentation map with shape [N, C, H, W]
    /// * `_target` - Unused, kept for API consistency
    ///
    /// # Returns
    /// Threshold regularization loss tensor
    pub fn forward(&self, pred: Tensor<B, 4>, _target: Tensor<B, 4>) -> Tensor<B, 1> {
        // Loss = 1 - (pred^2 + (pred-1)^2)
        // Simplified: 1 - (pred^2 + pred^2 - 2*pred + 1) = 1 - (2*pred^2 - 2*pred + 1) = 2*pred - 2*pred^2 = 2*pred*(1-pred)
        // The original implementation is `torch.mean(1 - ((pred - 0) ** 2 + (pred - 1) ** 2))`
        // which simplifies to `torch.mean(2*pred - 2*pred**2)`
        // Let's stick to the original formula for clarity, but simplified.
        let pred_sq = pred.clone().powf_scalar(2.0);
        let pred_minus_one_sq = (pred - 1.0).powf_scalar(2.0);
        let reg = (pred_sq + pred_minus_one_sq).mean();
        (Tensor::ones_like(&reg) - reg) * self.weight
    }
}

/// Configuration for Classification Loss.
#[derive(Config, Debug)]
pub struct ClsLossConfig {
    #[config(default = 1.0)]
    pub weight: f32,
}

/// Classification loss for auxiliary supervision.
#[derive(Module, Debug)]
pub struct ClsLoss<B: Backend> {
    pub weight: f32,
    pub ce_loss: CrossEntropyLoss<B>,
}

impl ClsLossConfig {
    /// Initialize a new classification loss with the given configuration.
    pub fn init<B: Backend>(&self, device: &B::Device) -> ClsLoss<B> {
        ClsLoss {
            weight: self.weight,
            ce_loss: CrossEntropyLossConfig::new().init(device),
        }
    }
}

// Note: No Default implementation for ClsLoss since device parameter is required

impl<B: Backend> ClsLoss<B> {
    /// Create a new classification loss with default configuration.
    pub fn new(device: &B::Device) -> Self {
        ClsLossConfig::new().init(device)
    }

    /// Calculate classification loss.
    ///
    /// # Arguments
    /// * `preds` - List of predicted class logits
    /// * `targets` - Ground truth class labels
    ///
    /// # Returns
    /// Classification loss tensor
    pub fn forward(&self, preds: Vec<Tensor<B, 2>>, targets: Tensor<B, 1, Int>) -> Tensor<B, 1> {
        if preds.is_empty() {
            return Tensor::zeros([1], &targets.device());
        }

        let mut total_loss = Tensor::zeros([1], &targets.device());

        for pred in preds.iter() {
            let loss = self.ce_loss.forward(pred.clone(), targets.clone());
            total_loss = total_loss + loss;
        }

        total_loss * self.weight / (preds.len() as f32)
    }
}

/// Configuration for MAE/MSE Loss.
#[derive(Config, Debug)]
pub struct MaeLossConfig {
    #[config(default = 1.0)]
    pub weight: f32,
}

/// Mean Absolute Error (L1) loss.
#[derive(Module, Debug)]
pub struct MaeLoss<B: Backend> {
    pub weight: f32,
    _phantom: std::marker::PhantomData<B>,
}

impl MaeLossConfig {
    /// Initialize a new MAE loss with the given configuration.
    pub const fn init<B: Backend>(&self) -> MaeLoss<B> {
        MaeLoss {
            weight: self.weight,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B: Backend> Default for MaeLoss<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> MaeLoss<B> {
    /// Create a new MAE loss with default configuration.
    pub fn new() -> Self {
        MaeLossConfig::new().init()
    }

    /// Calculate MAE loss.
    pub fn forward(&self, pred: Tensor<B, 4>, target: Tensor<B, 4>) -> Tensor<B, 1> {
        (pred - target).abs().mean() * self.weight
    }
}

/// Configuration for MSE Loss.
#[derive(Config, Debug)]
pub struct MseLossConfig {
    #[config(default = 1.0)]
    pub weight: f32,
}

/// Mean Squared Error (L2) loss.
#[derive(Module, Debug)]
pub struct MseLoss<B: Backend> {
    pub weight: f32,
    _phantom: std::marker::PhantomData<B>,
}

impl MseLossConfig {
    /// Initialize a new MSE loss with the given configuration.
    pub const fn init<B: Backend>(&self) -> MseLoss<B> {
        MseLoss {
            weight: self.weight,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B: Backend> Default for MseLoss<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> MseLoss<B> {
    /// Create a new MSE loss with default configuration.
    pub fn new() -> Self {
        MseLossConfig::new().init()
    }

    /// Calculate MSE loss.
    pub fn forward(&self, pred: Tensor<B, 4>, target: Tensor<B, 4>) -> Tensor<B, 1> {
        (pred - target).powf_scalar(2.0).mean() * self.weight
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

    #[test]
    fn test_contour_loss_config() {
        let config = ContourLossConfig::new()
            .with_length_weight(20.0)
            .with_epsilon(1e-7);
        assert_eq!(config.length_weight, 20.0);
        assert_eq!(config.epsilon, 1e-7);
    }

    #[test]
    fn test_contour_loss() {
        let loss_fn = ContourLoss::<TestBackend>::new();

        let pred = Tensor::random(
            [1, 1, 8, 8],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &Default::default(),
        );
        let target = Tensor::random(
            [1, 1, 8, 8],
            burn::tensor::Distribution::Uniform(0.0, 1.0),
            &Default::default(),
        );

        let loss = loss_fn.forward(pred, target);

        // Check that loss is a scalar tensor
        assert_eq!(loss.shape().dims, [1]);
    }

    #[test]
    fn test_ssim_loss_config() {
        let config = SSIMLossConfig::new()
            .with_window_size(7)
            .with_size_average(false);
        assert_eq!(config.window_size, 7);
        assert!(!config.size_average);
    }

    #[test]
    fn test_ssim_loss() {
        let loss_fn = SSIMLoss::<TestBackend>::new();

        let img1 = Tensor::random(
            [1, 1, 16, 16],
            burn::tensor::Distribution::Normal(0.5, 0.1),
            &Default::default(),
        );
        let img2 = Tensor::random(
            [1, 1, 16, 16],
            burn::tensor::Distribution::Normal(0.5, 0.1),
            &Default::default(),
        );

        let loss = loss_fn.forward(img1, img2);

        // Check that loss is a scalar tensor
        assert_eq!(loss.shape().dims, [1]);
    }

    #[test]
    fn test_patch_iou_loss_config() {
        let config = PatchIoULossConfig::new()
            .with_patch_size(32)
            .with_epsilon(1e-5);
        assert_eq!(config.patch_size, 32);
        assert_eq!(config.epsilon, 1e-5);
    }

    #[test]
    fn test_patch_iou_loss() {
        let loss_fn = PatchIoULoss::<TestBackend>::new();

        let pred = Tensor::random(
            [1, 1, 128, 128],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &Default::default(),
        );
        let target = Tensor::random(
            [1, 1, 128, 128],
            burn::tensor::Distribution::Uniform(0.0, 1.0),
            &Default::default(),
        );

        let loss = loss_fn.forward(pred, target);

        // Check that loss is a scalar tensor
        assert_eq!(loss.shape().dims, [1]);
    }

    #[test]
    fn test_thr_reg_loss_config() {
        let config = ThrRegLossConfig::new().with_weight(0.5);
        assert_eq!(config.weight, 0.5);
    }

    #[test]
    fn test_thr_reg_loss() {
        let loss_fn = ThrRegLoss::<TestBackend>::new();

        let pred = Tensor::random(
            [1, 1, 4, 4],
            burn::tensor::Distribution::Uniform(0.0, 1.0),
            &Default::default(),
        );
        let target = Tensor::zeros([1, 1, 4, 4], &Default::default());

        let loss = loss_fn.forward(pred, target);

        // Check that loss is a scalar tensor
        assert_eq!(loss.shape().dims, [1]);
    }

    #[test]
    fn test_cls_loss_config() {
        let config = ClsLossConfig::new().with_weight(2.0);
        assert_eq!(config.weight, 2.0);
    }

    #[test]
    fn test_cls_loss() {
        let device = Default::default();
        let loss_fn = ClsLoss::<TestBackend>::new(&device);

        let preds = vec![
            Tensor::random(
                [4, 2],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &Default::default(),
            ),
            Tensor::random(
                [4, 2],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &Default::default(),
            ),
        ];
        let targets = Tensor::from_data([0, 1, 0, 1], &Default::default());

        let loss = loss_fn.forward(preds, targets);

        // Check that loss is a scalar tensor
        assert_eq!(loss.shape().dims, [1]);
    }

    #[test]
    fn test_mae_loss_config() {
        let config = MaeLossConfig::new().with_weight(1.5);
        assert_eq!(config.weight, 1.5);
    }

    #[test]
    fn test_mae_loss() {
        let loss_fn = MaeLoss::<TestBackend>::new();

        let pred = Tensor::random(
            [1, 1, 4, 4],
            burn::tensor::Distribution::Normal(0.5, 0.1),
            &Default::default(),
        );
        let target = Tensor::random(
            [1, 1, 4, 4],
            burn::tensor::Distribution::Normal(0.5, 0.1),
            &Default::default(),
        );

        let loss = loss_fn.forward(pred, target);

        // Check that loss is a scalar tensor
        assert_eq!(loss.shape().dims, [1]);
    }

    #[test]
    fn test_mse_loss_config() {
        let config = MseLossConfig::new().with_weight(0.8);
        assert_eq!(config.weight, 0.8);
    }

    #[test]
    fn test_mse_loss() {
        let loss_fn = MseLoss::<TestBackend>::new();

        let pred = Tensor::random(
            [1, 1, 4, 4],
            burn::tensor::Distribution::Normal(0.5, 0.1),
            &Default::default(),
        );
        let target = Tensor::random(
            [1, 1, 4, 4],
            burn::tensor::Distribution::Normal(0.5, 0.1),
            &Default::default(),
        );

        let loss = loss_fn.forward(pred, target);

        // Check that loss is a scalar tensor
        assert_eq!(loss.shape().dims, [1]);
    }
}
