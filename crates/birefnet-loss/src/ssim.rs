//! SSIM (Structural Similarity Index) loss implementation.
//!
//! Computes the structural similarity index between two images and returns
//! the SSIM loss (1 - SSIM). The SSIM metric measures the similarity between
//! two images based on luminance, contrast, and structure.
//!
//! The loss is computed as:
//! ```text
//! SSIM = (2*μ₁*μ₂ + C₁)(2*σ₁₂ + C₂) / ((μ₁² + μ₂² + C₁)(σ₁² + σ₂² + C₂))
//! Loss = 1 - (1 + SSIM) / 2
//! ```

use burn::{
    config::Config,
    module::{Content, DisplaySettings, Module, ModuleDisplay, Param},
    nn::{conv::Conv2dConfig, loss::Reduction, PaddingConfig2d},
    tensor::{backend::Backend, Tensor},
};

/// Configuration for creating an [SSIM loss](SSIMLoss).
#[derive(Config, Debug)]
pub struct SSIMLossConfig {
    /// Size of the Gaussian window for SSIM computation. Default: 11
    #[config(default = 11)]
    pub window_size: usize,

    /// Sigma parameter for Gaussian window. Default: 1.5
    #[config(default = 1.5)]
    pub sigma: f64,

    /// First stability constant (luminance). Default: 0.01²
    #[config(default = 0.0001)]
    pub c1: f64,

    /// Second stability constant (contrast). Default: 0.03²
    #[config(default = 0.0009)]
    pub c2: f64,
}

impl SSIMLossConfig {
    /// Initialize [SSIM loss](SSIMLoss).

    pub fn init(&self) -> SSIMLoss {
        self.assertions();
        SSIMLoss {
            window_size: self.window_size,
            sigma: self.sigma,
            c1: self.c1,
            c2: self.c2,
        }
    }

    fn assertions(&self) {
        assert!(
            self.window_size > 0 && self.window_size % 2 == 1,
            "Window size for SSIMLoss must be positive and odd, got {}",
            self.window_size
        );
        assert!(
            self.sigma > 0.0,
            "Sigma for SSIMLoss must be positive, got {}",
            self.sigma
        );
        assert!(
            self.c1 > 0.0,
            "C1 for SSIMLoss must be positive, got {}",
            self.c1
        );
        assert!(
            self.c2 > 0.0,
            "C2 for SSIMLoss must be positive, got {}",
            self.c2
        );
    }
}

/// SSIM (Structural Similarity Index) loss.
///
/// Computes the structural similarity between two images and returns
/// the corresponding loss value. Uses a Gaussian window for local
/// similarity computation and includes stability constants.
#[derive(Module, Clone, Debug)]
#[module(custom_display)]
pub struct SSIMLoss {
    /// Size of the Gaussian window.
    pub window_size: usize,
    /// Sigma parameter for Gaussian window.
    pub sigma: f64,
    /// First stability constant (luminance).
    pub c1: f64,
    /// Second stability constant (contrast).
    pub c2: f64,
}

impl Default for SSIMLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl ModuleDisplay for SSIMLoss {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("window_size", &self.window_size)
            .add("sigma", &self.sigma)
            .add("c1", &self.c1)
            .add("c2", &self.c2)
            .optional()
    }
}

impl SSIMLoss {
    /// Create a new SSIM loss with default configuration.

    pub fn new() -> Self {
        SSIMLossConfig::new().init()
    }

    /// Compute the criterion on the input tensor with reduction.
    ///
    /// # Shapes
    ///
    /// - predictions: `[batch_size, channels, height, width]`
    /// - targets: `[batch_size, channels, height, width]`
    /// - output: `[1]`
    pub fn forward<B: Backend>(
        &self,
        predictions: Tensor<B, 4>,
        targets: Tensor<B, 4>,
        reduction: Reduction,
    ) -> Tensor<B, 1> {
        let loss = self.forward_no_reduction(predictions, targets);
        match reduction {
            Reduction::Mean | Reduction::Auto => loss.mean(),
            Reduction::Sum => loss.sum(),
        }
    }

    /// Compute the criterion on the input tensor without reduction.
    ///
    /// # Shapes
    ///
    /// - predictions: `[batch_size, channels, height, width]`
    /// - targets: `[batch_size, channels, height, width]`
    /// - output: `[batch_size]`
    pub fn forward_no_reduction<B: Backend>(
        &self,
        predictions: Tensor<B, 4>,
        targets: Tensor<B, 4>,
    ) -> Tensor<B, 1> {
        self.assertions(&predictions, &targets);

        let [batch_size, channels, height, width] = predictions.dims();
        let device = predictions.device();

        // Create Gaussian window
        let window = self.create_window::<B>(channels, &device);

        // SSIM constants (use f64 directly)
        let c1 = self.c1;
        let c2 = self.c2;

        // Calculate SSIM for the entire batch
        let ssim_values = self.compute_ssim_batch(predictions, targets, window, c1, c2);

        // Convert to loss: 1 - (1 + ssim) / 2
        let ones = Tensor::ones_like(&ssim_values);

        ones - (ssim_values + 1.0) / 2.0
    }

    /// Create Gaussian window for SSIM calculation.
    fn create_window<B: Backend>(&self, channels: usize, device: &B::Device) -> Tensor<B, 4> {
        let window_size = self.window_size as i32;
        let mean = window_size / 2;

        // Create 1D Gaussian window
        let mut gauss_1d = vec![0.0; self.window_size];
        let mut sum = 0.0;
        for (i, val) in gauss_1d.iter_mut().enumerate() {
            let x = i as i32 - mean;
            *val = (f64::from(-(x * x)) / (2.0 * self.sigma * self.sigma)).exp();
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

        // Expand to match channels: [channels, 1, window_size, window_size]
        window_2d
            .unsqueeze::<4>()
            .unsqueeze::<4>()
            .repeat(&[channels, 1, 1, 1])
    }

    /// Compute SSIM for batch of images.
    fn compute_ssim_batch<B: Backend>(
        &self,
        img1: Tensor<B, 4>,
        img2: Tensor<B, 4>,
        window: Tensor<B, 4>,
        c1: f64,
        c2: f64,
    ) -> Tensor<B, 1> {
        let [batch_size, channels, height, width] = img1.dims();
        let device = img1.device();
        let padding = self.window_size / 2;

        // Create convolution layer with Gaussian window
        let mut conv =
            Conv2dConfig::new([channels, channels], [self.window_size, self.window_size])
                .with_padding(PaddingConfig2d::Explicit(padding, padding))
                .with_groups(channels)
                .with_bias(false)
                .init(&device);
        conv.weight = Param::from_tensor(window);

        // Calculate local means
        let mu1 = conv.forward(img1.clone());
        let mu2 = conv.forward(img2.clone());

        let mu1_sq = mu1.clone().powi_scalar(2);
        let mu2_sq = mu2.clone().powi_scalar(2);
        let mu1_mu2 = mu1 * mu2;

        // Calculate local variances and covariance
        let sigma1_sq = conv.forward(img1.clone().powi_scalar(2)) - mu1_sq.clone();
        let sigma2_sq = conv.forward(img2.clone().powi_scalar(2)) - mu2_sq.clone();
        let sigma12 = conv.forward(img1 * img2) - mu1_mu2.clone();

        // SSIM calculation using add_scalar
        let ssim_n =
            (mu1_mu2.mul_scalar(2.0).add_scalar(c1)) * (sigma12.mul_scalar(2.0).add_scalar(c2));
        let ssim_d = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2);
        let ssim_map = ssim_n / ssim_d;

        // Average over spatial and channel dimensions for each batch element
        // [B, C, H, W] -> [B, 1] -> [B]
        ssim_map
            .reshape([batch_size as i32, -1])
            .mean_dim(1)
            .squeeze(1)
    }

    fn assertions<B: Backend>(&self, predictions: &Tensor<B, 4>, targets: &Tensor<B, 4>) {
        let pred_dims = predictions.dims();
        let target_dims = targets.dims();
        assert_eq!(
            pred_dims, target_dims,
            "Shape of predictions ({pred_dims:?}) must match targets ({target_dims:?})"
        );
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{cast::ToElement, TensorData};

    use super::*;
    use crate::tests::TestBackend;

    #[test]
    fn ssim_loss_forward_identical_images_returns_low_loss() {
        let device = Default::default();
        let loss = SSIMLoss::new();

        // Identical images should have SSIM ≈ 1, loss ≈ 0
        let img1 = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([[[[0.5, 0.6], [0.7, 0.8]]]]),
            &device,
        );
        let img2 = img1.clone();

        let result_mean = loss.forward(img1.clone(), img2.clone(), Reduction::Mean);
        let result_no_reduction = loss.forward_no_reduction(img1, img2);

        // Should be very low loss for identical images
        assert!(result_mean.clone().into_scalar().to_f64() >= 0.0);
        assert!(result_mean.into_scalar().to_f64() < 1.0);

        assert_eq!(result_no_reduction.dims(), [1]); // batch_size = 1
        assert!(result_no_reduction.into_scalar().to_f64() >= 0.0);
    }

    #[test]
    fn ssim_loss_forward_different_images_returns_higher_loss() {
        let device = Default::default();
        let loss = SSIMLoss::new();

        // Very different images
        let img1 = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([[[[1.0, 1.0], [1.0, 1.0]]]]),
            &device,
        );
        let img2 = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([[[[0.0, 0.0], [0.0, 0.0]]]]),
            &device,
        );

        let result_mean = loss.forward(img1.clone(), img2.clone(), Reduction::Mean);
        let result_sum = loss.forward(img1.clone(), img2.clone(), Reduction::Sum);
        let result_no_reduction = loss.forward_no_reduction(img1, img2);

        // Should be higher loss for different images
        assert!(result_mean.clone().into_scalar().to_f64() > 0.0);
        assert!(result_sum.into_scalar().to_f64() >= result_mean.into_scalar().to_f64());
        assert_eq!(result_no_reduction.dims(), [1]);
    }

    #[test]
    fn ssim_loss_forward_batch_images_validates_range_and_shapes() {
        let device = Default::default();
        let loss = SSIMLoss::new();

        // Batch of 2 images
        let img1 = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([
                [[[0.8, 0.2], [0.3, 0.9]]],    // Sample 1
                [[[0.1, 0.95], [0.85, 0.05]]], // Sample 2
            ]),
            &device,
        );
        let img2 = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([
                [[[0.7, 0.3], [0.4, 0.8]]],    // Sample 1 (similar)
                [[[0.9, 0.05], [0.15, 0.95]]], // Sample 2 (different)
            ]),
            &device,
        );

        let result_mean = loss.forward(img1.clone(), img2.clone(), Reduction::Mean);
        let result_no_reduction = loss.forward_no_reduction(img1, img2);

        // Check shapes
        assert_eq!(result_mean.dims(), [1]);
        assert_eq!(result_no_reduction.dims(), [2]); // batch_size = 2

        // All values should be finite and in [0, 1] range
        assert!(result_mean.into_scalar().to_f64().is_finite());
        for i in 0..2 {
            let sample_loss = result_no_reduction
                .clone()
                .select(0, Tensor::from_data([i], &device))
                .into_scalar()
                .to_f64();
            assert!(
                (0.0..=1.0).contains(&sample_loss),
                "Sample {i} loss should be in [0,1], got {sample_loss}"
            );
        }
    }

    #[test]
    fn ssim_loss_forward_multichannel_images_works() {
        let device = Default::default();
        let loss = SSIMLoss::new();

        // RGB images (3 channels)
        let img1 = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([[
                [[0.8, 0.2], [0.3, 0.9]], // R channel
                [[0.1, 0.7], [0.6, 0.4]], // G channel
                [[0.5, 0.5], [0.2, 0.8]], // B channel
            ]]),
            &device,
        );
        let img2 = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([[
                [[0.7, 0.3], [0.4, 0.8]], // R channel
                [[0.2, 0.6], [0.7, 0.3]], // G channel
                [[0.4, 0.6], [0.3, 0.7]], // B channel
            ]]),
            &device,
        );

        let result = loss.forward(img1, img2, Reduction::Mean);

        // Should handle multi-channel input
        assert!(result.clone().into_scalar().to_f64().is_finite());
        assert!(result.clone().into_scalar().to_f64() >= 0.0);
        assert!(result.into_scalar().to_f64() <= 1.0);
    }

    #[test]
    fn ssim_loss_with_custom_window_size_and_constants_works() {
        let device = Default::default();
        let config = SSIMLossConfig::new()
            .with_window_size(5)
            .with_sigma(1.0)
            .with_c1(0.0001)
            .with_c2(0.0009);
        let loss = config.init();

        let img1 = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([[[[0.5, 0.6], [0.7, 0.8]]]]),
            &device,
        );
        let img2 = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([[[[0.4, 0.7], [0.6, 0.9]]]]),
            &device,
        );

        let result = loss.forward(img1, img2, Reduction::Mean);

        // Should work with custom parameters
        assert!(result.clone().into_scalar().to_f64().is_finite());
        assert!(result.into_scalar().to_f64() >= 0.0);
    }

    #[test]
    fn ssim_loss_auto_reduction_equals_mean_reduction() {
        let device = Default::default();
        let loss = SSIMLoss::new();

        let img1 = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([[[[0.3, 0.7], [0.8, 0.2]]]]),
            &device,
        );
        let img2 = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([[[[0.4, 0.6], [0.7, 0.3]]]]),
            &device,
        );

        let result_auto = loss.forward(img1.clone(), img2.clone(), Reduction::Auto);
        let result_mean = loss.forward(img1, img2, Reduction::Mean);

        let auto_val = result_auto.into_scalar().to_f64();
        let mean_val = result_mean.into_scalar().to_f64();

        assert!((auto_val - mean_val).abs() < 1e-6, "Auto should equal Mean");
    }

    #[test]
    #[should_panic = "Window size for SSIMLoss must be positive and odd"]
    fn ssim_loss_config_even_window_size_panics() {
        let _loss = SSIMLossConfig::new().with_window_size(10).init();
    }

    #[test]
    #[should_panic = "Sigma for SSIMLoss must be positive"]
    fn ssim_loss_config_negative_sigma_panics() {
        let _loss = SSIMLossConfig::new().with_sigma(-1.0).init();
    }

    #[test]
    #[should_panic = "C1 for SSIMLoss must be positive"]
    fn ssim_loss_config_negative_c1_panics() {
        let _loss = SSIMLossConfig::new().with_c1(-0.01).init();
    }

    #[test]
    #[should_panic = "Shape of predictions"]
    fn ssim_loss_forward_mismatched_shapes_panics() {
        let device = Default::default();
        let loss = SSIMLoss::new();

        let img1 = Tensor::<TestBackend, 4>::from_data(TensorData::from([[[[1.0, 2.0]]]]), &device);
        let img2 = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([[[[1.0, 2.0], [3.0, 4.0]]]]),
            &device,
        );

        let _result = loss.forward_no_reduction(img1, img2);
    }

    #[test]
    fn ssim_loss_display_shows_configuration_parameters() {
        let config = SSIMLossConfig::new()
            .with_window_size(7)
            .with_sigma(2.0)
            .with_c1(0.001)
            .with_c2(0.002);
        let loss = config.init();

        let display_str = format!("{loss}");
        assert!(display_str.contains("SSIMLoss"));
        assert!(display_str.contains("window_size: 7"));
        assert!(display_str.contains("sigma: 2"));
        assert!(display_str.contains("c1: 0.001"));
        assert!(display_str.contains("c2: 0.002"));
    }
}
