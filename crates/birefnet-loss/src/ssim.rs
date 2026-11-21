//! Structural Similarity Index (SSIM) perceptual segmentation loss.

use burn::{
    module::{Content, DisplaySettings, ModuleDisplay},
    nn::loss::Reduction,
    prelude::*,
    tensor::{module::conv2d, ops::ConvOptions},
};

use crate::ops::{assert_same_shape, reduce_loss, usize_to_f64, usize_to_i32};

/// Configuration for [`SSIMLoss`].
#[derive(Config, Debug, PartialEq)]
pub struct SSIMLossConfig {
    /// Positive odd side length of the 2D Gaussian smoothing kernel.
    #[config(default = 11)]
    pub window_size: usize,
    /// Gaussian standard deviation in pixels.
    #[config(default = 1.5)]
    pub sigma: f64,
    /// Luminance stability constant (`0.01^2`).
    #[config(default = 0.0001)]
    pub c1: f64,
    /// Contrast stability constant (`0.03^2`).
    #[config(default = 0.0009)]
    pub c2: f64,
}

impl SSIMLossConfig {
    /// Initializes an [`SSIMLoss`] module.
    ///
    /// # Panics
    ///
    /// Panics if `window_size` is even or zero, or if `sigma`, `c1`, or `c2` are not finite and
    /// strictly positive.
    #[must_use]
    pub fn init(&self) -> SSIMLoss {
        assert!(
            self.window_size > 0 && self.window_size % 2 == 1,
            "SSIMLoss window_size must be positive and odd, got {}",
            self.window_size
        );
        assert!(
            self.sigma.is_finite() && self.sigma > 0.0,
            "SSIMLoss sigma must be finite and positive, got {}",
            self.sigma
        );
        assert!(
            self.c1.is_finite() && self.c1 > 0.0,
            "SSIMLoss c1 must be finite and positive, got {}",
            self.c1
        );
        assert!(
            self.c2.is_finite() && self.c2 > 0.0,
            "SSIMLoss c2 must be finite and positive, got {}",
            self.c2
        );

        SSIMLoss {
            window_size: self.window_size,
            sigma: self.sigma,
            c1: self.c1,
            c2: self.c2,
        }
    }
}

/// Structural Similarity Index (SSIM) loss mapped to `1 - (1 + SSIM) / 2` in `[0, 1]`.
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct SSIMLoss {
    window_size: usize,
    sigma: f64,
    c1: f64,
    c2: f64,
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
    /// Creates an [`SSIMLoss`] with default `11x11` Gaussian window (`sigma = 1.5`).
    #[must_use]
    pub fn new() -> Self {
        SSIMLossConfig::new().init()
    }

    /// Computes reduced SSIM loss `[1]` from probability maps and targets in `[0, 1]`.
    #[must_use]
    pub fn forward(
        &self,
        predictions: Tensor<4>,
        targets: Tensor<4>,
        reduction: Reduction,
    ) -> Tensor<1> {
        let loss = self.forward_no_reduction(predictions, targets);
        reduce_loss(loss, reduction)
    }

    /// Computes per-sample SSIM loss `[batch_size]` in `[0, 1]`.
    ///
    /// # Panics
    ///
    /// Panics if `predictions` and `targets` have different shapes.
    #[must_use]
    pub fn forward_no_reduction(&self, predictions: Tensor<4>, targets: Tensor<4>) -> Tensor<1> {
        assert_same_shape(&predictions, &targets);

        let [_, channels, ..] = predictions.dims();
        let device = predictions.device();
        let window = self.create_window(channels, &device);
        let ssim_values = self.compute_ssim_batch(predictions, targets, &window);

        ssim_values.ones_like() - (ssim_values + 1.0) / 2.0
    }

    fn create_window(&self, channels: usize, device: &Device) -> Tensor<4> {
        let half = usize_to_f64(self.window_size / 2);
        let two_sigma_sq = 2.0 * self.sigma * self.sigma;

        let mut gauss_1d: Vec<f64> = (0..self.window_size)
            .map(|idx| {
                let offset = usize_to_f64(idx) - half;
                (-(offset * offset) / two_sigma_sq).exp()
            })
            .collect();
        let sum: f64 = gauss_1d.iter().sum();
        for val in &mut gauss_1d {
            *val /= sum;
        }

        let window_1d = Tensor::<1>::from_floats(gauss_1d.as_slice(), device);
        let col = window_1d.clone().unsqueeze_dim::<2>(1);
        let row = window_1d.unsqueeze_dim::<2>(0);
        let window_2d = col.matmul(row);

        window_2d.unsqueeze::<4>().repeat(&[channels, 1, 1, 1])
    }

    fn compute_ssim_batch(
        &self,
        img1: Tensor<4>,
        img2: Tensor<4>,
        window: &Tensor<4>,
    ) -> Tensor<1> {
        let [batch_size, channels, ..] = img1.dims();
        let padding = self.window_size / 2;
        let conv_opts = ConvOptions::new([1, 1], [padding, padding], [1, 1], channels);

        let blur = |tensor: Tensor<4>| conv2d(tensor, window.clone(), None, conv_opts.clone());

        let mu1 = blur(img1.clone());
        let mu2 = blur(img2.clone());
        let mu1_sq = mu1.clone().powi_scalar(2);
        let mu2_sq = mu2.clone().powi_scalar(2);
        let mu1_mu2 = mu1 * mu2;

        let sigma1_sq = blur(img1.clone().powi_scalar(2)) - mu1_sq.clone();
        let sigma2_sq = blur(img2.clone().powi_scalar(2)) - mu2_sq.clone();
        let sigma12 = blur(img1 * img2) - mu1_mu2.clone();

        let ssim_num = (mu1_mu2.mul_scalar(2.0).add_scalar(self.c1))
            * (sigma12.mul_scalar(2.0).add_scalar(self.c2));
        let ssim_den = (mu1_sq + mu2_sq + self.c1) * (sigma1_sq + sigma2_sq + self.c2);
        let ssim_map = ssim_num / ssim_den;

        ssim_map
            .reshape([usize_to_i32(batch_size), -1])
            .mean_dim(1)
            .squeeze_dim::<1>(1)
    }
}
