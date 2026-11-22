//! Structural Similarity Index (SSIM) perceptual segmentation loss.

use burn::{
    module::{Content, DisplaySettings, ModuleDisplay},
    nn::loss::Reduction,
    prelude::*,
    tensor::{module::conv2d, ops::ConvOptions},
};

use crate::ops::{assert_same_shape, mean_per_sample, reduce_loss, usize_to_f64};

/// Configuration for [`SsimLoss`].
#[derive(Config, Debug, PartialEq)]
pub struct SsimLossConfig {
    /// Positive odd side length of the Gaussian smoothing window.
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

impl SsimLossConfig {
    /// Initializes an [`SsimLoss`] module.
    ///
    /// # Panics
    ///
    /// Panics if `window_size` is even or zero, or if `sigma`, `c1`, or `c2` are not finite and
    /// strictly positive.
    #[must_use]
    pub fn init(&self) -> SsimLoss {
        assert!(
            self.window_size > 0 && self.window_size % 2 == 1,
            "SsimLoss window_size must be positive and odd, got {}",
            self.window_size
        );
        for (name, value) in [("sigma", self.sigma), ("c1", self.c1), ("c2", self.c2)] {
            assert!(
                value.is_finite() && value > 0.0,
                "SsimLoss {name} must be finite and positive, got {value}"
            );
        }

        SsimLoss {
            window_size: self.window_size,
            sigma: self.sigma,
            c1: self.c1,
            c2: self.c2,
            kernel_1d: gaussian_kernel_1d(self.window_size, self.sigma),
        }
    }
}

/// Normalized 1D Gaussian taps centred on `window_size / 2`.
fn gaussian_kernel_1d(window_size: usize, sigma: f64) -> Vec<f64> {
    let half = usize_to_f64(window_size / 2);
    let two_sigma_sq = 2.0 * sigma * sigma;
    let mut taps = Vec::from_fn(window_size, |idx| {
        let offset = usize_to_f64(idx) - half;
        (-(offset * offset) / two_sigma_sq).exp()
    });
    let sum: f64 = taps.iter().sum();
    for tap in &mut taps {
        *tap /= sum;
    }
    taps
}

/// Structural Similarity Index (SSIM) loss mapped to `1 - (1 + SSIM) / 2` in `[0, 1]`.
///
/// The Gaussian window is applied as two separable 1D convolutions, which is exactly equal
/// to the 2D zero-padded window used by the reference implementation.
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct SsimLoss {
    window_size: usize,
    sigma: f64,
    c1: f64,
    c2: f64,
    kernel_1d: Vec<f64>,
}

impl Default for SsimLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl ModuleDisplay for SsimLoss {
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

impl SsimLoss {
    /// Creates an [`SsimLoss`] with default `11x11` Gaussian window (`sigma = 1.5`).
    #[must_use]
    pub fn new() -> Self {
        SsimLossConfig::new().init()
    }

    /// Computes reduced SSIM loss `[1]` from probability maps and targets in `[0, 1]`.
    ///
    /// # Panics
    ///
    /// Panics if `predictions` and `targets` have different shapes.
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
        let ssim = self.ssim_per_sample(predictions, targets);
        // 1 - (1 + ssim) / 2 == (1 - ssim) / 2
        ssim.neg().add_scalar(1.0).div_scalar(2.0)
    }

    /// Builds the vertical `[C, 1, K, 1]` and horizontal `[C, 1, 1, K]` depthwise kernels.
    fn kernels(&self, channels: usize, like: &Tensor<4>) -> (Tensor<4>, Tensor<4>) {
        let size = self.window_size;
        let data = TensorData::new(self.kernel_1d.clone(), [size]);
        let taps = Tensor::<1>::from_data(data, (&like.device(), like.dtype()));
        let vertical = taps
            .clone()
            .reshape([1, 1, size, 1])
            .repeat_dim(0, channels);
        let horizontal = taps.reshape([1, 1, 1, size]).repeat_dim(0, channels);
        (vertical, horizontal)
    }

    fn ssim_per_sample(&self, img1: Tensor<4>, img2: Tensor<4>) -> Tensor<1> {
        let [_, channels, ..] = img1.dims();
        let padding = self.window_size / 2;
        let (vertical, horizontal) = self.kernels(channels, &img1);
        let vertical_opts = ConvOptions::new([1, 1], [padding, 0], [1, 1], channels);
        let horizontal_opts = ConvOptions::new([1, 1], [0, padding], [1, 1], channels);

        let blur = |tensor: Tensor<4>| {
            let rows = conv2d(tensor, vertical.clone(), None, vertical_opts.clone());
            conv2d(rows, horizontal.clone(), None, horizontal_opts.clone())
        };

        let mu1 = blur(img1.clone());
        let mu2 = blur(img2.clone());
        let mu1_sq = mu1.clone().square();
        let mu2_sq = mu2.clone().square();
        let mu1_mu2 = mu1 * mu2;

        // E[X^2] - E[X]^2 can round slightly below zero; variances are non-negative by definition.
        let sigma1_sq = (blur(img1.clone().square()) - mu1_sq.clone()).clamp_min(0.0);
        let sigma2_sq = (blur(img2.clone().square()) - mu2_sq.clone()).clamp_min(0.0);
        let sigma12 = blur(img1 * img2) - mu1_mu2.clone();

        let ssim_num = (mu1_mu2.mul_scalar(2.0).add_scalar(self.c1))
            * (sigma12.mul_scalar(2.0).add_scalar(self.c2));
        let ssim_den =
            (mu1_sq + mu2_sq).add_scalar(self.c1) * (sigma1_sq + sigma2_sq).add_scalar(self.c2);

        mean_per_sample(ssim_num / ssim_den)
    }
}
