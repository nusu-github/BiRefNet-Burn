//! SSIM (Structural Similarity Index) loss implementation.

use burn::{
    module::Param,
    nn::{conv::Conv2dConfig, PaddingConfig2d},
    prelude::*,
    tensor::{backend::Backend, Tensor},
};

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
