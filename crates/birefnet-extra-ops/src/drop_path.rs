//! # DropPath Regularization
//!
//! Implements the DropPath regularization technique, also known as stochastic depth.
//! During training, it randomly drops entire paths (sub-networks) and scales the
//! remaining ones, effectively preventing co-adaptation of parallel paths.

use burn::{
    module::{Flag, Param},
    prelude::*,
    tensor::Distribution,
};

/// Configuration for the `DropPath` module.
#[derive(Config, Debug)]
pub struct DropPathConfig {
    /// The probability of dropping a path.
    #[config(default = "0.0")]
    pub drop_prob: f64,
    /// Whether to scale the output by the keep probability.
    #[config(default = "true")]
    pub scale_by_keep: bool,
}

impl DropPathConfig {
    /// Initializes a new `DropPath` module.
    pub fn init(&self) -> DropPath {
        DropPath {
            drop_prob: self.drop_prob,
            scale_by_keep: self.scale_by_keep,
            training: Param::from_bool(true),
        }
    }
}

/// DropPath module.
///
/// Like [`burn::nn::Dropout`], it is only active while its training flag is set (cleared by
/// [`Module::valid`] and [`Module::freeze`]) and its input lives on an autodiff device.
#[derive(Module, Debug)]
pub struct DropPath {
    drop_prob: f64,
    scale_by_keep: bool,
    training: Param<Flag>,
}

impl DropPath {
    /// Applies DropPath to the input tensor.
    ///
    /// Returns the input unchanged outside of training or when `drop_prob` is 0.
    /// Otherwise, it randomly zeros out entire examples in the batch with probability `drop_prob`.
    /// The mask is generated only for the batch dimension and broadcasted to all other dimensions.
    ///
    /// # Shapes
    /// - input: `[batch_size, ..., channels]`
    /// - output: `[batch_size, ..., channels]`
    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        if !self.training.is_enabled() || !x.device().is_autodiff() || self.drop_prob == 0.0 {
            return x;
        }
        let keep_prob = 1.0 - self.drop_prob;
        let batch_size = x.dims()[0];

        // Create mask with shape [batch_size, 1, 1, ...] for proper broadcasting
        // This matches timm's implementation where the mask is applied per batch item
        let mut mask_shape = [1; D];
        mask_shape[0] = batch_size;

        let random_tensor =
            Tensor::random(mask_shape, Distribution::Bernoulli(keep_prob), &x.device());

        if self.scale_by_keep {
            x * random_tensor / keep_prob
        } else {
            x * random_tensor
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::Tensor;
    use rstest::rstest;

    use super::*;

    #[test]
    fn droppath_eval_mode_returns_input_unchanged() {
        let device = Device::flex().autodiff();
        // `valid()` clears the training flag (evaluation mode).
        let drop_path = DropPathConfig::new().with_drop_prob(0.2).init().valid();

        // Test input tensor
        let x = Tensor::<4>::ones([2, 3, 4, 4], &device);

        // In evaluation mode, should return input unchanged
        let output = drop_path.forward(x.clone());

        // Verify input and output are equal
        let diff = (output - x).abs().sum();
        assert_eq!(
            diff.into_scalar::<f32>(),
            0.0,
            "In evaluation mode, input and output should be equal"
        );
    }

    #[test]
    fn droppath_zero_prob_returns_input_unchanged() {
        let device = Device::flex().autodiff();
        // Drop probability of 0
        let drop_path = DropPathConfig::new().with_drop_prob(0.0).init();

        // Test input tensor
        let x = Tensor::<4>::ones([2, 3, 4, 4], &device);

        // With drop_prob=0, should return input unchanged
        let output = drop_path.forward(x.clone());

        // Verify input and output are equal
        let diff = (output - x).abs().sum();
        assert_eq!(
            diff.into_scalar::<f32>(),
            0.0,
            "With drop_prob=0, input and output should be equal"
        );
    }

    #[test]
    fn droppath_is_inactive_on_a_plain_device() {
        // Like `Dropout`, inputs that are not on an autodiff device are never dropped.
        let device = Device::flex();
        let drop_path = DropPathConfig::new().with_drop_prob(0.99).init();
        let x = Tensor::<2>::ones([8, 4], &device);

        let output = drop_path.forward(x.clone());

        assert_eq!((output - x).abs().sum().into_scalar::<f32>(), 0.0);
    }

    #[rstest]
    #[case(vec![2, 512], "2D")]
    #[case(vec![2, 196, 256], "3D")]
    #[case(vec![2, 3, 4, 4], "4D")]
    fn droppath_preserves_tensor_dimensions(#[case] shape: Vec<usize>, #[case] description: &str) {
        let device = Device::flex().autodiff();
        let drop_path = DropPathConfig::new().with_drop_prob(0.5).init();

        let dims = shape.len();
        match dims {
            2 => {
                let x = Tensor::<2>::ones([shape[0], shape[1]], &device);
                let output = drop_path.forward(x.clone());
                assert_eq!(
                    output.dims(),
                    x.dims(),
                    "{description} shape should be preserved"
                );
            }
            3 => {
                let x = Tensor::<3>::ones([shape[0], shape[1], shape[2]], &device);
                let output = drop_path.forward(x.clone());
                assert_eq!(
                    output.dims(),
                    x.dims(),
                    "{description} shape should be preserved"
                );
            }
            4 => {
                let x = Tensor::<4>::ones([shape[0], shape[1], shape[2], shape[3]], &device);
                let output = drop_path.forward(x.clone());
                assert_eq!(
                    output.dims(),
                    x.dims(),
                    "{description} shape should be preserved"
                );
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn droppath_training_mode_achieves_expected_drop_rate() {
        let device = Device::flex().autodiff();
        let drop_path = DropPathConfig::new().with_drop_prob(0.5).init();

        // Test with batch size 10 (for statistical verification)
        let batch_size = 10;
        let x = Tensor::<4>::ones([batch_size, 3, 4, 4], &device);

        // Run multiple times to gather statistics
        let mut drop_counts = 0;
        let num_trials = 100;

        for _ in 0..num_trials {
            let output = drop_path.forward(x.clone());

            // Check if each batch element was dropped
            for i in 0..batch_size {
                let batch_output = output.clone().slice(s![i..=i, .., .., ..]);
                let sum = batch_output.sum().into_scalar::<f32>();

                // If sum == 0, it was dropped
                if sum.abs() < 1e-6 {
                    drop_counts += 1;
                }
            }
        }

        // Expected drop rate is 0.5
        let actual_drop_rate = drop_counts as f64 / (num_trials * batch_size) as f64;

        // Consider statistical error (±0.1 range)
        assert!(
            (actual_drop_rate - 0.5).abs() < 0.1,
            "Actual drop rate {actual_drop_rate} deviates significantly from expected 0.5"
        );
    }

    #[test]
    fn droppath_scaling_behavior_varies_with_scale_by_keep() {
        let device = Device::flex().autodiff();

        // Case with scale_by_keep = true
        // Set drop_prob to 0 to verify scaling
        let drop_path_with_scale = DropPathConfig::new().with_scale_by_keep(true).init();

        // Case with scale_by_keep = false
        let drop_path_no_scale = DropPathConfig::new().with_scale_by_keep(false).init();

        let x = Tensor::<2>::ones([2, 4], &device);

        let output_with_scale = drop_path_with_scale.forward(x.clone());
        let output_no_scale = drop_path_no_scale.forward(x.clone());

        // With drop_prob=0, both should return input unchanged
        assert_eq!(
            output_with_scale.sum().into_scalar::<f32>(),
            x.clone().sum().into_scalar::<f32>(),
            "With scale_by_keep=true and drop_prob=0"
        );
        assert_eq!(
            output_no_scale.sum().into_scalar::<f32>(),
            x.sum().into_scalar::<f32>(),
            "With scale_by_keep=false and drop_prob=0"
        );
    }

    #[test]
    fn droppath_applies_independently_per_batch_element() {
        let device = Device::flex().autodiff();
        let drop_path = DropPathConfig::new().with_drop_prob(0.5).init();

        // Test with batch size 4
        let x = Tensor::<3>::ones([4, 8, 16], &device);
        let output = drop_path.forward(x);

        // Check the state of each batch element
        let mut batch_states = vec![];
        for i in 0..4 {
            let batch_elem = output.clone().slice(s![i..=i, .., ..]);
            let sum = batch_elem.sum().into_scalar::<f32>();

            // Check if dropped or scaled and passed through
            if sum.abs() < 1e-6 {
                batch_states.push("dropped");
            } else if (sum - 256.0).abs() < 1e-6 {
                // 8*16*2 (scale factor 2 for keep_prob=0.5)
                batch_states.push("scaled");
            } else {
                panic!("Unexpected output value: {sum}");
            }
        }

        // Likely to have at least one dropped and one passed through
        // (though not guaranteed due to probabilistic nature)
        println!("Batch element states: {batch_states:?}");
    }
}
