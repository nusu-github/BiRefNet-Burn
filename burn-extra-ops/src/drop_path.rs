//! # DropPath Regularization
//!
//! Implements the DropPath regularization technique, also known as stochastic depth.
//! During training, it randomly drops entire paths (sub-networks) and scales the
//! remaining ones, effectively preventing co-adaptation of parallel paths.

use burn::{prelude::*, tensor::Distribution};

/// Configuration for the `DropPath` module.
#[derive(Config, Debug)]
pub struct DropPathConfig {
    /// The probability of dropping a path.
    #[config(default = "0.0")]
    pub drop_prob: f64,
    /// Whether the module is in training mode.
    #[config(default = "false")]
    pub training: bool,
    /// Whether to scale the output by the keep probability.
    #[config(default = "true")]
    pub scale_by_keep: bool,
}

impl DropPathConfig {
    /// Initializes a new `DropPath` module.
    pub const fn init<B: Backend>(&self, _device: &B::Device) -> DropPath<B> {
        DropPath {
            drop_prob: self.drop_prob,
            training: self.training,
            scale_by_keep: self.scale_by_keep,
            _phantom: std::marker::PhantomData,
        }
    }
}

/// DropPath module.
#[derive(Module, Debug)]
pub struct DropPath<B: Backend> {
    drop_prob: f64,
    training: bool,
    scale_by_keep: bool,
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> DropPath<B> {
    /// Applies DropPath to the input tensor.
    ///
    /// If not in training mode or `drop_prob` is 0, it returns the input tensor unchanged.
    /// Otherwise, it randomly zeros out entire examples in the batch with probability `drop_prob`.
    /// The mask is generated only for the batch dimension and broadcasted to all other dimensions.
    ///
    /// # Shapes
    /// - input: `[batch_size, ..., channels]`
    /// - output: `[batch_size, ..., channels]`
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        if !self.training || self.drop_prob == 0.0 {
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
    use super::*;
    use burn::{
        backend::{ndarray::NdArrayDevice, NdArray},
        tensor::Tensor,
    };

    type TestBackend = NdArray;
    type TestDevice = NdArrayDevice;

    #[test]
    fn test_droppath_eval_mode() {
        let device = TestDevice::default();
        let config = DropPathConfig {
            drop_prob: 0.2,
            training: false, // Evaluation mode
            scale_by_keep: true,
        };
        let drop_path = config.init::<TestBackend>(&device);

        // Test input tensor
        let x = Tensor::<TestBackend, 4>::ones([2, 3, 4, 4], &device);

        // In evaluation mode, should return input unchanged
        let output = drop_path.forward(x.clone());

        // Verify input and output are equal
        let diff = (output - x).abs().sum();
        assert_eq!(
            diff.into_scalar(),
            0.0,
            "In evaluation mode, input and output should be equal"
        );
    }

    #[test]
    fn test_droppath_zero_prob() {
        let device = TestDevice::default();
        let config = DropPathConfig {
            drop_prob: 0.0, // Drop probability of 0
            training: true,
            scale_by_keep: true,
        };
        let drop_path = config.init::<TestBackend>(&device);

        // Test input tensor
        let x = Tensor::<TestBackend, 4>::ones([2, 3, 4, 4], &device);

        // With drop_prob=0, should return input unchanged
        let output = drop_path.forward(x.clone());

        // Verify input and output are equal
        let diff = (output - x).abs().sum();
        assert_eq!(
            diff.into_scalar(),
            0.0,
            "With drop_prob=0, input and output should be equal"
        );
    }

    #[test]
    fn test_droppath_shape_preservation() {
        let device = TestDevice::default();
        let config = DropPathConfig {
            drop_prob: 0.5,
            training: true,
            scale_by_keep: true,
        };
        let drop_path = config.init::<TestBackend>(&device);

        // Test with various shapes
        let shapes = vec![
            vec![2, 3, 4, 4],  // 4D: [batch, channels, height, width]
            vec![2, 196, 256], // 3D: [batch, sequence, dim]
            vec![2, 512],      // 2D: [batch, features]
        ];

        for shape in shapes {
            let dims = shape.len();
            match dims {
                2 => {
                    let x = Tensor::<TestBackend, 2>::ones([shape[0], shape[1]], &device);
                    let output = drop_path.forward(x.clone());
                    assert_eq!(output.dims(), x.dims(), "2D shape should be preserved");
                }
                3 => {
                    let x = Tensor::<TestBackend, 3>::ones([shape[0], shape[1], shape[2]], &device);
                    let output = drop_path.forward(x.clone());
                    assert_eq!(output.dims(), x.dims(), "3D shape should be preserved");
                }
                4 => {
                    let x = Tensor::<TestBackend, 4>::ones(
                        [shape[0], shape[1], shape[2], shape[3]],
                        &device,
                    );
                    let output = drop_path.forward(x.clone());
                    assert_eq!(output.dims(), x.dims(), "4D shape should be preserved");
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn test_droppath_training_mode_behavior() {
        let device = TestDevice::default();
        let config = DropPathConfig {
            drop_prob: 0.5,
            training: true,
            scale_by_keep: true,
        };
        let drop_path = config.init::<TestBackend>(&device);

        // Test with batch size 10 (for statistical verification)
        let batch_size = 10;
        let x = Tensor::<TestBackend, 4>::ones([batch_size, 3, 4, 4], &device);

        // Run multiple times to gather statistics
        let mut drop_counts = 0;
        let num_trials = 100;

        for _ in 0..num_trials {
            let output = drop_path.forward(x.clone());

            // Check if each batch element was dropped
            for i in 0..batch_size {
                let batch_output = output.clone().slice([i..i + 1, 0..3, 0..4, 0..4]);
                let sum = batch_output.sum().into_scalar();

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
    fn test_droppath_scaling() {
        let device = TestDevice::default();

        // Case with scale_by_keep = true
        let config_with_scale = DropPathConfig {
            drop_prob: 0.0, // Set to 0 to verify scaling
            training: true,
            scale_by_keep: true,
        };
        let drop_path_with_scale = config_with_scale.init::<TestBackend>(&device);

        // Case with scale_by_keep = false
        let config_no_scale = DropPathConfig {
            drop_prob: 0.0,
            training: true,
            scale_by_keep: false,
        };
        let drop_path_no_scale = config_no_scale.init::<TestBackend>(&device);

        let x = Tensor::<TestBackend, 2>::ones([2, 4], &device);

        let output_with_scale = drop_path_with_scale.forward(x.clone());
        let output_no_scale = drop_path_no_scale.forward(x.clone());

        // With drop_prob=0, both should return input unchanged
        assert_eq!(
            output_with_scale.sum().into_scalar(),
            x.clone().sum().into_scalar(),
            "With scale_by_keep=true and drop_prob=0"
        );
        assert_eq!(
            output_no_scale.sum().into_scalar(),
            x.sum().into_scalar(),
            "With scale_by_keep=false and drop_prob=0"
        );
    }

    #[test]
    fn test_droppath_batch_independence() {
        let device = TestDevice::default();
        let config = DropPathConfig {
            drop_prob: 0.5,
            training: true,
            scale_by_keep: true,
        };
        let drop_path = config.init::<TestBackend>(&device);

        // Test with batch size 4
        let x = Tensor::<TestBackend, 3>::ones([4, 8, 16], &device);
        let output = drop_path.forward(x);

        // Check the state of each batch element
        let mut batch_states = vec![];
        for i in 0..4 {
            let batch_elem = output.clone().slice([i..i + 1, 0..8, 0..16]);
            let sum = batch_elem.sum().into_scalar();

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
