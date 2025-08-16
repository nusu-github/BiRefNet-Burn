//! Classification loss for auxiliary supervision.
//!
//! Provides classification loss for auxiliary outputs in segmentation models.
//! Each prediction level is evaluated against ground truth using cross-entropy loss.
//!
//! The total classification loss is computed as:
//! ```text
//! Loss = Σ(weight * CrossEntropy(pred_i, target)) for each prediction level
//! ```

use burn::{
    config::Config,
    module::{Content, DisplaySettings, Module, ModuleDisplay},
    nn::loss::{CrossEntropyLoss, CrossEntropyLossConfig, Reduction},
    tensor::{backend::Backend, Int, Tensor},
};

/// Configuration for creating a [Classification loss](ClassificationLoss).
#[derive(Config, Debug)]
pub struct ClassificationLossConfig {
    /// Weight factor applied to each cross-entropy loss. Default: 1.0
    #[config(default = 1.0)]
    pub weight: f64,
}

impl ClassificationLossConfig {
    /// Initialize [Classification loss](ClassificationLoss).
    pub fn init<B: Backend>(&self, device: &B::Device) -> ClassificationLoss<B> {
        self.assertions();
        ClassificationLoss {
            weight: self.weight,
            ce_loss: CrossEntropyLossConfig::new().init(device),
        }
    }

    fn assertions(&self) {
        assert!(
            self.weight > 0.0,
            "Weight for ClassificationLoss must be positive, got {}",
            self.weight
        );
    }
}

/// Classification loss for auxiliary supervision.
///
/// Combines cross-entropy losses from multiple prediction levels with configurable weighting.
/// Used for auxiliary supervision in segmentation models where intermediate features
/// are also trained for classification tasks.
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct ClassificationLoss<B: Backend> {
    /// Weight factor applied to each cross-entropy loss.
    pub weight: f64,
    /// Cross-entropy loss criterion.
    pub ce_loss: CrossEntropyLoss<B>,
}

impl<B: Backend> ModuleDisplay for ClassificationLoss<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("weight", &self.weight)
            .add("ce_loss", &self.ce_loss)
            .optional()
    }
}

impl<B: Backend> ClassificationLoss<B> {
    /// Create a new classification loss with default configuration.
    pub fn new(device: &B::Device) -> Self {
        ClassificationLossConfig::new().init(device)
    }

    /// Compute the criterion on multiple prediction levels with reduction.
    ///
    /// # Shapes
    ///
    /// - predictions: `&[Tensor<B, 2>]` where each tensor is `[batch_size, num_classes]`
    /// - targets: `[batch_size]`
    /// - output: `[1]`
    pub fn forward(
        &self,
        predictions: &[Tensor<B, 2>],
        targets: &Tensor<B, 1, Int>,
        reduction: Reduction,
    ) -> Tensor<B, 1> {
        let loss = self.forward_no_reduction(predictions, targets);
        match reduction {
            Reduction::Mean | Reduction::Auto => loss.mean(),
            Reduction::Sum => loss.sum(),
        }
    }

    /// Compute the criterion on multiple prediction levels without reduction.
    ///
    /// # Shapes
    ///
    /// - predictions: `&[Tensor<B, 2>]` where each tensor is `[batch_size, num_classes]`
    /// - targets: `[batch_size]`
    /// - output: `[num_prediction_levels]`
    pub fn forward_no_reduction(
        &self,
        predictions: &[Tensor<B, 2>],
        targets: &Tensor<B, 1, Int>,
    ) -> Tensor<B, 1> {
        self.assertions(predictions, targets);

        if predictions.is_empty() {
            return Tensor::zeros([1], &targets.device());
        }

        let mut losses = Vec::with_capacity(predictions.len());

        // Compute cross-entropy loss for each prediction level
        for pred in predictions {
            let loss = self.ce_loss.forward(pred.clone(), targets.clone());
            // Apply weight factor using mul_scalar
            let weighted_loss = loss.mul_scalar(self.weight);
            losses.push(weighted_loss);
        }

        // Concatenate all losses and sum them
        if losses.len() == 1 {
            losses[0].clone()
        } else {
            Tensor::cat(losses, 0).sum()
        }
    }

    fn assertions(&self, predictions: &[Tensor<B, 2>], targets: &Tensor<B, 1, Int>) {
        if predictions.is_empty() {
            return;
        }

        let target_batch_size = targets.dims()[0];

        for (i, pred) in predictions.iter().enumerate() {
            let pred_dims = pred.dims();
            let pred_batch_size = pred_dims[0];

            assert_eq!(
                pred_batch_size, target_batch_size,
                "Prediction[{i}] batch size ({pred_batch_size}) must match targets batch size ({target_batch_size})"
            );

            assert_eq!(
                pred_dims.len(),
                2,
                "Prediction[{i}] must be 2D tensor [batch_size, num_classes], got shape {pred_dims:?}"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{TensorData, Tolerance, Transaction};

    use super::*;
    use crate::tests::TestBackend;

    #[test]
    fn cls_loss_forward_single_prediction_computes_weighted_cross_entropy() {
        let device = Default::default();
        let loss = ClassificationLoss::<TestBackend>::new(&device);

        // Single prediction level
        let pred = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[2.0, 1.0, 0.5], [0.1, 3.0, 0.2]]),
            &device,
        );
        let targets = Tensor::<TestBackend, 1, Int>::from_data(TensorData::from([0, 1]), &device);

        let predictions = vec![pred.clone()];
        let result_mean = loss.forward(&predictions, &targets, Reduction::Mean);
        let result_no_reduction = loss.forward_no_reduction(&predictions, &targets);

        // Should compute weighted cross-entropy loss
        let expected_ce = loss.ce_loss.forward(pred, targets);
        let expected_weighted = expected_ce.mul_scalar(loss.weight);

        let [result_mean_data, result_no_reduction_data, expected_weighted_data1, expected_weighted_data2] =
            Transaction::default()
                .register(result_mean)
                .register(result_no_reduction)
                .register(expected_weighted.clone())
                .register(expected_weighted)
                .execute()
                .try_into()
                .expect("Correct amount of tensor data");

        result_mean_data.assert_approx_eq::<f32>(&expected_weighted_data1, Tolerance::default());
        result_no_reduction_data
            .assert_approx_eq::<f32>(&expected_weighted_data2, Tolerance::default());
    }

    #[test]
    fn cls_loss_forward_multiple_predictions_sums_weighted_losses() {
        let device = Default::default();
        let config = ClassificationLossConfig::new().with_weight(0.5);
        let loss = config.init(&device);

        // Multiple prediction levels
        let pred1 = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[2.0, 1.0], [0.1, 3.0]]),
            &device,
        );
        let pred2 = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.5, 2.0], [0.5, 2.5]]),
            &device,
        );
        let targets = Tensor::<TestBackend, 1, Int>::from_data(TensorData::from([0, 1]), &device);

        let predictions = vec![pred1.clone(), pred2.clone()];
        let result = loss.forward(&predictions, &targets, Reduction::Mean);

        // Expected: sum of weighted cross-entropy losses
        let ce1 = loss.ce_loss.forward(pred1, targets.clone());
        let ce2 = loss.ce_loss.forward(pred2, targets);
        let expected = (ce1 + ce2).mul_scalar(loss.weight);

        let [result_data, expected_data] = Transaction::default()
            .register(result)
            .register(expected)
            .execute()
            .try_into()
            .expect("Correct amount of tensor data");

        result_data.assert_approx_eq::<f32>(&expected_data, Tolerance::default());
    }

    #[test]
    fn cls_loss_forward_empty_predictions_returns_zero() {
        let device = Default::default();
        let loss = ClassificationLoss::<TestBackend>::new(&device);

        let targets = Tensor::<TestBackend, 1, Int>::from_data(TensorData::from([0, 1]), &device);

        let predictions: Vec<Tensor<TestBackend, 2>> = vec![];
        let result = loss.forward(&predictions, &targets, Reduction::Mean);

        // Should return zero loss for empty predictions
        let expected = TensorData::from([0.0]);
        result
            .into_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }

    #[test]
    fn cls_loss_auto_reduction_equals_mean_and_sum() {
        let device = Default::default();
        let loss = ClassificationLoss::<TestBackend>::new(&device);

        let pred = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[2.0, 1.0], [0.1, 3.0]]),
            &device,
        );
        let targets = Tensor::<TestBackend, 1, Int>::from_data(TensorData::from([0, 1]), &device);

        let predictions = vec![pred];
        let result_auto = loss.forward(&predictions, &targets, Reduction::Auto);
        let result_mean = loss.forward(&predictions, &targets, Reduction::Mean);
        let result_sum = loss.forward(&predictions, &targets, Reduction::Sum);

        let [result_auto_data, result_mean_data1, result_sum_data, result_mean_data2] =
            Transaction::default()
                .register(result_auto)
                .register(result_mean.clone())
                .register(result_sum)
                .register(result_mean)
                .execute()
                .try_into()
                .expect("Correct amount of tensor data");

        // Auto should equal Mean for single prediction
        result_auto_data.assert_approx_eq::<f32>(&result_mean_data1, Tolerance::default());

        // Sum should also equal Mean for single prediction
        result_sum_data.assert_approx_eq::<f32>(&result_mean_data2, Tolerance::default());
    }

    #[test]
    fn cls_loss_with_custom_weight_multiplies_default_result() {
        let device = Default::default();
        let config = ClassificationLossConfig::new().with_weight(2.0);
        let loss = config.init(&device);

        let pred = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[2.0, 1.0], [0.1, 3.0]]),
            &device,
        );
        let targets = Tensor::<TestBackend, 1, Int>::from_data(TensorData::from([0, 1]), &device);

        let predictions = vec![pred];
        let result = loss.forward(&predictions, &targets, Reduction::Mean);

        // Compare with default weight loss
        let default_loss = ClassificationLoss::<TestBackend>::new(&device);
        let default_result = default_loss.forward(&predictions, &targets, Reduction::Mean);

        let expected = default_result * 2.0;

        let [result_data, expected_data] = Transaction::default()
            .register(result)
            .register(expected)
            .execute()
            .try_into()
            .expect("Correct amount of tensor data");

        result_data.assert_approx_eq::<f32>(&expected_data, Tolerance::default());
    }

    #[test]
    #[should_panic = "Weight for ClassificationLoss must be positive"]
    fn cls_loss_config_negative_weight_panics() {
        let device = Default::default();
        let _loss = ClassificationLossConfig::new()
            .with_weight(-1.0)
            .init::<TestBackend>(&device);
    }

    #[test]
    #[should_panic = "Prediction[0] batch size (1) must match targets batch size (2)"]
    fn cls_loss_forward_mismatched_batch_sizes_panics() {
        let device = Default::default();
        let loss = ClassificationLoss::<TestBackend>::new(&device);

        let pred = Tensor::<TestBackend, 2>::from_data(TensorData::from([[2.0, 1.0]]), &device);
        let targets = Tensor::<TestBackend, 1, Int>::from_data(TensorData::from([0, 1]), &device);

        let predictions = vec![pred];
        let _result = loss.forward_no_reduction(&predictions, &targets);
    }

    #[test]
    fn cls_loss_display_shows_weight_parameter() {
        let device = Default::default();
        let config = ClassificationLossConfig::new().with_weight(0.5);
        let loss = config.init::<TestBackend>(&device);

        let display_str = format!("{loss}");
        assert!(display_str.contains("ClassificationLoss"));
        assert!(display_str.contains("weight: 0.5"));
    }
}
