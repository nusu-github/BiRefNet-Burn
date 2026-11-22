//! Burn `Metric` adapters for training and validation loops (`train` feature).

use std::sync::Arc;

use burn::{
    prelude::*,
    tensor::TensorReadError,
    train::metric::{
        Metric, MetricAttributes, MetricMetadata, MetricName, Numeric, NumericAttributes,
        NumericEntry, SerializedEntry,
        state::{FormatOptions, NumericMetricState},
    },
};

use crate::{
    error::MetricError,
    sample::{Sample, usize_f64},
    scores::{check_alpha, check_beta_sq, check_dilation_ratio},
};

/// Predictions and targets in `[0, 1]`, shaped `[batch_size, 1, height, width]`.
///
/// Model outputs implement `Adaptor<SegmentationInput>` to feed every [`SegmentationMetric`].
#[derive(Debug, Clone)]
pub struct SegmentationInput {
    /// Predicted foreground probabilities.
    pub predictions: Tensor<4>,
    /// Ground-truth masks.
    pub targets: Tensor<4>,
}

impl SegmentationInput {
    /// Creates a metric input from `[B, 1, H, W]` predictions and targets.
    #[must_use]
    pub const fn new(predictions: Tensor<4>, targets: Tensor<4>) -> Self {
        Self {
            predictions,
            targets,
        }
    }
}

/// Per-sample measure tracked by a [`SegmentationMetric`].
///
/// Only measures that average linearly over samples are offered; curve maxima such as `maxFm`
/// need [`MetricsAggregator`](crate::MetricsAggregator).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Measure {
    /// Mean absolute error (lower is better).
    Mae,
    /// Mean squared error (lower is better).
    Mse,
    /// Adaptive-threshold F-measure with the given `β²`.
    AdaptiveFMeasure {
        /// `β²` (`0.3` in salient object detection).
        beta_sq: f64,
    },
    /// F-measure averaged over all 256 thresholds with the given `β²`.
    MeanFMeasure {
        /// `β²` (`0.3` in salient object detection).
        beta_sq: f64,
    },
    /// Adaptive-threshold E-measure.
    AdaptiveEMeasure,
    /// E-measure averaged over all 256 thresholds.
    MeanEMeasure,
    /// Structure measure with object/region balance `alpha`.
    SMeasure {
        /// Object/region balance (`0.5` by default).
        alpha: f64,
    },
    /// Weighted F-measure with the given `β²`.
    WeightedFMeasure {
        /// `β²` (`1.0` in the reference evaluation).
        beta_sq: f64,
    },
    /// Boundary IoU averaged over all 256 thresholds.
    MeanBiou {
        /// Boundary width as a fraction of the image diagonal (`0.02` by default).
        dilation_ratio: f64,
    },
}

impl Measure {
    fn validate(self) -> Result<Self, MetricError> {
        match self {
            Self::AdaptiveFMeasure { beta_sq }
            | Self::MeanFMeasure { beta_sq }
            | Self::WeightedFMeasure { beta_sq } => check_beta_sq(beta_sq)?,
            Self::SMeasure { alpha } => check_alpha(alpha)?,
            Self::MeanBiou { dilation_ratio } => check_dilation_ratio(dilation_ratio)?,
            Self::Mae | Self::Mse | Self::AdaptiveEMeasure | Self::MeanEMeasure => {}
        }
        Ok(self)
    }

    /// Unique, parameterized display name (Burn keys metric state by name).
    fn name(self) -> String {
        match self {
            Self::Mae => "MAE".to_owned(),
            Self::Mse => "MSE".to_owned(),
            Self::AdaptiveFMeasure { beta_sq } => format!("Adaptive F-measure (beta^2={beta_sq})"),
            Self::MeanFMeasure { beta_sq } => format!("Mean F-measure (beta^2={beta_sq})"),
            Self::AdaptiveEMeasure => "Adaptive E-measure".to_owned(),
            Self::MeanEMeasure => "Mean E-measure".to_owned(),
            Self::SMeasure { alpha } => format!("S-measure (alpha={alpha})"),
            Self::WeightedFMeasure { beta_sq } => {
                format!("Weighted F-measure (beta^2={beta_sq})")
            }
            Self::MeanBiou { dilation_ratio } => format!("Mean BIoU (dilation={dilation_ratio})"),
        }
    }

    const fn higher_is_better(self) -> bool {
        !matches!(self, Self::Mae | Self::Mse)
    }

    fn evaluate(self, sample: &Sample) -> f64 {
        match self {
            Self::Mae => sample.mae(),
            Self::Mse => sample.mse(),
            Self::AdaptiveFMeasure { beta_sq } => sample.f_measure(beta_sq).adaptive,
            Self::MeanFMeasure { beta_sq } => sample.f_measure(beta_sq).curve.mean(),
            Self::AdaptiveEMeasure => sample.e_measure().adaptive,
            Self::MeanEMeasure => sample.e_measure().curve.mean(),
            Self::SMeasure { alpha } => sample.s_measure(alpha),
            Self::WeightedFMeasure { beta_sq } => sample.weighted_f_measure(beta_sq),
            Self::MeanBiou { dilation_ratio } => sample.biou(dilation_ratio).mean(),
        }
    }
}

/// Burn [`Metric`] reporting the per-sample average of one [`Measure`].
///
/// Each update reads the whole batch back from the device in a single transaction.
#[derive(Clone)]
pub struct SegmentationMetric {
    measure: Measure,
    name: MetricName,
    state: NumericMetricState,
}

impl std::fmt::Debug for SegmentationMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SegmentationMetric")
            .field("measure", &self.measure)
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl SegmentationMetric {
    /// Creates a metric tracking `measure`.
    ///
    /// # Errors
    ///
    /// Returns [`MetricError`] if a measure parameter is out of range.
    pub fn new(measure: Measure) -> Result<Self, MetricError> {
        let measure = measure.validate()?;
        Ok(Self {
            measure,
            name: Arc::new(measure.name()),
            state: NumericMetricState::default(),
        })
    }

    /// Returns the tracked measure.
    #[must_use]
    pub const fn measure(&self) -> Measure {
        self.measure
    }

    fn format(&self) -> FormatOptions {
        FormatOptions::new(Arc::clone(&self.name)).precision(5)
    }
}

impl Metric for SegmentationMetric {
    type Input = SegmentationInput;

    fn name(&self) -> MetricName {
        Arc::clone(&self.name)
    }

    /// # Panics
    ///
    /// Panics if predictions and targets do not share one `[B, 1, H, W]` shape, which is a
    /// bug in the model output adaptor rather than a recoverable condition.
    fn update(
        &mut self,
        item: &Self::Input,
        _metadata: &MetricMetadata,
    ) -> Result<SerializedEntry, TensorReadError> {
        let samples = match Sample::batch_from_tensors(&item.predictions, &item.targets) {
            Ok(samples) => samples,
            Err(err) => match err.into_tensor_read() {
                Ok(read_error) => return Err(read_error),
                Err(invalid) => panic!("{} received invalid inputs: {invalid}", self.name),
            },
        };
        if samples.is_empty() {
            return Ok(SerializedEntry::not_available(None));
        }

        let total: f64 = samples
            .iter()
            .map(|sample| self.measure.evaluate(sample))
            .sum();
        self.state
            .update(total / usize_f64(samples.len()), samples.len());
        Ok(self.state.compute_update(self.format()))
    }

    fn compute(&mut self) -> Result<SerializedEntry, TensorReadError> {
        Ok(self.state.compute_final(self.format()))
    }

    fn attributes(&self) -> MetricAttributes {
        NumericAttributes {
            unit: None,
            higher_is_better: self.measure.higher_is_better(),
        }
        .into()
    }

    fn clear(&mut self) {
        self.state.reset();
    }
}

impl Numeric for SegmentationMetric {
    fn value(&self) -> Option<NumericEntry> {
        Some(self.state.current_value())
    }

    fn running_value(&self) -> Option<NumericEntry> {
        Some(self.state.running_value())
    }

    fn final_value(&self) -> NumericEntry {
        self.state.final_value()
    }
}
