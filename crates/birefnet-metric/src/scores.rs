//! Full evaluation of samples and dataset-level aggregation.

use std::fmt;

use burn::tensor::Tensor;

use crate::{
    curve::ThresholdCurve,
    error::MetricError,
    measures::{EMeasure, FMeasure},
    sample::{Sample, usize_f64},
};

/// Parameters of a full evaluation; defaults match the BiRefNet evaluation scripts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvaluationConfig {
    f_beta_sq: f64,
    weighted_f_beta_sq: f64,
    s_alpha: f64,
    biou_dilation_ratio: f64,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl EvaluationConfig {
    /// `β² = 0.3` for F-measure, `β² = 1` for weighted F-measure, `α = 0.5` for S-measure,
    /// and a boundary width of 2% of the image diagonal for BIoU.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            f_beta_sq: 0.3,
            weighted_f_beta_sq: 1.0,
            s_alpha: 0.5,
            biou_dilation_ratio: 0.02,
        }
    }

    /// Sets `β²` of the (adaptive and curve) F-measure.
    ///
    /// # Errors
    ///
    /// Returns [`MetricError`] unless `beta_sq` is finite and non-negative.
    pub fn with_f_beta_sq(self, beta_sq: f64) -> Result<Self, MetricError> {
        check_beta_sq(beta_sq)?;
        Ok(Self {
            f_beta_sq: beta_sq,
            ..self
        })
    }

    /// Sets `β²` of the weighted F-measure.
    ///
    /// # Errors
    ///
    /// Returns [`MetricError`] unless `beta_sq` is finite and non-negative.
    pub fn with_weighted_f_beta_sq(self, beta_sq: f64) -> Result<Self, MetricError> {
        check_beta_sq(beta_sq)?;
        Ok(Self {
            weighted_f_beta_sq: beta_sq,
            ..self
        })
    }

    /// Sets the object/region balance `α` of the S-measure.
    ///
    /// # Errors
    ///
    /// Returns [`MetricError`] unless `alpha` lies in `[0, 1]`.
    pub fn with_s_alpha(self, alpha: f64) -> Result<Self, MetricError> {
        check_alpha(alpha)?;
        Ok(Self {
            s_alpha: alpha,
            ..self
        })
    }

    /// Sets the BIoU boundary width as a fraction of the image diagonal.
    ///
    /// # Errors
    ///
    /// Returns [`MetricError`] unless `ratio` is finite and positive.
    pub fn with_biou_dilation_ratio(self, ratio: f64) -> Result<Self, MetricError> {
        check_dilation_ratio(ratio)?;
        Ok(Self {
            biou_dilation_ratio: ratio,
            ..self
        })
    }

    /// Returns `β²` of the F-measure.
    #[must_use]
    pub const fn f_beta_sq(&self) -> f64 {
        self.f_beta_sq
    }

    /// Returns `β²` of the weighted F-measure.
    #[must_use]
    pub const fn weighted_f_beta_sq(&self) -> f64 {
        self.weighted_f_beta_sq
    }

    /// Returns `α` of the S-measure.
    #[must_use]
    pub const fn s_alpha(&self) -> f64 {
        self.s_alpha
    }

    /// Returns the BIoU boundary width ratio.
    #[must_use]
    pub const fn biou_dilation_ratio(&self) -> f64 {
        self.biou_dilation_ratio
    }
}

pub(crate) fn check_beta_sq(beta_sq: f64) -> Result<(), MetricError> {
    if beta_sq.is_finite() && beta_sq >= 0.0 {
        Ok(())
    } else {
        Err(MetricError::invalid_config(format!(
            "beta_sq must be finite and non-negative, got {beta_sq}"
        )))
    }
}

pub(crate) fn check_alpha(alpha: f64) -> Result<(), MetricError> {
    if (0.0..=1.0).contains(&alpha) {
        Ok(())
    } else {
        Err(MetricError::invalid_config(format!(
            "alpha must be in [0, 1], got {alpha}"
        )))
    }
}

pub(crate) fn check_dilation_ratio(ratio: f64) -> Result<(), MetricError> {
    if ratio.is_finite() && ratio > 0.0 {
        Ok(())
    } else {
        Err(MetricError::invalid_config(format!(
            "dilation_ratio must be finite and positive, got {ratio}"
        )))
    }
}

/// Every metric of one sample, or the average over a set of samples.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Scores {
    /// Mean absolute error.
    pub mae: f64,
    /// Mean squared error.
    pub mse: f64,
    /// Structure measure.
    pub s_measure: f64,
    /// Weighted F-measure.
    pub weighted_f_measure: f64,
    /// Adaptive F-measure and its threshold curves.
    pub f_measure: FMeasure,
    /// Adaptive E-measure and its threshold curve.
    pub e_measure: EMeasure,
    /// Boundary IoU at each threshold.
    pub biou: ThresholdCurve,
}

impl Sample {
    /// Computes every metric of this sample.
    #[must_use]
    pub fn scores(&self, config: &EvaluationConfig) -> Scores {
        Scores {
            mae: self.mae(),
            mse: self.mse(),
            s_measure: self.s_measure(config.s_alpha),
            weighted_f_measure: self.weighted_f_measure(config.weighted_f_beta_sq),
            f_measure: self.f_measure(config.f_beta_sq),
            e_measure: self.e_measure(),
            biou: self.biou(config.biou_dilation_ratio),
        }
    }
}

impl Scores {
    fn accumulate(&mut self, other: &Self) {
        self.mae += other.mae;
        self.mse += other.mse;
        self.s_measure += other.s_measure;
        self.weighted_f_measure += other.weighted_f_measure;
        self.f_measure.adaptive += other.f_measure.adaptive;
        self.f_measure.curve += &other.f_measure.curve;
        self.f_measure.precision += &other.f_measure.precision;
        self.f_measure.recall += &other.f_measure.recall;
        self.e_measure.adaptive += other.e_measure.adaptive;
        self.e_measure.curve += &other.e_measure.curve;
        self.biou += &other.biou;
    }

    fn divided_by(mut self, count: f64) -> Self {
        self.mae /= count;
        self.mse /= count;
        self.s_measure /= count;
        self.weighted_f_measure /= count;
        self.f_measure.adaptive /= count;
        self.f_measure.curve /= count;
        self.f_measure.precision /= count;
        self.f_measure.recall /= count;
        self.e_measure.adaptive /= count;
        self.e_measure.curve /= count;
        self.biou /= count;
        self
    }
}

impl fmt::Display for Scores {
    /// Renders the DIS5K score table order: `maxFm wFmeasure MAE Smeasure meanEm maxEm meanFm
    /// adpEm adpFm maxBIoU meanBIoU`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "maxFm={:.3} wFmeasure={:.3} MAE={:.3} Smeasure={:.3} meanEm={:.3} maxEm={:.3} \
             meanFm={:.3} adpEm={:.3} adpFm={:.3} maxBIoU={:.3} meanBIoU={:.3}",
            self.f_measure.curve.max(),
            self.weighted_f_measure,
            self.mae,
            self.s_measure,
            self.e_measure.curve.mean(),
            self.e_measure.curve.max(),
            self.f_measure.curve.mean(),
            self.e_measure.adaptive,
            self.f_measure.adaptive,
            self.biou.max(),
            self.biou.mean(),
        )
    }
}

/// Accumulates [`Scores`] over a dataset with constant memory.
///
/// Like the reference evaluation, every sample has equal weight and curves are averaged per
/// threshold before taking maxima.
#[derive(Debug, Clone, Default)]
pub struct MetricsAggregator {
    config: EvaluationConfig,
    sum: Scores,
    count: usize,
}

impl MetricsAggregator {
    /// Creates an empty aggregator evaluating with `config`.
    #[must_use]
    pub fn new(config: EvaluationConfig) -> Self {
        Self {
            config,
            sum: Scores::default(),
            count: 0,
        }
    }

    /// Returns the evaluation parameters.
    #[must_use]
    pub const fn config(&self) -> &EvaluationConfig {
        &self.config
    }

    /// Adds already computed per-sample scores.
    pub fn push(&mut self, scores: &Scores) {
        self.sum.accumulate(scores);
        self.count += 1;
    }

    /// Evaluates a `[B, 1, H, W]` batch in `[0, 1]` (one device readback) and adds every sample.
    ///
    /// Nothing is added if the batch cannot be read or validated.
    ///
    /// # Errors
    ///
    /// Returns [`MetricError`] if the shapes are invalid or the readback fails.
    pub fn update(
        &mut self,
        predictions: &Tensor<4>,
        targets: &Tensor<4>,
    ) -> Result<usize, MetricError> {
        let scores: Vec<Scores> = Sample::batch_from_tensors(predictions, targets)?
            .iter()
            .map(|sample| sample.scores(&self.config))
            .collect();
        for sample_scores in &scores {
            self.push(sample_scores);
        }
        Ok(scores.len())
    }

    /// Returns the number of accumulated samples.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.count
    }

    /// Returns `true` if no sample has been accumulated.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Returns the average scores, or `None` before the first sample.
    #[must_use]
    pub fn summary(&self) -> Option<Scores> {
        (self.count > 0).then(|| self.sum.divided_by(usize_f64(self.count)))
    }

    /// Clears all accumulated samples, keeping the configuration.
    pub fn reset(&mut self) {
        self.sum = Scores::default();
        self.count = 0;
    }
}
