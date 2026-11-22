//! Host-side prediction / target pairs prepared like `py_sod_metrics._prepare_data`.

use burn::tensor::{Tensor, Transaction};

use crate::{
    curve::{ThresholdHistogram, quantize},
    error::MetricError,
};

/// Targets strictly above this value are foreground: `gt > 128` on 8-bit masks.
pub const TARGET_THRESHOLD: f32 = 128.0 / 255.0;

/// One prediction / ground-truth pair, read to the host and prepared for evaluation.
///
/// Inputs are in `[0, 1]`. Targets are binarized at [`TARGET_THRESHOLD`]; predictions are
/// min-max normalized to `[0, 1]` unless constant, exactly as `py_sod_metrics` treats
/// 8-bit images. Every metric is computed on the host in `f64` from this one readback.
#[derive(Debug, Clone, PartialEq)]
pub struct Sample {
    pred: Vec<f64>,
    gt: Vec<bool>,
    dims: [usize; 2],
    fg_count: usize,
}

impl Sample {
    /// Prepares a sample from row-major `[height, width]` host buffers in `[0, 1]`.
    ///
    /// # Errors
    ///
    /// Returns [`MetricError`] if either buffer length differs from `height * width`, the
    /// image is empty, or a prediction is not finite.
    pub fn from_slices(
        predictions: &[f32],
        targets: &[f32],
        dims: [usize; 2],
    ) -> Result<Self, MetricError> {
        let [height, width] = dims;
        let len = height * width;
        if len == 0 || predictions.len() != len || targets.len() != len {
            return Err(MetricError::invalid_input(format!(
                "expected two non-empty {height}x{width} buffers, got {} predictions and {} targets",
                predictions.len(),
                targets.len()
            )));
        }
        if let Some(bad) = predictions.iter().find(|value| !value.is_finite()) {
            return Err(MetricError::invalid_input(format!(
                "predictions must be finite, found {bad}"
            )));
        }
        Ok(Self::prepare(
            predictions.iter().map(|&v| f64::from(v)).collect(),
            targets.iter().map(|&t| t > TARGET_THRESHOLD).collect(),
            dims,
        ))
    }

    /// Prepares a sample from 8-bit images, reproducing `py_sod_metrics` bit for bit.
    ///
    /// Predictions are divided by `255`; targets above `128` are foreground.
    ///
    /// # Errors
    ///
    /// Returns [`MetricError`] if either buffer length differs from `height * width` or the
    /// image is empty.
    pub fn from_u8(
        predictions: &[u8],
        targets: &[u8],
        dims: [usize; 2],
    ) -> Result<Self, MetricError> {
        let [height, width] = dims;
        let len = height * width;
        if len == 0 || predictions.len() != len || targets.len() != len {
            return Err(MetricError::invalid_input(format!(
                "expected two non-empty {height}x{width} buffers, got {} predictions and {} targets",
                predictions.len(),
                targets.len()
            )));
        }
        Ok(Self::prepare(
            predictions.iter().map(|&v| f64::from(v) / 255.0).collect(),
            targets.iter().map(|&t| t > 128).collect(),
            dims,
        ))
    }

    /// Min-max normalizes non-constant predictions (`(p - min) / (max - min)`, as NumPy does).
    fn prepare(mut pred: Vec<f64>, gt: Vec<bool>, dims: [usize; 2]) -> Self {
        let (min, max) = pred
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
                (lo.min(v), hi.max(v))
            });
        if max > min {
            let range = max - min;
            for value in &mut pred {
                *value = (*value - min) / range;
            }
        }
        let fg_count = gt.iter().filter(|&&is_fg| is_fg).count();
        Self {
            pred,
            gt,
            dims,
            fg_count,
        }
    }

    /// Reads a `[height, width]` prediction / target pair in one device transaction.
    ///
    /// # Errors
    ///
    /// Returns [`MetricError`] if the shapes differ, the image is empty, or the readback fails.
    pub fn from_tensors(predictions: &Tensor<2>, targets: &Tensor<2>) -> Result<Self, MetricError> {
        let [height, width] = predictions.dims();
        let batch = Self::batch_from_tensors(
            &predictions.clone().reshape([1, 1, height, width]),
            &targets
                .clone()
                .reshape([1, 1, targets.dims()[0], targets.dims()[1]]),
        )?;
        batch
            .into_iter()
            .next()
            .ok_or_else(|| MetricError::invalid_input("empty batch"))
    }

    /// Reads a `[B, 1, H, W]` batch in a single device transaction and prepares every sample.
    ///
    /// # Errors
    ///
    /// Returns [`MetricError`] if the shapes differ, there is more than one channel, an image is
    /// empty, or the readback fails.
    pub fn batch_from_tensors(
        predictions: &Tensor<4>,
        targets: &Tensor<4>,
    ) -> Result<Vec<Self>, MetricError> {
        let dims = predictions.dims();
        let [batch, channels, height, width] = dims;
        if dims != targets.dims() || channels != 1 {
            return Err(MetricError::invalid_input(format!(
                "predictions {dims:?} and targets {:?} must share one [B, 1, H, W] shape",
                targets.dims()
            )));
        }
        if batch == 0 {
            return Ok(Vec::new());
        }

        let [pred_data, gt_data]: [_; 2] = Transaction::default()
            .register(predictions.clone())
            .register(targets.clone())
            .try_execute()
            .map_err(burn::tensor::TensorReadError::from)?
            .try_into()
            .map_err(|data: Vec<_>| {
                MetricError::invalid_input(format!(
                    "transaction returned {} tensors, expected 2",
                    data.len()
                ))
            })?;
        let to_f32 = |data: burn::tensor::TensorData| {
            data.try_into_vec_as::<f32>()
                .map_err(|err| MetricError::from(burn::tensor::TensorReadError::from(err)))
        };
        let (pred_values, gt_values) = (to_f32(pred_data)?, to_f32(gt_data)?);

        let plane = height * width;
        pred_values
            .chunks_exact(plane.max(1))
            .zip(gt_values.chunks_exact(plane.max(1)))
            .map(|(pred, gt)| Self::from_slices(pred, gt, [height, width]))
            .collect()
    }

    /// Returns the `[height, width]` image size.
    #[must_use]
    pub const fn dims(&self) -> [usize; 2] {
        self.dims
    }

    /// Returns the prepared predictions in `[0, 1]`, row-major.
    #[must_use]
    pub fn predictions(&self) -> &[f64] {
        &self.pred
    }

    /// Returns the binarized targets, row-major.
    #[must_use]
    pub fn targets(&self) -> &[bool] {
        &self.gt
    }

    /// Returns the number of foreground target pixels.
    #[must_use]
    pub const fn foreground_count(&self) -> usize {
        self.fg_count
    }

    pub(crate) const fn len(&self) -> usize {
        self.dims[0] * self.dims[1]
    }

    /// Mean absolute error `mean(|pred - gt|)`.
    #[must_use]
    pub fn mae(&self) -> f64 {
        self.mean_of(|p, g| (p - g).abs())
    }

    /// Mean squared error `mean((pred - gt)^2)`.
    #[must_use]
    pub fn mse(&self) -> f64 {
        self.mean_of(|p, g| (p - g) * (p - g))
    }

    fn mean_of(&self, f: impl Fn(f64, f64) -> f64) -> f64 {
        let sum = self
            .pred
            .iter()
            .zip(&self.gt)
            .map(|(&p, &g)| f(p, f64::from(u8::from(g))))
            .fold(0.0, f64::algebraic_add);
        sum / usize_f64(self.len())
    }

    /// `min(2 * mean(pred), 1)`, the adaptive binarization threshold.
    pub(crate) fn adaptive_threshold(&self) -> f64 {
        let mean = self.pred.iter().copied().fold(0.0, f64::algebraic_add) / usize_f64(self.len());
        (2.0 * mean).min(1.0)
    }

    /// `(true positives, false positives)` when predictions `>= threshold` are positive.
    pub(crate) fn confusion_at(&self, threshold: f64) -> (u64, u64) {
        self.pred
            .iter()
            .zip(&self.gt)
            .filter(|&(&p, _)| p >= threshold)
            .fold(
                (0, 0),
                |(tp, fp), (_, &g)| if g { (tp + 1, fp) } else { (tp, fp + 1) },
            )
    }

    pub(crate) fn histogram(&self) -> ThresholdHistogram {
        ThresholdHistogram::new(
            self.pred
                .iter()
                .zip(&self.gt)
                .map(|(&p, &g)| (quantize(p), g)),
        )
    }
}

/// Exact `usize -> f64` conversion for pixel counts.
pub(crate) fn usize_f64(value: usize) -> f64 {
    #[expect(
        clippy::cast_precision_loss,
        reason = "pixel and sample counts stay far below 2^53 and convert exactly"
    )]
    let value = value as f64;
    value
}
