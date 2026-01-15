//! # BiRefNet Metrics
//!
//! Custom evaluation metrics for BiRefNet (Bilateral Reference Network) implemented in Rust using the Burn framework.
//!
//! This crate provides evaluation metrics for tasks like Dichotomous Image Segmentation (DIS),
//! Camouflaged Object Detection (COD), High-Resolution Salient Object Detection (HRSOD),
//! and general image matting. The implementation is based on the original Python evaluation
//! metrics from BiRefNet/evaluation/metrics.py.
//!
//! ## ⚠️ Status: Work in Progress
//!
//! **This crate is currently under development and has not been thoroughly tested.**
//!
//! - ✅ **Compiles successfully** - All Rust compilation errors have been resolved
//! - ⚠️ **Functionality unverified** - Most metrics haven't been tested against the Python reference
//! - ⚠️ **Not used in training/inference** - This is evaluation-only code, not part of the model pipeline
//! - 🔻 **Low priority** - Since it's not used in core BiRefNet functionality
//!
//! ## Implemented Metrics
//!
//! - [`FMeasureMetric`]: Adaptive and changeable F-measure with precision-recall curves
//! - [`MAEMetric`]: Mean Absolute Error with data preprocessing
//! - [`MSEMetric`]: Mean Squared Error with normalization
//! - [`BIoUMetric`]: Boundary IoU replacing standard IoU
//! - [`WeightedFMeasureMetric`]: Distance-weighted F-measure for boundary evaluation
//! - [`LossMetric`]: Simple loss value tracking
//!
//! ## Usage
//!
//! ```rust,ignore
//! use birefnet_metric::{FMeasureInput, FMeasureMetric};
//! use burn::prelude::*;
//!
//! # fn example<B: burn::tensor::backend::Backend>() {
//! // Create metric
//! let mut f_measure = FMeasureMetric::new();
//!
//! // Prepare input (4D tensors: [batch, channel, height, width])
//! let predictions = Tensor::<B, 4>::zeros([1, 1, 256, 256], &Default::default());
//! let targets = Tensor::<B, 4>::zeros([1, 1, 256, 256], &Default::default());
//!
//! // Calculate F-measure
//! let input = FMeasureInput::new(predictions, targets);
//! let metadata = burn::train::metric::MetricMetadata::fake();
//! f_measure.update(&input, &metadata);
//!
//! println!("F-measure: {}", f_measure.value());
//! # }
//! ```
//!
//! ## Data Processing
//!
//! All metrics implement the `_prepare_data` function following the Python reference:
//!
//! 1. **Ground truth binarization**: `gt = gt > 128`
//! 2. **Prediction normalization**: `pred = pred / 255`
//! 3. **Range normalization**: If there's variation, normalize to [0,1]
//!
//! ## Architecture
//!
//! The crate follows Burn's metric patterns:
//! - Generic over `Backend` for hardware portability
//! - Uses `Metric`, `Numeric`, and `NumericMetricState` traits
//! - Standardized 4D tensor inputs `[batch, channel, height, width]`
//! - Modular structure with each metric in separate module

// Module declarations
pub mod aggregator;
pub mod biou;
pub mod e_measure;
pub mod f_measure;
pub mod input;
pub mod loss;
pub mod mae;
pub mod mse;
pub mod s_measure;
pub mod utils;
pub mod weighted_f_measure;

// Re-export main types and traits
pub use biou::{BIoUMetric, BIoUMetricConfig};
pub use f_measure::{FMeasureMetric, FMeasureMetricConfig};
pub use input::*;
pub use loss::LossMetric;
pub use mae::{MAEMetric, MAEMetricConfig};
pub use mse::{MSEMetric, MSEMetricConfig};
// Re-export lesser-used items with warning
#[deprecated(note = "S-measure is not yet fully implemented and tested")]
pub use s_measure::*;
pub use utils::*;
pub use weighted_f_measure::{WeightedFMeasureMetric, WeightedFMeasureMetricConfig};

#[cfg(test)]
mod tests {
    use burn::backend::Cpu;

    pub type TestBackend = Cpu;
}
