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
//!
//! ## Usage
//!
//! ```rust
//! use birefnet_metric::{FMeasureInput, FMeasureMetric};
//! use burn::{
//!     data::dataloader::Progress,
//!     prelude::*,
//!     train::metric::{Metric, MetricMetadata, Numeric},
//! };
//!
//! let device = Device::flex();
//!
//! // Create metric
//! let mut f_measure = FMeasureMetric::new();
//!
//! // Prepare input (4D tensors: [batch, channel, height, width], values in [0, 255])
//! let predictions = Tensor::<4>::full([1, 1, 32, 32], 255.0, &device);
//! let targets = Tensor::<4>::full([1, 1, 32, 32], 255.0, &device);
//!
//! // Calculate F-measure
//! let input = FMeasureInput::new(predictions, targets);
//! let metadata = MetricMetadata {
//!     progress: Progress {
//!         items_processed: 1,
//!         items_total: 1,
//!         unit: None,
//!     },
//!     iteration: None,
//!     lr: None,
//! };
//! f_measure.update(&input, &metadata).unwrap();
//!
//! println!("F-measure: {:?}", f_measure.value());
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
//! - Backend-agnostic: tensors carry their runtime `Device`
//! - Uses `Metric`, `Numeric`, and `NumericMetricState` traits
//! - Standardized 4D tensor inputs `[batch, channel, height, width]`
//! - Modular structure with each metric in separate module

// Module declarations
pub mod aggregator;
pub mod biou;
pub mod e_measure;
pub mod f_measure;
pub mod input;
pub mod mae;
pub mod mse;
pub mod s_measure;
pub mod utils;
pub mod weighted_f_measure;

// Re-export main types and traits
#[doc(inline)]
pub use biou::{BIoUMetric, BIoUMetricConfig};
#[doc(inline)]
pub use f_measure::{FMeasureMetric, FMeasureMetricConfig};
#[doc(inline)]
pub use input::{
    BIoUInput, EMeasureInput, FMeasureInput, MAEInput, MSEInput, SMeasureInput,
    WeightedFMeasureInput,
};
#[doc(inline)]
pub use mae::{MAEMetric, MAEMetricConfig};
#[doc(inline)]
pub use mse::{MSEMetric, MSEMetricConfig};
// Re-export lesser-used items with warning
#[deprecated(note = "S-measure is not yet fully implemented and tested")]
#[doc(inline)]
pub use s_measure::{
    SMeasureInput as SMeasureInputDeprecated, SMeasureMetric, SMeasureMetricConfig,
    calculate_s_measure,
};
#[doc(inline)]
pub use utils::{AllMetricsResult, calculate_all_metrics};
#[doc(inline)]
pub use weighted_f_measure::{
    WeightedFMeasureMetric, WeightedFMeasureMetricConfig, calculate_weighted_f_measure,
};
