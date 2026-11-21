//! Evaluation metrics for `BiRefNet` segmentation performance (`MAE`, `MSE`, `FMeasure`,
//! `BIoU`, `SMeasure`, `EMeasure`, and `WeightedFMeasure`).

mod aggregator;
mod biou;
mod common;
mod e_measure;
mod f_measure;
mod input;
mod mae;
mod mse;
mod s_measure;
mod utils;
mod weighted_f_measure;

#[doc(inline)]
pub use aggregator::MetricsAggregator;
#[doc(inline)]
pub use biou::{BIoUMetric, BIoUMetricConfig, calculate_biou};
#[doc(inline)]
pub use e_measure::{
    EMeasureMetric, EMeasureMetricConfig, EMeasureState, calculate_e_measure, e_measure_results,
};
#[doc(inline)]
pub use f_measure::{FMeasureCurves, FMeasureMetric, FMeasureMetricConfig, calculate_f_measure};
#[doc(inline)]
pub use input::{
    BIoUInput, EMeasureInput, FMeasureInput, MAEInput, MSEInput, SMeasureInput,
    WeightedFMeasureInput,
};
#[doc(inline)]
pub use mae::{MAEMetric, MAEMetricConfig, calculate_mae};
#[doc(inline)]
pub use mse::{MSEMetric, MSEMetricConfig, calculate_mse};
#[doc(inline)]
pub use s_measure::{SMeasureMetric, SMeasureMetricConfig, calculate_s_measure};
#[doc(inline)]
pub use utils::{AllMetricsResult, calculate_all_metrics};
#[doc(inline)]
pub use weighted_f_measure::{
    WeightedFMeasureMetric, WeightedFMeasureMetricConfig, calculate_weighted_f_measure,
};
