#![doc = include_str!("../README.md")]

#[cfg(feature = "train")]
mod burn_metric;
mod curve;
mod error;
mod measures;
mod sample;
mod scores;

#[cfg(feature = "train")]
#[doc(inline)]
pub use burn_metric::{Measure, SegmentationInput, SegmentationMetric};
#[doc(inline)]
pub use curve::ThresholdCurve;
#[doc(inline)]
pub use error::MetricError;
#[doc(inline)]
pub use measures::{EMeasure, FMeasure};
#[doc(inline)]
pub use sample::{Sample, TARGET_THRESHOLD};
#[doc(inline)]
pub use scores::{EvaluationConfig, MetricsAggregator, Scores};
