//! Input structures for `BiRefNet` evaluation metrics.

use burn::prelude::*;

macro_rules! define_metric_input {
    ($name:ident, $doc:expr) => {
        #[doc = $doc]
        #[derive(Debug, Clone)]
        pub struct $name {
            /// Prediction tensor with shape `[batch_size, channels, height, width]`.
            pub predictions: Tensor<4>,
            /// Ground-truth tensor with shape `[batch_size, channels, height, width]`.
            pub targets: Tensor<4>,
        }

        impl $name {
            /// Creates a metric input from `[B, C, H, W]` predictions and targets.
            #[must_use]
            pub const fn new(predictions: Tensor<4>, targets: Tensor<4>) -> Self {
                Self {
                    predictions,
                    targets,
                }
            }
        }
    };
}

define_metric_input!(FMeasureInput, "Input pair for [`crate::FMeasureMetric`].");
define_metric_input!(MAEInput, "Input pair for [`crate::MAEMetric`].");
define_metric_input!(MSEInput, "Input pair for [`crate::MSEMetric`].");
define_metric_input!(BIoUInput, "Input pair for [`crate::BIoUMetric`].");
define_metric_input!(
    WeightedFMeasureInput,
    "Input pair for [`crate::WeightedFMeasureMetric`]."
);
define_metric_input!(SMeasureInput, "Input pair for [`crate::SMeasureMetric`].");
define_metric_input!(EMeasureInput, "Input pair for [`crate::EMeasureMetric`].");
