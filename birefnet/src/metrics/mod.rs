//! Metrics for BiRefNet training and evaluation.
//!
//! This module implements the evaluation metrics used in BiRefNet,
//! including F-measure, MAE (Mean Absolute Error), IoU, and other segmentation metrics.
//!
//! The implementation follows the original PyTorch BiRefNet evaluation metrics.

// Module declarations
pub mod aggregator;
pub mod e_measure;
pub mod f_measure;
pub mod input;
pub mod iou;
pub mod loss;
pub mod mae;
pub mod mse;
pub mod s_measure;
pub mod utils;
pub mod weighted_f_measure;

// Re-export everything for backward compatibility
pub use aggregator::*;
pub use e_measure::*;
pub use f_measure::*;
pub use input::*;
pub use iou::*;
pub use loss::*;
pub use mae::*;
pub use mse::*;
pub use s_measure::*;
pub use utils::*;
pub use weighted_f_measure::*;
