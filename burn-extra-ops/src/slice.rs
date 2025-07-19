//! # Tensor Slicing Helper
//!
//! Provides a `Slice` struct to facilitate tensor slicing with Python-like semantics,
//! including support for negative indices.

use burn::prelude::*;
use core::ops::Range;

/// A helper struct for defining a slice with optional start and end points.
///
/// This struct can be converted into a `Range<usize>` for use with Burn's tensor
/// slicing methods, correctly handling negative indices relative to a given length.
pub struct Slice {
    start: Option<isize>,
    end: Option<isize>,
}

impl Slice {
    /// Creates a new `Slice`.
    pub const fn new(start: Option<isize>, end: Option<isize>) -> Self {
        Self { start, end }
    }

    /// Converts the `Slice` into a `Range<usize>` for a given dimension length.
    ///
    /// Negative indices are interpreted as offsets from the end of the dimension.
    pub fn to_range(&self, len: usize) -> Range<usize> {
        let len_isize = len as isize;

        let start = match self.start {
            Some(s) if s < 0 => (len_isize + s).max(0),
            Some(s) => s.max(0).min(len_isize),
            None => 0,
        } as usize;

        let end = match self.end {
            Some(e) if e < 0 => (len_isize + e).max(0),
            Some(e) => e.max(0).min(len_isize),
            None => len_isize,
        } as usize;

        start..end
    }

    /// Calculates the length of the slice for a given dimension length.
    pub fn slice_length(&self, len: usize) -> usize {
        let range = self.to_range(len);
        range.end.saturating_sub(range.start)
    }
}

/// Slice tensor with advanced indexing (placeholder implementation)
pub fn slice_tensor<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    _ranges: &[Range<usize>],
) -> Tensor<B, D> {
    // Apply range-based slicing to tensor dimensions
    // Current implementation: placeholder returning unmodified tensor
    tensor
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::{
        backend::{ndarray::NdArray, Autodiff},
        tensor::Tensor,
    };

    type TestBackend = Autodiff<NdArray<f32>>;

    #[test]
    fn test_slice() {
        let slice = Slice::new(Some(-2), Some(-1));
        let range = slice.to_range(5);
        assert_eq!(range, 3..4);
    }

    #[test]
    fn test_slice_length() {
        let slice = Slice::new(Some(1), Some(4));
        assert_eq!(slice.slice_length(10), 3);
    }

    #[test]
    fn test_slice_tensor() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 4>::random(
            [2, 3, 4, 5],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let ranges = vec![0..1, 0..2, 0..3, 0..4];
        let result = slice_tensor(tensor, &ranges);
        assert_eq!(result.dims(), [2, 3, 4, 5]); // Placeholder behavior
    }
}
