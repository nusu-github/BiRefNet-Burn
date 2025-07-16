//! # Tensor Slicing Helper
//!
//! Provides a `Slice` struct to facilitate tensor slicing with Python-like semantics,
//! including support for negative indices.

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
