//! Per-threshold metric curves and the shared 8-bit threshold histogram.

use std::ops::{AddAssign, DivAssign};

/// Metric values at the 256 binarization thresholds of an 8-bit prediction.
///
/// Index `i` holds the value for threshold `255 - i` (pixels with `value >= threshold` are
/// positive), the descending order produced by `py_sod_metrics`. Use [`at`](Self::at) to
/// address a threshold directly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThresholdCurve([f64; 256]);

impl Default for ThresholdCurve {
    fn default() -> Self {
        Self::zeros()
    }
}

impl ThresholdCurve {
    /// Number of thresholds (`0..=255`).
    pub const LEN: usize = 256;

    /// Returns an all-zero curve.
    #[must_use]
    pub const fn zeros() -> Self {
        Self([0.0; 256])
    }

    /// Builds a curve from `f(threshold)` for every threshold `0..=255`.
    #[must_use]
    pub fn from_thresholds(mut f: impl FnMut(u8) -> f64) -> Self {
        Self(std::array::from_fn(|idx| f(threshold_at(idx))))
    }

    /// Returns the value at binarization `threshold`.
    #[must_use]
    pub const fn at(&self, threshold: u8) -> f64 {
        self.0[255 - threshold as usize]
    }

    /// Returns the values ordered from threshold `255` down to `0`.
    #[must_use]
    pub const fn values(&self) -> &[f64; 256] {
        &self.0
    }

    /// Returns the mean over all thresholds (for example `meanFm`).
    #[must_use]
    pub fn mean(&self) -> f64 {
        self.0.iter().copied().fold(0.0, f64::algebraic_add) / 256.0
    }

    /// Returns the maximum over all thresholds (for example `maxFm`).
    #[must_use]
    pub fn max(&self) -> f64 {
        self.0.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    }
}

impl AddAssign<&Self> for ThresholdCurve {
    fn add_assign(&mut self, rhs: &Self) {
        for (dst, src) in self.0.iter_mut().zip(&rhs.0) {
            *dst += src;
        }
    }
}

impl DivAssign<f64> for ThresholdCurve {
    fn div_assign(&mut self, rhs: f64) {
        for value in &mut self.0 {
            *value /= rhs;
        }
    }
}

/// Threshold represented by curve index `idx`.
fn threshold_at(idx: usize) -> u8 {
    u8::try_from(255 - idx).unwrap_or_else(|_| panic!("curve index {idx} exceeds 255"))
}

/// Quantizes a `[0, 1]` value to 8 bits by truncation, like NumPy's `(x * 255).astype(uint8)`.
pub(crate) fn quantize(value: f64) -> u8 {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "`as` saturates to 0..=255 and truncates like NumPy's uint8 cast"
    )]
    let byte = (value * 255.0) as u8;
    byte
}

/// Counts of foreground / background pixels per 8-bit prediction value.
pub(crate) struct ThresholdHistogram {
    fg: [u64; 256],
    bg: [u64; 256],
}

/// Cumulative positives at each threshold, ordered like [`ThresholdCurve`].
pub(crate) struct CumulativeCounts {
    /// Foreground pixels predicted positive (true positives).
    pub(crate) fg: [u64; 256],
    /// Background pixels predicted positive (false positives).
    pub(crate) bg: [u64; 256],
}

impl ThresholdHistogram {
    pub(crate) fn new(values: impl IntoIterator<Item = (u8, bool)>) -> Self {
        let mut hist = Self {
            fg: [0; 256],
            bg: [0; 256],
        };
        for (value, is_fg) in values {
            let bin = if is_fg { &mut hist.fg } else { &mut hist.bg };
            bin[usize::from(value)] += 1;
        }
        hist
    }

    /// Counts of pixels `>= threshold` for every threshold from `255` down to `0`.
    pub(crate) fn cumulative(&self) -> CumulativeCounts {
        let mut out = CumulativeCounts {
            fg: [0; 256],
            bg: [0; 256],
        };
        let (mut fg, mut bg) = (0_u64, 0_u64);
        for idx in 0..256 {
            fg += self.fg[255 - idx];
            bg += self.bg[255 - idx];
            out.fg[idx] = fg;
            out.bg[idx] = bg;
        }
        out
    }
}

/// Exact `u64 -> f64` conversion for pixel counts.
pub(crate) fn count_f64(count: u64) -> f64 {
    #[expect(
        clippy::cast_precision_loss,
        reason = "pixel counts stay far below 2^53 and convert exactly"
    )]
    let value = count as f64;
    value
}
