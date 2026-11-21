//! Array and statistical tensor utilities missing from core Burn APIs.

use burn::tensor::{Tensor, TensorData};

/// Computes an unweighted histogram of finite tensor elements across `bins` uniformly spaced bins.
///
/// If `range` is `None`, `[min, max]` is derived from the finite elements of `tensor`.
///
/// # Panics
///
/// Panics if `bins == 0`, if an explicit `range` has non-finite endpoints or `min > max`,
/// or if `tensor` has `D == 0`.
#[must_use]
pub fn histogram<const D: usize>(
    tensor: Tensor<D>,
    bins: usize,
    range: Option<(f64, f64)>,
) -> Tensor<1> {
    const { assert!(D >= 1, "histogram requires tensor rank >= 1") };
    assert!(bins > 0, "histogram requires bins > 0, got {bins}");

    let device = tensor.device();
    let values: Vec<f64> = tensor
        .flatten::<1>(0, D - 1)
        .into_data()
        .iter::<f64>()
        .collect();

    let (min_val, max_val) = if let Some((min_v, max_v)) = range {
        assert!(
            min_v.is_finite() && max_v.is_finite() && min_v <= max_v,
            "histogram range must be finite with min <= max, got ({min_v}, {max_v})"
        );
        (min_v, max_v)
    } else {
        let mut min_v = f64::INFINITY;
        let mut max_v = f64::NEG_INFINITY;
        for &val in &values {
            if val.is_finite() {
                min_v = min_v.min(val);
                max_v = max_v.max(val);
            }
        }
        if !min_v.is_finite() {
            return Tensor::zeros([bins], &device);
        }
        (min_v, max_v)
    };

    let mut counts = vec![0_u64; bins];
    let span = max_val - min_val;
    if span <= 0.0 {
        for &val in &values {
            if val.is_finite() && (val - min_val).abs() <= f64::EPSILON {
                counts[0] += 1;
            }
        }
    } else {
        #[expect(
            clippy::cast_precision_loss,
            reason = "histogram bin count fits in 52-bit mantissa"
        )]
        let bin_width = span / (bins as f64);
        for &val in &values {
            if val.is_finite() {
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "clamped non-negative index within [0, bins - 1]"
                )]
                let raw_idx = ((val - min_val) / bin_width).floor().max(0.0) as usize;
                counts[raw_idx.min(bins - 1)] += 1;
            }
        }
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "counts fit cleanly within f32 range before conversion"
    )]
    let hist_floats: Vec<f32> = counts.into_iter().map(|count| count as f32).collect();
    Tensor::<1>::from_data(TensorData::from(hist_floats.as_slice()), &device)
}

/// Computes the 1D cumulative sum along axis 0.
#[must_use]
pub fn cumsum_1d(tensor: Tensor<1>) -> Tensor<1> {
    tensor.cumsum(0)
}

/// Computes the 2D cumulative sum along axis 0.
#[must_use]
pub fn cumsum_2d_axis0(tensor: Tensor<2>) -> Tensor<2> {
    tensor.cumsum(0)
}

/// Reverses elements of a 1D tensor along axis 0.
#[must_use]
pub fn flip_1d(tensor: Tensor<1>) -> Tensor<1> {
    tensor.flip([0])
}

/// Counts non-zero elements in `tensor`.
#[must_use]
pub fn count_nonzero<const D: usize>(tensor: Tensor<D>) -> f64 {
    tensor.not_equal_elem(0).float().sum().into_scalar::<f64>()
}

/// Returns coordinates of all non-zero elements in row-major order.
#[must_use]
pub fn argwhere<const D: usize>(tensor: Tensor<D>) -> Vec<[usize; D]> {
    let values: Vec<i64> = tensor
        .not_equal_elem(0)
        .argwhere()
        .into_data()
        .iter::<i64>()
        .collect();
    values
        .as_chunks::<D>()
        .0
        .iter()
        .map(|chunk| {
            let mut coords = [0_usize; D];
            for (coord, &val) in coords.iter_mut().zip(chunk) {
                coords_assign(coord, val);
            }
            coords
        })
        .collect()
}

fn coords_assign(dst: &mut usize, val: i64) {
    *dst = usize::try_from(val).expect("argwhere coordinate must be non-negative");
}

/// Computes sample/population standard deviation with denominator `max(n - ddof, 0)`.
#[must_use]
pub fn std_with_ddof<const D: usize>(tensor: Tensor<D>, ddof: usize) -> f64 {
    let total_elements = tensor.shape().num_elements();
    let n = total_elements.saturating_sub(ddof);
    if n == 0 {
        return 0.0;
    }

    let mean = tensor.clone().mean().into_scalar::<f64>();
    let centered = tensor - mean;
    let sum_sq = centered.powi_scalar(2).sum().into_scalar::<f64>();
    #[expect(
        clippy::cast_precision_loss,
        reason = "element counts are within exact f64 range"
    )]
    {
        (sum_sq / (n as f64)).sqrt()
    }
}
