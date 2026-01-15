//! Basic array operations for metrics implementation
//!
//! This module provides essential array operations that are missing from Burn's core
//! but required for implementing computer vision evaluation metrics.

use burn::tensor::{ElementConversion, Tensor, backend::Backend, s};

/// Compute histogram of tensor values
///
/// # Arguments  
/// * `tensor` - Input tensor of any dimension
/// * `bins` - Number of histogram bins
/// * `range` - Value range as (min, max) tuple. If None, uses tensor min/max
///
/// # Returns
/// Histogram counts as 1D tensor of length `bins`
pub fn histogram<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    bins: usize,
    range: Option<(f64, f64)>,
) -> Tensor<B, 1> {
    let device = tensor.device();

    // Get value range
    let (min_val, max_val) = if let Some((min, max)) = range {
        (min, max)
    } else {
        let flat = tensor.clone().flatten::<1>(0, D - 1);
        let min_val = flat.clone().min().into_scalar().elem::<f64>();
        let max_val = flat.max().into_scalar().elem::<f64>();
        (min_val, max_val)
    };

    // Create bin edges
    let bin_width = (max_val - min_val) / bins as f64;
    let mut hist_counts = vec![0_i64; bins];

    // Flatten tensor and convert to data
    let flat_tensor = tensor.flatten::<1>(0, D - 1);
    let data = flat_tensor.into_data().convert::<f64>();
    let values = data.as_slice::<f64>().unwrap();

    // Count values in each bin
    for &value in values {
        if value.is_finite() {
            let bin_idx = ((value - min_val) / bin_width).floor() as usize;
            let bin_idx = bin_idx.min(bins - 1); // Clamp to valid range
            hist_counts[bin_idx] += 1;
        }
    }

    // Convert to tensor
    let hist_data: Vec<f64> = hist_counts.into_iter().map(|x| x as f64).collect();
    Tensor::from_floats(hist_data.as_slice(), &device)
}

/// Compute cumulative sum along specified dimension (specialized for 1D)
///
/// # Arguments
/// * `tensor` - Input 1D tensor
///
/// # Returns  
/// 1D tensor with cumulative sums
pub fn cumsum_1d<B: Backend>(tensor: Tensor<B, 1>) -> Tensor<B, 1> {
    let [size] = tensor.dims();

    if size <= 1 {
        return tensor;
    }

    let mut result = tensor.clone();

    // Iterative accumulation for 1D case
    for i in 1..size {
        let prev_value = result.clone().narrow(0, i - 1, 1);
        let curr_value = tensor.clone().narrow(0, i, 1);
        let accumulated = prev_value + curr_value;
        result = result.slice_assign(s![i..i + 1], accumulated);
    }

    result
}

/// Compute cumulative sum along axis 0 for 2D tensors
///
/// # Arguments
/// * `tensor` - Input 2D tensor
///
/// # Returns  
/// 2D tensor with cumulative sums along axis 0
pub fn cumsum_2d_axis0<B: Backend>(tensor: Tensor<B, 2>) -> Tensor<B, 2> {
    let [rows, _cols] = tensor.dims();

    if rows <= 1 {
        return tensor;
    }

    let mut result = tensor.clone();

    for i in 1..rows {
        let prev_row = result.clone().narrow(0, i - 1, 1);
        let curr_row = tensor.clone().narrow(0, i, 1);
        let accumulated = prev_row + curr_row;
        result = result.slice_assign(s![i..i + 1, ..], accumulated);
    }

    result
}

/// Flip 1D tensor (reverse order)
///
/// # Arguments
/// * `tensor` - Input 1D tensor
///
/// # Returns
/// Flipped tensor with same shape
pub fn flip_1d<B: Backend>(tensor: Tensor<B, 1>) -> Tensor<B, 1> {
    let [size] = tensor.dims();

    if size <= 1 {
        return tensor; // Nothing to flip
    }

    let mut slices = Vec::with_capacity(size);

    // Collect slices in reverse order
    for i in (0..size).rev() {
        let slice = tensor.clone().narrow(0, i, 1);
        slices.push(slice);
    }

    // Concatenate slices
    Tensor::cat(slices, 0)
}

/// Count non-zero elements in tensor
///
/// # Arguments
/// * `tensor` - Input tensor
///
/// # Returns
/// Number of non-zero elements as f64
pub fn count_nonzero<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> f64 {
    let zero = Tensor::zeros_like(&tensor);
    let mask = tensor.not_equal(zero);
    // Convert bool tensor to float before summing
    let mask_float = mask.float();
    mask_float.sum().into_scalar().elem::<f64>()
}

/// Find indices where tensor is non-zero (equivalent to numpy.argwhere)
///
/// # Arguments
/// * `tensor` - Input tensor
///
/// # Returns
/// Vector of indices where tensor is non-zero
pub fn argwhere<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Vec<[usize; D]> {
    let zero = Tensor::zeros_like(&tensor);
    let mask = tensor.not_equal(zero);
    let data = mask.into_data();
    let shape = data.shape.clone();
    let values = data.as_slice::<bool>().unwrap();

    let mut indices = Vec::new();

    // Convert flat index to multi-dimensional indices
    for (flat_idx, &is_nonzero) in values.iter().enumerate() {
        if is_nonzero {
            let mut coords = [0; D];
            let mut remaining = flat_idx;

            for dim in 0..D {
                let stride = shape[dim + 1..].iter().product::<usize>();
                coords[dim] = remaining / stride;
                remaining %= stride;
            }

            indices.push(coords);
        }
    }

    indices
}

/// Compute standard deviation with specified degrees of freedom
///
/// # Arguments
/// * `tensor` - Input tensor
/// * `ddof` - Delta degrees of freedom (default: 1)
///
/// # Returns
/// Standard deviation as scalar
pub fn std_with_ddof<B: Backend, const D: usize>(tensor: Tensor<B, D>, ddof: usize) -> f64 {
    let mean = tensor.clone().mean().into_scalar().elem::<f64>();
    let variance = tensor.clone() - mean;
    let variance = variance.clone() * variance;
    let sum_var = variance.sum().into_scalar().elem::<f64>();

    let total_elements = tensor.shape().num_elements();
    let n = total_elements.saturating_sub(ddof);

    if n == 0 {
        0.0
    } else {
        (sum_var / n as f64).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use burn::backend::Cpu;
    use rstest::*;

    use super::*;

    type TestBackend = Cpu;

    // Helper function to create test data vectors
    fn create_test_data_1d(pattern: &str, size: usize) -> Vec<f32> {
        match pattern {
            "zeros" => vec![0.0; size],
            "ones" => vec![1.0; size],
            "range" => (0..size).map(|i| i as f32).collect(),
            "reverse_range" => (0..size).rev().map(|i| i as f32).collect(),
            "alternating" => (0..size)
                .map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })
                .collect(),
            "negative" => (0..size).map(|i| -(i as f32)).collect(),
            "random_positive" => vec![0.1, 0.3, 0.5, 0.7, 0.9][..size.min(5)].to_vec(),
            "with_nans" => {
                let mut data = vec![1.0; size];
                if size > 0 {
                    data[0] = f32::NAN;
                }
                if size > 1 {
                    data[size - 1] = f32::INFINITY;
                }
                data
            }
            "mixed_signs" => (0..size)
                .map(|i| if i % 2 == 0 { i as f32 } else { -(i as f32) })
                .collect(),
            _ => vec![0.0; size],
        }
    }

    fn assert_vec_approx_eq(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len(), "Vector lengths don't match");
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            if e.is_nan() {
                assert!(a.is_nan(), "Expected NaN at index {}, got {}", i, a);
            } else if e.is_infinite() {
                assert!(
                    a.is_infinite() && a.signum() == e.signum(),
                    "Expected {} at index {}, got {}",
                    e,
                    i,
                    a
                );
            } else {
                assert!(
                    (a - e).abs() < tolerance,
                    "Values differ at index {}: actual={}, expected={}, diff={}",
                    i,
                    a,
                    e,
                    (a - e).abs()
                );
            }
        }
    }

    // === Histogram Tests ===

    #[rstest]
    #[case(5, "random_positive", Some((0.0, 1.0)), 5)] // uniform distribution
    #[case(10, "range", None, 10)] // auto range
    #[case(3, "ones", Some((-1.0, 2.0)), 3)] // constant values
    #[case(7, "mixed_signs", None, 7)] // mixed positive/negative
    fn histogram_with_different_inputs(
        #[case] size: usize,
        #[case] pattern: &str,
        #[case] range: Option<(f64, f64)>,
        #[case] bins: usize,
    ) {
        let device = Default::default();
        let data = create_test_data_1d(pattern, size);
        let tensor = Tensor::<TestBackend, 1>::from_floats(data.as_slice(), &device);

        let hist = histogram(tensor, bins, range);

        // Verify shape
        assert_eq!(hist.dims(), [bins]);

        // Verify non-negative counts
        let hist_data = hist.into_data();
        let counts = hist_data.as_slice::<f32>().unwrap();
        for &count in counts {
            assert!(count >= 0.0, "Histogram counts should be non-negative");
        }

        // Verify total count equals input size (for finite values)
        let finite_count = data.iter().filter(|&&x| x.is_finite()).count();
        let total_count: f32 = counts.iter().sum();
        assert_relative_eq!(total_count, finite_count as f32, epsilon = 1e-5);
    }

    #[rstest]
    #[case(0)] // empty
    #[case(1)] // single element
    #[case(2)] // minimal
    fn histogram_edge_cases(#[case] size: usize) {
        let device = Default::default();
        let data = if size == 0 {
            vec![]
        } else {
            create_test_data_1d("ones", size)
        };
        let tensor = Tensor::<TestBackend, 1>::from_floats(data.as_slice(), &device);

        let hist = histogram(tensor, 5, Some((0.0, 2.0)));
        assert_eq!(hist.dims(), [5]);
    }

    #[test]
    fn histogram_with_nan_and_infinity() {
        let device = Default::default();
        let data = vec![1.0, 2.0, f32::NAN, f32::INFINITY, -f32::INFINITY];
        let tensor = Tensor::<TestBackend, 1>::from_floats(data.as_slice(), &device);

        let hist = histogram(tensor, 3, Some((0.0, 3.0)));

        // Only finite values should be counted
        let hist_data = hist.into_data();
        let counts = hist_data.as_slice::<f32>().unwrap();
        let total_count: f32 = counts.iter().sum();
        assert_relative_eq!(total_count, 2.0, epsilon = 1e-5); // Only 1.0 and 2.0
    }

    // === Cumulative Sum Tests ===

    #[rstest]
    #[case("range", 5, &[0.0, 1.0, 3.0, 6.0, 10.0])] // 0,1,2,3,4
    #[case("ones", 4, &[1.0, 2.0, 3.0, 4.0])] // 1,1,1,1
    #[case("alternating", 6, &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0])] // 1,0,1,0,1,0
    #[case("negative", 3, &[0.0, -1.0, -3.0])] // 0,-1,-2
    fn cumsum_1d_with_patterns(
        #[case] pattern: &str,
        #[case] size: usize,
        #[case] expected: &[f32],
    ) {
        let device = Default::default();
        let data = create_test_data_1d(pattern, size);
        let tensor = Tensor::<TestBackend, 1>::from_floats(data.as_slice(), &device);

        let result = cumsum_1d(tensor);
        assert_eq!(result.dims(), [size]);

        let result_data = result.into_data();
        let values = result_data.as_slice::<f32>().unwrap();
        assert_vec_approx_eq(values, expected, 1e-6);
    }

    #[rstest]
    #[case(0)] // empty
    #[case(1)] // single element
    fn cumsum_1d_edge_cases(#[case] size: usize) {
        let device = Default::default();
        let data = create_test_data_1d("ones", size);
        let tensor = Tensor::<TestBackend, 1>::from_floats(data.as_slice(), &device);

        let result = cumsum_1d(tensor.clone());
        assert_eq!(result.dims(), tensor.dims());

        if size <= 1 {
            // Should return original tensor for size <= 1
            let orig_data = tensor.into_data();
            let result_data = result.into_data();
            let orig_values = orig_data.as_slice::<f32>().unwrap();
            let result_values = result_data.as_slice::<f32>().unwrap();
            assert_vec_approx_eq(result_values, orig_values, 1e-6);
        }
    }

    #[rstest]
    #[case(2, 3)] // 2x3 matrix
    #[case(4, 2)] // 4x2 matrix
    #[case(1, 5)] // 1x5 matrix (edge case)
    fn cumsum_2d_axis0_accumulates_rows(#[case] rows: usize, #[case] cols: usize) {
        let device = Default::default();

        // Create test tensor using specific cases instead of dynamic arrays
        let tensor = match (rows, cols) {
            (2, 3) => {
                Tensor::<TestBackend, 2>::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device)
            }
            (4, 2) => Tensor::<TestBackend, 2>::from_data(
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
                &device,
            ),
            (1, 5) => Tensor::<TestBackend, 2>::from_data([[1.0, 2.0, 3.0, 4.0, 5.0]], &device),
            _ => panic!("Unsupported matrix size for test"),
        };

        let result = cumsum_2d_axis0(tensor);
        assert_eq!(result.dims(), [rows, cols]);

        // Verify cumulative sum property: each row should be sum of current and all previous rows
        let result_data = result.into_data();
        let result_slice = result_data.as_slice::<f32>().unwrap();

        // For first row, should equal original
        for j in 0..cols {
            let expected = (j + 1) as f32;
            let actual = result_slice[j];
            assert_relative_eq!(actual, expected, epsilon = 1e-6);
        }
    }

    // === Flip Tests ===

    #[rstest]
    #[case("range", 5)] // 0,1,2,3,4 -> 4,3,2,1,0
    #[case("alternating", 6)] // 1,0,1,0,1,0 -> 0,1,0,1,0,1
    #[case("ones", 3)] // 1,1,1 -> 1,1,1
    #[case("mixed_signs", 4)] // 0,-1,2,-3 -> -3,2,-1,0
    fn flip_1d_reverses_correctly(#[case] pattern: &str, #[case] size: usize) {
        let device = Default::default();
        let data = create_test_data_1d(pattern, size);
        let tensor = Tensor::<TestBackend, 1>::from_floats(data.as_slice(), &device);

        let flipped = flip_1d(tensor);
        assert_eq!(flipped.dims(), [size]);

        let flipped_data = flipped.into_data();
        let flipped_values = flipped_data.as_slice::<f32>().unwrap();

        // Verify it's the reverse
        let mut expected = data.clone();
        expected.reverse();
        assert_vec_approx_eq(flipped_values, &expected, 1e-6);
    }

    #[rstest]
    #[case(0)] // empty
    #[case(1)] // single element
    fn flip_1d_edge_cases(#[case] size: usize) {
        let device = Default::default();
        let data = create_test_data_1d("range", size);
        let tensor = Tensor::<TestBackend, 1>::from_floats(data.as_slice(), &device);

        let flipped = flip_1d(tensor.clone());
        assert_eq!(flipped.dims(), tensor.dims());

        if size <= 1 {
            // Should return original tensor for size <= 1
            let orig_data = tensor.into_data();
            let flipped_data = flipped.into_data();
            let orig_values = orig_data.as_slice::<f32>().unwrap();
            let flipped_values = flipped_data.as_slice::<f32>().unwrap();
            assert_vec_approx_eq(flipped_values, orig_values, 1e-6);
        }
    }

    // === Count Non-Zero Tests ===

    #[rstest]
    #[case("zeros", 5, 0.0)] // all zeros
    #[case("ones", 4, 4.0)] // all ones
    #[case("alternating", 6, 3.0)] // half zeros
    #[case("range", 5, 4.0)] // 0,1,2,3,4 -> 4 non-zeros
    #[case("mixed_signs", 4, 3.0)] // 0,-1,2,-3 -> 3 non-zeros
    fn count_nonzero_counts_correctly(
        #[case] pattern: &str,
        #[case] size: usize,
        #[case] expected_count: f64,
    ) {
        let device = Default::default();
        let data = create_test_data_1d(pattern, size);
        let tensor = Tensor::<TestBackend, 1>::from_floats(data.as_slice(), &device);

        let count = count_nonzero(tensor);
        assert_relative_eq!(count, expected_count, epsilon = 1e-6);
    }

    #[test]
    fn count_nonzero_2d_tensor() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 2>::from_data(
            [
                [0.0, 1.0, 0.0, 2.0],
                [3.0, 0.0, 0.0, 4.0],
                [0.0, 0.0, 5.0, 0.0],
            ],
            &device,
        );

        let count = count_nonzero(tensor);
        assert_relative_eq!(count, 5.0, epsilon = 1e-6); // 1,2,3,4,5
    }

    #[test]
    fn count_nonzero_3d_tensor() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 3>::from_data(
            [[[1.0, 0.0], [0.0, 2.0]], [[0.0, 3.0], [4.0, 0.0]]],
            &device,
        );

        let count = count_nonzero(tensor);
        assert_relative_eq!(count, 4.0, epsilon = 1e-6); // 1,2,3,4
    }

    // === Argwhere Tests ===

    #[test]
    fn argwhere_finds_nonzero_indices_1d() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 1>::from_floats([0.0, 1.0, 0.0, 3.0, 0.0], &device);

        let indices = argwhere(tensor);
        let expected_indices = vec![[1], [3]];

        assert_eq!(indices.len(), expected_indices.len());
        for (actual, expected) in indices.iter().zip(expected_indices.iter()) {
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn argwhere_finds_nonzero_indices_2d() {
        let device = Default::default();
        let tensor =
            Tensor::<TestBackend, 2>::from_data([[0.0, 1.0, 0.0], [2.0, 0.0, 3.0]], &device);

        let indices = argwhere(tensor);
        let expected_indices = vec![[0, 1], [1, 0], [1, 2]];

        assert_eq!(indices.len(), expected_indices.len());
        for (actual, expected) in indices.iter().zip(expected_indices.iter()) {
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn argwhere_empty_result_for_all_zeros() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 2>::from_data([[0.0, 0.0], [0.0, 0.0]], &device);

        let indices = argwhere(tensor);
        assert!(indices.is_empty());
    }

    #[test]
    fn argwhere_finds_all_indices_for_all_nonzeros() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);

        let indices = argwhere(tensor);
        let expected_indices = vec![[0, 0], [0, 1], [1, 0], [1, 1]];

        assert_eq!(indices.len(), expected_indices.len());
        for (actual, expected) in indices.iter().zip(expected_indices.iter()) {
            assert_eq!(actual, expected);
        }
    }

    // === Standard Deviation Tests ===

    #[rstest]
    #[case("ones", 5, 0, 0.0)] // constant values, any ddof
    #[case("ones", 5, 1, 0.0)] // constant values, any ddof
    #[case("range", 5, 0, 1.4142135623730951)] // population std of [0,1,2,3,4]
    #[case("range", 5, 1, 1.5811388300841898)] // sample std of [0,1,2,3,4] (adjusted)
    fn std_with_ddof_computes_correctly(
        #[case] pattern: &str,
        #[case] size: usize,
        #[case] ddof: usize,
        #[case] expected_std: f64,
    ) {
        let device = Default::default();
        let data = create_test_data_1d(pattern, size);
        let tensor = Tensor::<TestBackend, 1>::from_floats(data.as_slice(), &device);

        let std_dev = std_with_ddof(tensor, ddof);

        if pattern == "ones" {
            // Constant values should have zero std
            assert_relative_eq!(std_dev, 0.0, epsilon = 1e-10);
        } else {
            assert_relative_eq!(std_dev, expected_std, epsilon = 1e-6);
        }
    }

    #[test]
    fn std_with_ddof_edge_cases() {
        let device = Default::default();

        // Single element
        let tensor = Tensor::<TestBackend, 1>::from_floats([5.0], &device);
        let std_dev = std_with_ddof(tensor, 1);
        assert_relative_eq!(std_dev, 0.0, epsilon = 1e-10);

        // ddof >= n should return 0.0
        let tensor = Tensor::<TestBackend, 1>::from_floats([1.0, 2.0], &device);
        let std_dev = std_with_ddof(tensor, 2);
        assert_relative_eq!(std_dev, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn std_with_ddof_known_values() {
        let device = Default::default();

        // Known case: [1, 2, 3, 4, 5] with ddof=1
        // Mean = 3, variance = [(1-3)², (2-3)², (3-3)², (4-3)², (5-3)²] / 4 = [4+1+0+1+4]/4 = 10/4 = 2.5
        // Std = sqrt(2.5) = 1.5811388300841898
        let tensor = Tensor::<TestBackend, 1>::from_floats([1.0, 2.0, 3.0, 4.0, 5.0], &device);
        let std_dev = std_with_ddof(tensor, 1);
        let expected = (2.5f64).sqrt();
        assert_relative_eq!(std_dev, expected, epsilon = 1e-6);
    }

    // === Property-based and Integration Tests ===

    #[test]
    fn cumsum_1d_then_flip_1d_property() {
        let device = Default::default();
        let data = create_test_data_1d("range", 5);
        let tensor = Tensor::<TestBackend, 1>::from_floats(data.as_slice(), &device);

        // Apply cumsum then flip
        let cumsum_result = cumsum_1d(tensor.clone());
        let flipped_cumsum = flip_1d(cumsum_result);

        // Apply flip then cumsum
        let flipped_tensor = flip_1d(tensor);
        let cumsum_flipped = cumsum_1d(flipped_tensor);

        // Results should be different (not commutative)
        let data1 = flipped_cumsum.into_data();
        let data2 = cumsum_flipped.into_data();
        let values1 = data1.as_slice::<f32>().unwrap();
        let values2 = data2.as_slice::<f32>().unwrap();

        // They should have same length but different values (for non-constant input)
        assert_eq!(values1.len(), values2.len());
        // For range input [0,1,2,3,4], these operations will produce different results
        let are_different = values1
            .iter()
            .zip(values2.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(
            are_different,
            "cumsum-flip and flip-cumsum should produce different results"
        );
    }

    #[test]
    fn count_nonzero_equals_argwhere_length() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 2>::from_data(
            [[0.0, 1.0, 0.0], [2.0, 0.0, 3.0], [0.0, 4.0, 0.0]],
            &device,
        );

        let count = count_nonzero(tensor.clone());
        let indices = argwhere(tensor);

        assert_relative_eq!(count, indices.len() as f64, epsilon = 1e-6);
    }

    #[test]
    fn histogram_total_equals_input_size_for_finite_values() {
        let device = Default::default();
        let data = vec![1.0, 2.0, 3.0, f32::NAN, 4.0, 5.0, f32::INFINITY];
        let tensor = Tensor::<TestBackend, 1>::from_floats(data.as_slice(), &device);

        let hist = histogram(tensor, 10, Some((0.0, 10.0)));
        let hist_data = hist.into_data();
        let counts = hist_data.as_slice::<f32>().unwrap();
        let total_count: f32 = counts.iter().sum();

        // Should count only finite values: 1,2,3,4,5 = 5 values
        assert_relative_eq!(total_count, 5.0, epsilon = 1e-5);
    }
}
