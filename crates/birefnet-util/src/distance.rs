//! Distance transform operations for computer vision
//!
//! This module provides distance transform algorithms required for implementing
//! evaluation metrics like WeightedFMeasure that depend on spatial distance calculations.

use burn::tensor::{ElementConversion, Tensor, backend::Backend, s};

/// Euclidean distance transform with indices (simplified implementation)
///
/// This function computes the Euclidean distance transform of a binary image,
/// equivalent to scipy.ndimage.distance_transform_edt with return_indices=True.
///
/// # Arguments
/// * `binary_image` - Input binary image tensor [B, C, H, W] where 0=foreground, 1=background
///
/// # Returns
/// Tuple of (distance_tensor, (indices_y, indices_x)) where:
/// - distance_tensor: [B, C, H, W] with distances to nearest foreground pixel
/// - indices_y: [B, C, H, W] with y-coordinates of nearest foreground pixels  
/// - indices_x: [B, C, H, W] with x-coordinates of nearest foreground pixels
pub fn euclidean_distance_transform<B: Backend>(
    binary_image: Tensor<B, 4>,
) -> (Tensor<B, 4>, (Tensor<B, 4>, Tensor<B, 4>)) {
    let [batch_size, channels, height, width] = binary_image.dims();
    let device = binary_image.device();

    // For simplicity, use brute-force approach: for each background pixel,
    // find the nearest foreground pixel by checking all foreground pixels
    let large_value = (height + width) as f32;
    let mut distances =
        Tensor::<B, 4>::ones([batch_size, channels, height, width], &device) * large_value;
    let mut indices_y = Tensor::<B, 4>::zeros([batch_size, channels, height, width], &device);
    let mut indices_x = Tensor::<B, 4>::zeros([batch_size, channels, height, width], &device);

    // Collect foreground pixel positions
    let mut fg_pixels = Vec::new();
    for b in 0..batch_size {
        for c in 0..channels {
            let mut batch_fg = Vec::new();
            for y in 0..height {
                for x in 0..width {
                    let pixel_val = binary_image
                        .clone()
                        .slice(s![b..b + 1, c..c + 1, y..y + 1, x..x + 1])
                        .into_scalar()
                        .elem::<f32>();

                    if pixel_val < 0.5 {
                        // Foreground pixel
                        batch_fg.push((y, x));
                    }
                }
            }
            fg_pixels.push(batch_fg);
        }
    }

    // For each pixel, find distance to nearest foreground pixel
    for b in 0..batch_size {
        for c in 0..channels {
            for y in 0..height {
                for x in 0..width {
                    let pixel_val = binary_image
                        .clone()
                        .slice(s![b..b + 1, c..c + 1, y..y + 1, x..x + 1])
                        .into_scalar()
                        .elem::<f32>();

                    if pixel_val < 0.5 {
                        // Foreground pixel - distance is 0, indices point to self
                        let zero_tensor =
                            Tensor::<B, 1>::from_floats([0.0], &device).reshape([1, 1, 1, 1]);
                        let y_tensor =
                            Tensor::<B, 1>::from_floats([y as f32], &device).reshape([1, 1, 1, 1]);
                        let x_tensor =
                            Tensor::<B, 1>::from_floats([x as f32], &device).reshape([1, 1, 1, 1]);

                        distances = distances
                            .slice_assign(s![b..b + 1, c..c + 1, y..y + 1, x..x + 1], zero_tensor);
                        indices_y = indices_y
                            .slice_assign(s![b..b + 1, c..c + 1, y..y + 1, x..x + 1], y_tensor);
                        indices_x = indices_x
                            .slice_assign(s![b..b + 1, c..c + 1, y..y + 1, x..x + 1], x_tensor);
                    } else {
                        // Background pixel - find nearest foreground pixel
                        let fg_list = &fg_pixels[b * channels + c];

                        if !fg_list.is_empty() {
                            let mut min_dist = large_value;
                            let mut best_y = 0;
                            let mut best_x = 0;

                            for &(fy, fx) in fg_list {
                                let dy = y as f32 - fy as f32;
                                let dx = x as f32 - fx as f32;
                                let dist = dy.hypot(dx);

                                if dist < min_dist {
                                    min_dist = dist;
                                    best_y = fy;
                                    best_x = fx;
                                }
                            }

                            let dist_tensor = Tensor::<B, 1>::from_floats([min_dist], &device)
                                .reshape([1, 1, 1, 1]);
                            let y_tensor = Tensor::<B, 1>::from_floats([best_y as f32], &device)
                                .reshape([1, 1, 1, 1]);
                            let x_tensor = Tensor::<B, 1>::from_floats([best_x as f32], &device)
                                .reshape([1, 1, 1, 1]);

                            distances = distances.slice_assign(
                                s![b..b + 1, c..c + 1, y..y + 1, x..x + 1],
                                dist_tensor,
                            );
                            indices_y = indices_y
                                .slice_assign(s![b..b + 1, c..c + 1, y..y + 1, x..x + 1], y_tensor);
                            indices_x = indices_x
                                .slice_assign(s![b..b + 1, c..c + 1, y..y + 1, x..x + 1], x_tensor);
                        }
                    }
                }
            }
        }
    }

    (distances, (indices_y, indices_x))
}

/// Manhattan distance transform
///
/// Computes the Manhattan (L1) distance transform of a binary image.
///
/// # Arguments
/// * `binary_image` - Input binary image tensor [B, C, H, W] where 0=foreground, 1=background
///
/// # Returns
/// Distance tensor [B, C, H, W] with Manhattan distances to nearest foreground pixel
pub fn manhattan_distance_transform<B: Backend>(binary_image: Tensor<B, 4>) -> Tensor<B, 4> {
    let [batch_size, channels, height, width] = binary_image.dims();
    let device = binary_image.device();

    // Initialize distance tensor
    let large_value = (height + width) as f32;
    let mut distances =
        Tensor::<B, 4>::ones([batch_size, channels, height, width], &device) * large_value;

    // Set distance to 0 for foreground pixels
    let zero_threshold = Tensor::<B, 4>::zeros_like(&binary_image) + 0.5;
    let foreground_mask = binary_image.lower(zero_threshold);
    distances = distances.mask_fill(foreground_mask, 0.0);

    // Two-pass algorithm for Manhattan distance
    // Forward pass: top-left to bottom-right
    for b in 0..batch_size {
        for c in 0..channels {
            for y in 0..height {
                for x in 0..width {
                    let current_dist = distances
                        .clone()
                        .slice(s![b..b + 1, c..c + 1, y..y + 1, x..x + 1])
                        .into_scalar()
                        .elem::<f32>();

                    let mut min_dist = current_dist;

                    // Check top and left neighbors
                    if y > 0 {
                        let top_dist = distances
                            .clone()
                            .slice(s![b..b + 1, c..c + 1, y - 1..y, x..x + 1])
                            .into_scalar()
                            .elem::<f32>();
                        min_dist = min_dist.min(top_dist + 1.0);
                    }

                    if x > 0 {
                        let left_dist = distances
                            .clone()
                            .slice(s![b..b + 1, c..c + 1, y..y + 1, x - 1..x])
                            .into_scalar()
                            .elem::<f32>();
                        min_dist = min_dist.min(left_dist + 1.0);
                    }

                    if min_dist < current_dist {
                        let dist_tensor =
                            Tensor::<B, 1>::from_floats([min_dist], &device).reshape([1, 1, 1, 1]);
                        distances = distances
                            .slice_assign(s![b..b + 1, c..c + 1, y..y + 1, x..x + 1], dist_tensor);
                    }
                }
            }
        }
    }

    // Backward pass: bottom-right to top-left
    for b in 0..batch_size {
        for c in 0..channels {
            for y in (0..height).rev() {
                for x in (0..width).rev() {
                    let current_dist = distances
                        .clone()
                        .slice(s![b..b + 1, c..c + 1, y..y + 1, x..x + 1])
                        .into_scalar()
                        .elem::<f32>();

                    let mut min_dist = current_dist;

                    // Check bottom and right neighbors
                    if y < height - 1 {
                        let bottom_dist = distances
                            .clone()
                            .slice(s![b..b + 1, c..c + 1, y + 1..y + 2, x..x + 1])
                            .into_scalar()
                            .elem::<f32>();
                        min_dist = min_dist.min(bottom_dist + 1.0);
                    }

                    if x < width - 1 {
                        let right_dist = distances
                            .clone()
                            .slice(s![b..b + 1, c..c + 1, y..y + 1, x + 1..x + 2])
                            .into_scalar()
                            .elem::<f32>();
                        min_dist = min_dist.min(right_dist + 1.0);
                    }

                    if min_dist < current_dist {
                        let dist_tensor =
                            Tensor::<B, 1>::from_floats([min_dist], &device).reshape([1, 1, 1, 1]);
                        distances = distances
                            .slice_assign(s![b..b + 1, c..c + 1, y..y + 1, x..x + 1], dist_tensor);
                    }
                }
            }
        }
    }

    distances
}

/// Simplified Euclidean distance transform without indices
///
/// # Arguments
/// * `binary_image` - Input binary image tensor [B, C, H, W] where 0=foreground, 1=background
///
/// # Returns
/// Distance tensor [B, C, H, W] with Euclidean distances to nearest foreground pixel
pub fn euclidean_distance_transform_simple<B: Backend>(binary_image: Tensor<B, 4>) -> Tensor<B, 4> {
    let (distances, _indices) = euclidean_distance_transform(binary_image);
    distances
}

/// Get distance transform for scipy.ndimage.bwdist compatibility
///
/// This function matches the behavior of scipy's bwdist function used in WeightedFMeasure.
///
/// # Arguments
/// * `binary_image` - Input binary image tensor [B, C, H, W] where False=object, True=background
/// * `return_indices` - Whether to return nearest pixel indices
///
/// # Returns
/// If return_indices is true: (distances, (indices_y, indices_x))
/// If return_indices is false: distances only
pub fn bwdist<B: Backend>(
    binary_image: Tensor<B, 4>,
    return_indices: bool,
) -> (Tensor<B, 4>, Option<(Tensor<B, 4>, Tensor<B, 4>)>) {
    if return_indices {
        let (dist, indices) = euclidean_distance_transform(binary_image);
        (dist, Some(indices))
    } else {
        let dist = euclidean_distance_transform_simple(binary_image);
        (dist, None)
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::Cpu;

    use super::*;

    type TestBackend = Cpu;

    #[test]
    fn manhattan_distance_transform_single_point() {
        let device = Default::default();

        // Create 5x5 image with single foreground pixel at center
        let mut image_data = [[1.0; 5]; 5]; // All background
        image_data[2][2] = 0.0; // Center pixel is foreground

        let image = Tensor::<TestBackend, 4>::from_data([[image_data]], &device);
        let distances = manhattan_distance_transform(image);

        // Check center distance is 0
        let center_dist = distances
            .clone()
            .slice(s![0..1, 0..1, 2..3, 2..3])
            .into_scalar()
            .elem::<f32>();
        assert!((center_dist - 0.0).abs() < 1e-6);

        // Check distance at (1,2) should be 1 (Manhattan distance)
        let neighbor_dist = distances
            .clone()
            .slice(s![0..1, 0..1, 1..2, 2..3])
            .into_scalar()
            .elem::<f32>();
        assert!((neighbor_dist - 1.0).abs() < 1e-6);

        // Check distance at corner (0,0) should be 4 (2+2)
        let corner_dist = distances
            .clone()
            .slice(s![0..1, 0..1, 0..1, 0..1])
            .into_scalar()
            .elem::<f32>();
        assert!((corner_dist - 4.0).abs() < 1e-6);
    }

    #[test]
    fn euclidean_distance_transform_single_point() {
        let device = Default::default();

        // Create 3x3 image with single foreground pixel at center
        let mut image_data = [[1.0; 3]; 3]; // All background
        image_data[1][1] = 0.0; // Center pixel is foreground

        let image = Tensor::<TestBackend, 4>::from_data([[image_data]], &device);
        let (distances, (indices_y, indices_x)) = euclidean_distance_transform(image);

        // Check center distance is 0
        let center_dist = distances
            .clone()
            .slice(s![0..1, 0..1, 1..2, 1..2])
            .into_scalar()
            .elem::<f32>();
        assert!((center_dist - 0.0).abs() < 1e-6);

        // Check that indices point to center for all pixels
        let corner_idx_y = indices_y
            .clone()
            .slice(s![0..1, 0..1, 0..1, 0..1])
            .into_scalar()
            .elem::<f32>();
        let corner_idx_x = indices_x
            .clone()
            .slice(s![0..1, 0..1, 0..1, 0..1])
            .into_scalar()
            .elem::<f32>();

        assert!((corner_idx_y - 1.0).abs() < 1e-6); // Should point to center y=1
        assert!((corner_idx_x - 1.0).abs() < 1e-6); // Should point to center x=1
    }

    #[test]
    fn euclidean_distance_transform_diagonal_distance() {
        let device = Default::default();

        // Create 3x3 image with single foreground pixel at center
        let mut image_data = [[1.0; 3]; 3];
        image_data[1][1] = 0.0; // Center at (1,1)

        let image = Tensor::<TestBackend, 4>::from_data([[image_data]], &device);
        let (distances, _) = euclidean_distance_transform(image);

        // Check diagonal distance (should be sqrt(2) ≈ 1.414)
        let corner_dist = distances
            .clone()
            .slice(s![0..1, 0..1, 0..1, 0..1])
            .into_scalar()
            .elem::<f32>();

        let expected_diagonal = 2.0_f32.sqrt(); // sqrt(1^2 + 1^2)
        assert!((corner_dist - expected_diagonal).abs() < 0.1); // Allow some tolerance for approximation
    }

    #[test]
    fn bwdist_compatibility_function_works() {
        let device = Default::default();

        let mut image_data = [[1.0; 3]; 3];
        image_data[1][1] = 0.0; // Center foreground

        let image = Tensor::<TestBackend, 4>::from_data([[image_data]], &device);

        // Test without indices
        let (distances, indices_opt) = bwdist(image.clone(), false);
        assert!(indices_opt.is_none());

        // Test with indices
        let (distances_with_idx, indices_opt) = bwdist(image, true);
        assert!(indices_opt.is_some());

        // Distances should be the same
        let diff = (distances - distances_with_idx)
            .abs()
            .sum()
            .into_scalar()
            .elem::<f32>();
        assert!(diff < 1e-6);
    }

    #[test]
    fn distance_transforms_handle_all_foreground() {
        let device = Default::default();

        // Create image with all foreground pixels
        let image_data = [[0.0; 3]; 3]; // All foreground
        let image = Tensor::<TestBackend, 4>::from_data([[image_data]], &device);

        let manhattan_dist = manhattan_distance_transform(image.clone());
        let euclidean_dist = euclidean_distance_transform_simple(image);

        // All distances should be 0
        let manhattan_sum = manhattan_dist.sum().into_scalar().elem::<f32>();
        let euclidean_sum = euclidean_dist.sum().into_scalar().elem::<f32>();

        assert!(manhattan_sum < 1e-6);
        assert!(euclidean_sum < 1e-6);
    }
}
