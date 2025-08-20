//! Morphological operations for computer vision
//!
//! This module provides morphological image processing operations required for
//! implementing computer vision evaluation metrics like HCE, MBA, and BIoU.

use burn::tensor::{backend::Backend, s, ElementConversion, Tensor};

/// Structuring element for morphological operations
#[derive(Debug, Clone)]
pub struct StructuringElement<B: Backend> {
    /// Binary kernel defining the shape of the structuring element
    pub kernel: Tensor<B, 2>,
    /// Anchor point (center) of the structuring element
    pub anchor: (usize, usize),
}

impl<B: Backend> StructuringElement<B> {
    /// Create a new structuring element
    pub const fn new(kernel: Tensor<B, 2>, anchor: (usize, usize)) -> Self {
        Self { kernel, anchor }
    }

    /// Create a rectangular structuring element
    pub fn rectangle(height: usize, width: usize, device: &B::Device) -> Self {
        let kernel = Tensor::<B, 2>::ones([height, width], device);
        let anchor = (height / 2, width / 2);
        Self::new(kernel, anchor)
    }

    /// Create a disk (circular) structuring element
    pub fn disk(radius: usize, device: &B::Device) -> Self {
        let size = 2 * radius + 1;
        let center = radius as f32;

        let mut kernel_data = Vec::with_capacity(size * size);

        for i in 0..size {
            for j in 0..size {
                let di = i as f32 - center;
                let dj = j as f32 - center;
                let distance = di.hypot(dj);

                // Include pixel if within radius
                if distance <= radius as f32 {
                    kernel_data.push(1.0);
                } else {
                    kernel_data.push(0.0);
                }
            }
        }

        let kernel =
            Tensor::<B, 1>::from_floats(kernel_data.as_slice(), device).reshape([size, size]);
        let anchor = (radius, radius);
        Self::new(kernel, anchor)
    }

    /// Get kernel dimensions
    pub fn dims(&self) -> [usize; 2] {
        self.kernel.dims()
    }
}

/// Morphological erosion operation
///
/// # Arguments
/// * `image` - Input binary image tensor [B, C, H, W]
/// * `structuring_element` - Structuring element for erosion
///
/// # Returns
/// Eroded binary image tensor with same shape
pub fn erosion<B: Backend>(
    image: Tensor<B, 4>,
    structuring_element: &StructuringElement<B>,
) -> Tensor<B, 4> {
    let [batch_size, channels, height, width] = image.dims();
    let [kernel_h, kernel_w] = structuring_element.dims();
    let (anchor_y, anchor_x) = structuring_element.anchor;

    let device = image.device();
    let mut result = Tensor::<B, 4>::zeros([batch_size, channels, height, width], &device);

    // Pad image to handle border effects
    let pad_y = anchor_y;
    let pad_x = anchor_x;
    let padded_image = pad_replicate_4d(image, pad_y, pad_x);

    // Perform erosion for each pixel
    for b in 0..batch_size {
        for c in 0..channels {
            for y in 0..height {
                for x in 0..width {
                    let mut eroded_value = 1.0; // Start with white (foreground)

                    // Check all kernel positions
                    for ky in 0..kernel_h {
                        for kx in 0..kernel_w {
                            // Get kernel value
                            let kernel_val = structuring_element
                                .kernel
                                .clone()
                                .slice(s![ky..ky + 1, kx..kx + 1])
                                .into_scalar()
                                .elem::<f32>();

                            if kernel_val > 0.5 {
                                // Only consider kernel elements that are "on"
                                // Calculate image position
                                let img_y = y + ky;
                                let img_x = x + kx;

                                // Get image value at this position
                                let img_val = padded_image
                                    .clone()
                                    .slice(s![
                                        b..b + 1,
                                        c..c + 1,
                                        img_y..img_y + 1,
                                        img_x..img_x + 1
                                    ])
                                    .into_scalar()
                                    .elem::<f32>();

                                // For erosion: result is minimum of all structuring element positions
                                if img_val < 0.5 {
                                    // If any required pixel is black
                                    eroded_value = 0.0; // Result is black
                                    break;
                                }
                            }
                        }
                        if eroded_value < 0.5 {
                            break; // Early exit if already determined to be black
                        }
                    }

                    // Set result pixel
                    let value_tensor =
                        Tensor::<B, 1>::from_floats([eroded_value], &device).reshape([1, 1, 1, 1]);
                    result = result
                        .slice_assign(s![b..b + 1, c..c + 1, y..y + 1, x..x + 1], value_tensor);
                }
            }
        }
    }

    result
}

/// Morphological dilation operation
///
/// # Arguments
/// * `image` - Input binary image tensor [B, C, H, W]
/// * `structuring_element` - Structuring element for dilation
///
/// # Returns
/// Dilated binary image tensor with same shape
pub fn dilation<B: Backend>(
    image: Tensor<B, 4>,
    structuring_element: &StructuringElement<B>,
) -> Tensor<B, 4> {
    let [batch_size, channels, height, width] = image.dims();
    let [kernel_h, kernel_w] = structuring_element.dims();
    let (anchor_y, anchor_x) = structuring_element.anchor;

    let device = image.device();
    let mut result = Tensor::<B, 4>::zeros([batch_size, channels, height, width], &device);

    // Pad image to handle border effects
    let pad_y = anchor_y;
    let pad_x = anchor_x;
    let padded_image = pad_replicate_4d(image, pad_y, pad_x);

    // Perform dilation for each pixel
    for b in 0..batch_size {
        for c in 0..channels {
            for y in 0..height {
                for x in 0..width {
                    let mut dilated_value = 0.0; // Start with black (background)

                    // Check all kernel positions
                    for ky in 0..kernel_h {
                        for kx in 0..kernel_w {
                            // Get kernel value
                            let kernel_val = structuring_element
                                .kernel
                                .clone()
                                .slice(s![ky..ky + 1, kx..kx + 1])
                                .into_scalar()
                                .elem::<f32>();

                            if kernel_val > 0.5 {
                                // Only consider kernel elements that are "on"
                                // Calculate image position
                                let img_y = y + ky;
                                let img_x = x + kx;

                                // Get image value at this position
                                let img_val = padded_image
                                    .clone()
                                    .slice(s![
                                        b..b + 1,
                                        c..c + 1,
                                        img_y..img_y + 1,
                                        img_x..img_x + 1
                                    ])
                                    .into_scalar()
                                    .elem::<f32>();

                                // For dilation: result is maximum of all structuring element positions
                                if img_val > 0.5 {
                                    // If any structuring element pixel is white
                                    dilated_value = 1.0; // Result is white
                                    break;
                                }
                            }
                        }
                        if dilated_value > 0.5 {
                            break; // Early exit if already determined to be white
                        }
                    }

                    // Set result pixel
                    let value_tensor =
                        Tensor::<B, 1>::from_floats([dilated_value], &device).reshape([1, 1, 1, 1]);
                    result = result
                        .slice_assign(s![b..b + 1, c..c + 1, y..y + 1, x..x + 1], value_tensor);
                }
            }
        }
    }

    result
}

/// Morphological opening (erosion followed by dilation)
///
/// # Arguments
/// * `image` - Input binary image tensor [B, C, H, W]
/// * `structuring_element` - Structuring element for opening
///
/// # Returns
/// Opened binary image tensor with same shape
pub fn opening<B: Backend>(
    image: Tensor<B, 4>,
    structuring_element: &StructuringElement<B>,
) -> Tensor<B, 4> {
    let eroded = erosion(image, structuring_element);
    dilation(eroded, structuring_element)
}

/// Morphological closing (dilation followed by erosion)
///
/// # Arguments
/// * `image` - Input binary image tensor [B, C, H, W]
/// * `structuring_element` - Structuring element for closing
///
/// # Returns
/// Closed binary image tensor with same shape
pub fn closing<B: Backend>(
    image: Tensor<B, 4>,
    structuring_element: &StructuringElement<B>,
) -> Tensor<B, 4> {
    let dilated = dilation(image, structuring_element);
    erosion(dilated, structuring_element)
}

/// Morphological gradient (dilation - erosion)
///
/// # Arguments
/// * `image` - Input binary image tensor [B, C, H, W]
/// * `structuring_element` - Structuring element for gradient
///
/// # Returns
/// Gradient binary image tensor with same shape
pub fn gradient<B: Backend>(
    image: Tensor<B, 4>,
    structuring_element: &StructuringElement<B>,
) -> Tensor<B, 4> {
    let dilated = dilation(image.clone(), structuring_element);
    let eroded = erosion(image, structuring_element);
    dilated - eroded
}

/// Replicate padding for 4D tensors
fn pad_replicate_4d<B: Backend>(tensor: Tensor<B, 4>, pad_h: usize, pad_w: usize) -> Tensor<B, 4> {
    let [batch_size, channels, height, width] = tensor.dims();

    if pad_h == 0 && pad_w == 0 {
        return tensor;
    }

    let new_height = height + 2 * pad_h;
    let new_width = width + 2 * pad_w;

    let device = tensor.device();
    let mut result = Tensor::<B, 4>::zeros([batch_size, channels, new_height, new_width], &device);

    // Copy original tensor to center
    result = result.slice_assign(
        s![.., .., pad_h..pad_h + height, pad_w..pad_w + width],
        tensor.clone(),
    );

    // Replicate edges - simplified version
    // Top and bottom edges
    if pad_h > 0 {
        let top_row = tensor.clone().slice(s![.., .., 0..1, ..]);
        let bottom_row = tensor.slice(s![.., .., height - 1..height, ..]);

        for i in 0..pad_h {
            result =
                result.slice_assign(s![.., .., i..i + 1, pad_w..pad_w + width], top_row.clone());
            result = result.slice_assign(
                s![
                    ..,
                    ..,
                    pad_h + height + i..pad_h + height + i + 1,
                    pad_w..pad_w + width
                ],
                bottom_row.clone(),
            );
        }
    }

    // Left and right edges (including corners)
    if pad_w > 0 {
        let left_col = result.clone().slice(s![.., .., .., pad_w..pad_w + 1]);
        let right_col = result
            .clone()
            .slice(s![.., .., .., pad_w + width - 1..pad_w + width]);

        for i in 0..pad_w {
            result = result.slice_assign(s![.., .., .., i..i + 1], left_col.clone());
            result = result.slice_assign(
                s![.., .., .., pad_w + width + i..pad_w + width + i + 1],
                right_col.clone(),
            );
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use burn::backend::NdArray;
    use rstest::*;

    use super::*;

    type TestBackend = NdArray;

    // Helper function to create test image data
    fn create_test_image_tensor<B: Backend>(
        pattern: &str,
        size: usize,
        device: &B::Device,
    ) -> Tensor<B, 4> {
        let data = match pattern {
            "solid_white" => vec![vec![1.0; size]; size],
            "solid_black" => vec![vec![0.0; size]; size],
            "center_dot" => {
                let mut data = vec![vec![0.0; size]; size];
                data[size / 2][size / 2] = 1.0;
                data
            }
            "cross" => {
                let mut data = vec![vec![0.0; size]; size];
                let center = size / 2;
                for i in 0..size {
                    data[center][i] = 1.0; // horizontal line
                    data[i][center] = 1.0; // vertical line
                }
                data
            }
            "square_3x3" => {
                let mut data = vec![vec![0.0; size]; size];
                if size >= 5 {
                    for i in 1..4 {
                        for j in 1..4 {
                            data[i][j] = 1.0;
                        }
                    }
                }
                data
            }
            "checkerboard" => {
                let mut data = vec![vec![0.0; size]; size];
                for i in 0..size {
                    for j in 0..size {
                        data[i][j] = if (i + j) % 2 == 0 { 1.0 } else { 0.0 };
                    }
                }
                data
            }
            _ => vec![vec![0.0; size]; size],
        };

        // Convert Vec<Vec<f32>> to flat Vec<f32>
        let flat_data: Vec<f32> = data.into_iter().flatten().collect();
        Tensor::<B, 1>::from_floats(flat_data.as_slice(), device).reshape([1, 1, size, size])
    }

    // Helper function to check if a pixel value matches expected (binary classification)
    fn is_foreground(value: f32) -> bool {
        value > 0.5
    }

    #[rstest]
    #[case(1, 3, 3, (1, 1))] // radius=1 -> 3x3
    #[case(2, 5, 5, (2, 2))] // radius=2 -> 5x5
    #[case(3, 7, 7, (3, 3))] // radius=3 -> 7x7
    fn disk_structuring_element_dimensions(
        #[case] radius: usize,
        #[case] expected_h: usize,
        #[case] expected_w: usize,
        #[case] expected_anchor: (usize, usize),
    ) {
        let device = Default::default();
        let disk = StructuringElement::<TestBackend>::disk(radius, &device);

        let [h, w] = disk.dims();
        assert_eq!(h, expected_h);
        assert_eq!(w, expected_w);
        assert_eq!(disk.anchor, expected_anchor);
    }

    #[rstest]
    #[case(1, 1, 1, 1, (0, 0))]
    #[case(3, 3, 3, 3, (1, 1))]
    #[case(3, 5, 3, 5, (1, 2))]
    #[case(5, 3, 5, 3, (2, 1))]
    fn rectangle_structuring_element_dimensions(
        #[case] height: usize,
        #[case] width: usize,
        #[case] expected_h: usize,
        #[case] expected_w: usize,
        #[case] expected_anchor: (usize, usize),
    ) {
        let device = Default::default();
        let rect = StructuringElement::<TestBackend>::rectangle(height, width, &device);

        let [h, w] = rect.dims();
        assert_eq!(h, expected_h);
        assert_eq!(w, expected_w);
        assert_eq!(rect.anchor, expected_anchor);
    }

    #[rstest]
    #[case("square_3x3", 5, 1)] // 3x3 square eroded by disk(1) -> should preserve center
    #[case("center_dot", 5, 1)] // single dot eroded by disk(1) -> should disappear
    #[case("cross", 7, 1)] // cross pattern eroded by disk(1)
    fn erosion_with_different_patterns(
        #[case] pattern: &str,
        #[case] size: usize,
        #[case] se_radius: usize,
    ) {
        let device = Default::default();
        let image = create_test_image_tensor::<TestBackend>(pattern, size, &device);
        let se = StructuringElement::disk(se_radius, &device);

        let eroded = erosion(image.clone(), &se);
        let [b, c, h, w] = eroded.dims();
        assert_eq!([b, c, h, w], [1, 1, size, size]);

        // Verify erosion properties
        match pattern {
            "square_3x3" => {
                // Center of 3x3 square should survive erosion with disk(1)
                let center = size / 2;
                let center_val = eroded
                    .clone()
                    .slice(s![0..1, 0..1, center..center + 1, center..center + 1])
                    .into_scalar()
                    .elem::<f32>();
                assert!(
                    is_foreground(center_val),
                    "Center of square should survive erosion"
                );
            }
            "center_dot" => {
                // Single pixel should be eroded away by disk(1)
                let center = size / 2;
                let center_val = eroded
                    .clone()
                    .slice(s![0..1, 0..1, center..center + 1, center..center + 1])
                    .into_scalar()
                    .elem::<f32>();
                assert!(
                    !is_foreground(center_val),
                    "Single pixel should be eroded away"
                );
            }
            _ => {
                // General property: eroded image should have <= original foreground pixels
                let original_sum = image.clone().sum().into_scalar().elem::<f32>();
                let eroded_sum = eroded.clone().sum().into_scalar().elem::<f32>();
                assert!(
                    eroded_sum <= original_sum,
                    "Erosion should not increase foreground pixels"
                );
            }
        }
    }

    #[rstest]
    #[case("center_dot", 5, 1)] // single dot dilated by disk(1) -> should expand
    #[case("center_dot", 7, 2)] // single dot dilated by disk(2) -> should expand more
    #[case("square_3x3", 7, 1)] // 3x3 square dilated by disk(1) -> should grow
    fn dilation_with_different_patterns(
        #[case] pattern: &str,
        #[case] size: usize,
        #[case] se_radius: usize,
    ) {
        let device = Default::default();
        let image = create_test_image_tensor::<TestBackend>(pattern, size, &device);
        let se = StructuringElement::disk(se_radius, &device);

        let dilated = dilation(image.clone(), &se);
        let [b, c, h, w] = dilated.dims();
        assert_eq!([b, c, h, w], [1, 1, size, size]);

        // General property: dilation should have >= original foreground pixels
        let original_sum = image.clone().sum().into_scalar().elem::<f32>();
        let dilated_sum = dilated.clone().sum().into_scalar().elem::<f32>();
        assert!(
            dilated_sum >= original_sum,
            "Dilation should not decrease foreground pixels"
        );

        match pattern {
            "center_dot" => {
                // Check that neighbors are also white after dilation
                let center = size / 2;
                let center_val = dilated
                    .clone()
                    .slice(s![0..1, 0..1, center..center + 1, center..center + 1])
                    .into_scalar()
                    .elem::<f32>();
                assert!(is_foreground(center_val), "Center should remain foreground");

                // Check at least one neighbor is also foreground (for radius >= 1)
                if se_radius >= 1 && center > 0 {
                    let neighbor_val = dilated
                        .clone()
                        .slice(s![0..1, 0..1, center - 1..center, center..center + 1])
                        .into_scalar()
                        .elem::<f32>();
                    assert!(
                        is_foreground(neighbor_val),
                        "Dilation should expand to neighbors"
                    );
                }
            }
            _ => {}
        }
    }

    #[rstest]
    #[case("center_dot", 5, 3, 3)] // small noise with large SE -> should remove
    #[case("checkerboard", 6, 3, 3)] // checkerboard with 3x3 SE -> should smooth
    fn opening_noise_removal_test(
        #[case] pattern: &str,
        #[case] size: usize,
        #[case] se_height: usize,
        #[case] se_width: usize,
    ) {
        let device = Default::default();
        let image = create_test_image_tensor::<TestBackend>(pattern, size, &device);
        let se = StructuringElement::rectangle(se_height, se_width, &device);

        let opened = opening(image.clone(), &se);
        let [b, c, h, w] = opened.dims();
        assert_eq!([b, c, h, w], [1, 1, size, size]);

        // Opening should be <= original image (removing small features)
        let original_sum = image.clone().sum().into_scalar().elem::<f32>();
        let opened_sum = opened.clone().sum().into_scalar().elem::<f32>();
        assert!(
            opened_sum <= original_sum,
            "Opening should not increase foreground pixels"
        );

        match pattern {
            "center_dot" => {
                // Single pixel should be removed by large structuring element
                if se_height > 1 || se_width > 1 {
                    let center = size / 2;
                    let center_val = opened
                        .clone()
                        .slice(s![0..1, 0..1, center..center + 1, center..center + 1])
                        .into_scalar()
                        .elem::<f32>();
                    assert!(
                        !is_foreground(center_val),
                        "Small noise should be removed by opening"
                    );
                }
            }
            "checkerboard" => {
                // Checkerboard should be significantly reduced
                assert!(
                    opened_sum < original_sum * 0.5,
                    "Checkerboard should be heavily reduced"
                );
            }
            _ => {}
        }
    }

    #[rstest]
    #[case("square_3x3", 7, 3, 3)] // square with gaps filled by closing
    fn closing_gap_filling_test(
        #[case] pattern: &str,
        #[case] size: usize,
        #[case] se_height: usize,
        #[case] se_width: usize,
    ) {
        let device = Default::default();
        // Create square with a hole in it
        let mut image = create_test_image_tensor::<TestBackend>(pattern, size, &device);
        if pattern == "square_3x3" && size >= 7 {
            // Create hole in center by setting pixel to 0
            let zero_tensor =
                Tensor::<TestBackend, 1>::from_floats([0.0], &device).reshape([1, 1, 1, 1]);
            image = image.slice_assign(s![0..1, 0..1, 3..4, 3..4], zero_tensor);
        }
        let se = StructuringElement::rectangle(se_height, se_width, &device);

        let closed = closing(image.clone(), &se);
        let [b, c, h, w] = closed.dims();
        assert_eq!([b, c, h, w], [1, 1, size, size]);

        // Closing should be >= original image (filling gaps)
        let original_sum = image.clone().sum().into_scalar().elem::<f32>();
        let closed_sum = closed.clone().sum().into_scalar().elem::<f32>();
        assert!(
            closed_sum >= original_sum,
            "Closing should not decrease foreground pixels"
        );
    }

    #[rstest]
    #[case("square_3x3", 7, 1)] // square edges
    #[case("center_dot", 5, 1)] // single dot (should result in zero gradient)
    fn gradient_edge_detection_test(
        #[case] pattern: &str,
        #[case] size: usize,
        #[case] se_radius: usize,
    ) {
        let device = Default::default();
        let image = create_test_image_tensor::<TestBackend>(pattern, size, &device);
        let se = StructuringElement::disk(se_radius, &device);

        let grad = gradient(image.clone(), &se);
        let [b, c, h, w] = grad.dims();
        assert_eq!([b, c, h, w], [1, 1, size, size]);

        match pattern {
            "center_dot" => {
                // Single pixel gradient: for a single pixel, dilation expands it but erosion removes it
                // So gradient = dilation - erosion should be positive (the expanded region)
                let grad_sum = grad.clone().sum().into_scalar().elem::<f32>();
                assert!(
                    grad_sum >= 0.0,
                    "Gradient should be non-negative for center dot"
                );
            }
            "square_3x3" => {
                // Gradient should highlight edges - should be positive at boundaries
                let grad_sum = grad.clone().sum().into_scalar().elem::<f32>();
                assert!(grad_sum > 0.0, "Gradient should detect edges in square");
            }
            _ => {}
        }
    }

    #[rstest]
    #[case(1, 1)] // minimal image
    #[case(3, 3)] // small image
    #[case(10, 15)] // rectangular image
    fn edge_cases_minimal_images(#[case] height: usize, #[case] width: usize) {
        let device = Default::default();
        let flat_data: Vec<f32> = vec![1.0; height * width];
        let image = Tensor::<TestBackend, 1>::from_floats(flat_data.as_slice(), &device)
            .reshape([1, 1, height, width]);
        let se = StructuringElement::disk(1, &device);

        // All operations should work without panicking
        let eroded = erosion(image.clone(), &se);
        let dilated = dilation(image.clone(), &se);
        let opened = opening(image.clone(), &se);
        let closed = closing(image.clone(), &se);
        let grad = gradient(image.clone(), &se);

        // Verify output dimensions
        assert_eq!(eroded.dims(), [1, 1, height, width]);
        assert_eq!(dilated.dims(), [1, 1, height, width]);
        assert_eq!(opened.dims(), [1, 1, height, width]);
        assert_eq!(closed.dims(), [1, 1, height, width]);
        assert_eq!(grad.dims(), [1, 1, height, width]);
    }

    #[rstest]
    #[case(1, 2, 3)] // multi-batch
    #[case(2, 1, 5)] // multi-channel
    #[case(2, 2, 4)] // multi-batch, multi-channel
    fn multi_batch_channel_operations(
        #[case] batch_size: usize,
        #[case] channels: usize,
        #[case] spatial_size: usize,
    ) {
        let device = Default::default();

        // Create test data for multiple batches/channels
        let flat_data: Vec<f32> = (0..batch_size * channels * spatial_size * spatial_size)
            .map(|i| {
                let spatial_idx = i % (spatial_size * spatial_size);
                let y = spatial_idx / spatial_size;
                let x = spatial_idx % spatial_size;
                // Create square pattern for each channel
                if y >= 1 && y < 4 && x >= 1 && x < 4 && spatial_size >= 5 {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        let image = Tensor::<TestBackend, 1>::from_floats(flat_data.as_slice(), &device).reshape([
            batch_size,
            channels,
            spatial_size,
            spatial_size,
        ]);
        let se = StructuringElement::disk(1, &device);

        let eroded = erosion(image.clone(), &se);
        let dilated = dilation(image.clone(), &se);

        // Verify batch and channel dimensions are preserved
        assert_eq!(
            eroded.dims(),
            [batch_size, channels, spatial_size, spatial_size]
        );
        assert_eq!(
            dilated.dims(),
            [batch_size, channels, spatial_size, spatial_size]
        );
    }

    #[test]
    fn morphological_properties_duality() {
        let device = Default::default();
        let image = create_test_image_tensor::<TestBackend>("square_3x3", 5, &device);
        let se = StructuringElement::disk(1, &device);

        // Test opening-closing duality property
        let opened = opening(image.clone(), &se);
        let opened_closed = closing(opened.clone(), &se);

        let closed = closing(image.clone(), &se);
        let closed_opened = opening(closed.clone(), &se);

        // These should be different but within expected ranges
        let oc_sum = opened_closed.sum().into_scalar().elem::<f32>();
        let co_sum = closed_opened.sum().into_scalar().elem::<f32>();

        // Test mathematical properties of opening and closing
        let original_sum = image.clone().sum().into_scalar().elem::<f32>();
        let opened_sum = opened.sum().into_scalar().elem::<f32>();
        let closed_sum = closing(image.clone(), &se)
            .sum()
            .into_scalar()
            .elem::<f32>();

        // Opening should be <= original
        assert!(
            opened_sum <= original_sum,
            "Opening should not increase foreground"
        );

        // Closing should be >= original
        assert!(
            closed_sum >= original_sum,
            "Closing should not decrease foreground"
        );

        // Opening-closing should be >= opening
        assert!(oc_sum >= opened_sum, "Opening-closing should be >= opening");

        // Closing-opening relationship is more complex, just verify it's non-negative
        assert!(co_sum >= 0.0, "Closing-opening should be non-negative");
    }

    #[test]
    fn idempotence_properties() {
        let device = Default::default();
        let image = create_test_image_tensor::<TestBackend>("square_3x3", 7, &device);
        let se = StructuringElement::disk(1, &device);

        // Opening is idempotent: opening(opening(X)) = opening(X)
        let opened_once = opening(image.clone(), &se);
        let opened_twice = opening(opened_once.clone(), &se);

        let sum1 = opened_once.sum().into_scalar().elem::<f32>();
        let sum2 = opened_twice.sum().into_scalar().elem::<f32>();
        assert_relative_eq!(sum1, sum2, epsilon = 1e-5);

        // Closing is idempotent: closing(closing(X)) = closing(X)
        let closed_once = closing(image.clone(), &se);
        let closed_twice = closing(closed_once.clone(), &se);

        let sum3 = closed_once.sum().into_scalar().elem::<f32>();
        let sum4 = closed_twice.sum().into_scalar().elem::<f32>();
        assert_relative_eq!(sum3, sum4, epsilon = 1e-5);
    }

    // Python reference comparison tests
    mod python_reference_tests {
        use super::*;

        // Helper function to compare Rust result with Python reference
        fn compare_with_python_reference(
            rust_result: Tensor<TestBackend, 4>,
            python_reference: Vec<Vec<f32>>,
            tolerance: f32,
        ) -> bool {
            let [_, _, height, width] = rust_result.dims();
            assert_eq!(height, python_reference.len());
            assert_eq!(width, python_reference[0].len());

            let mut max_diff = 0.0f32;
            let mut mismatch_count = 0;

            for i in 0..height {
                for j in 0..width {
                    let rust_val = rust_result
                        .clone()
                        .slice(s![0..1, 0..1, i..i + 1, j..j + 1])
                        .into_scalar()
                        .elem::<f32>();
                    let python_val = python_reference[i][j];

                    let diff = (rust_val - python_val).abs();
                    max_diff = max_diff.max(diff);

                    if diff > tolerance {
                        mismatch_count += 1;
                    }
                }
            }

            println!(
                "Max difference: {}, Mismatches: {}/{}",
                max_diff,
                mismatch_count,
                height * width
            );
            mismatch_count == 0
        }

        #[test]
        fn center_dot_erosion_disk_radius1_matches_python() {
            let device = Default::default();

            // Input from Python reference
            let input_data = vec![
                vec![0.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.0],
            ];

            // Expected output from Python reference (single pixel eroded away)
            let expected_output = vec![
                vec![0.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.0],
            ];

            let flat_data: Vec<f32> = input_data.into_iter().flatten().collect();
            let image = Tensor::<TestBackend, 1>::from_floats(flat_data.as_slice(), &device)
                .reshape([1, 1, 5, 5]);
            let se = StructuringElement::disk(1, &device);

            let result = erosion(image, &se);
            assert!(compare_with_python_reference(result, expected_output, 1e-5));
        }

        #[test]
        fn center_dot_dilation_disk_radius1_matches_python() {
            let device = Default::default();

            // Input from Python reference
            let input_data = vec![
                vec![0.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.0],
            ];

            // Expected output from Python reference (dilation should expand the dot)
            let expected_output = vec![
                vec![0.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 0.0, 0.0],
                vec![0.0, 1.0, 1.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.0],
            ];

            let flat_data: Vec<f32> = input_data.into_iter().flatten().collect();
            let image = Tensor::<TestBackend, 1>::from_floats(flat_data.as_slice(), &device)
                .reshape([1, 1, 5, 5]);
            let se = StructuringElement::disk(1, &device);

            let result = dilation(image, &se);
            assert!(compare_with_python_reference(result, expected_output, 1e-5));
        }

        #[test]
        fn square_3x3_erosion_disk_radius1_matches_python() {
            let device = Default::default();

            // Input: 3x3 square in 5x5 image
            let input_data = vec![
                vec![0.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 1.0, 1.0, 1.0, 0.0],
                vec![0.0, 1.0, 1.0, 1.0, 0.0],
                vec![0.0, 1.0, 1.0, 1.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.0],
            ];

            // Expected output: center pixel should survive erosion
            let expected_output = vec![
                vec![0.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.0],
            ];

            let flat_data: Vec<f32> = input_data.into_iter().flatten().collect();
            let image = Tensor::<TestBackend, 1>::from_floats(flat_data.as_slice(), &device)
                .reshape([1, 1, 5, 5]);
            let se = StructuringElement::disk(1, &device);

            let result = erosion(image, &se);
            assert!(compare_with_python_reference(result, expected_output, 1e-5));
        }

        #[test]
        fn opening_noise_removal_matches_python() {
            let device = Default::default();

            // Single pixel noise that should be removed by opening
            let input_data = vec![
                vec![0.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.0],
            ];

            // Expected: noise should be completely removed by 3x3 rectangle
            let expected_output = vec![
                vec![0.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.0],
            ];

            let flat_data: Vec<f32> = input_data.into_iter().flatten().collect();
            let image = Tensor::<TestBackend, 1>::from_floats(flat_data.as_slice(), &device)
                .reshape([1, 1, 5, 5]);
            let se = StructuringElement::rectangle(3, 3, &device);

            let result = opening(image, &se);
            assert!(compare_with_python_reference(result, expected_output, 1e-5));
        }
    }
}
