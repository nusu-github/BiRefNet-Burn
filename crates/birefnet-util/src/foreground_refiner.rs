use burn::tensor::{Tensor, backend::Backend, module::avg_pool2d, s};

/// Small epsilon value to prevent division by zero in blur fusion calculations
const EPSILON: f32 = 1e-5;

/// Radius for the second refinement pass in the blur fusion algorithm
const SECOND_PASS_RADIUS: usize = 6;

/// Padding configuration for tensor operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Padding {
    pub left: usize,
    pub right: usize,
    pub top: usize,
    pub bottom: usize,
}

impl Padding {
    /// Create new padding with all sides equal
    pub const fn uniform(pad: usize) -> Self {
        Self {
            left: pad,
            right: pad,
            top: pad,
            bottom: pad,
        }
    }

    /// Create new padding from tuple (left, right, top, bottom)
    pub const fn from_tuple(padding: (usize, usize, usize, usize)) -> Self {
        Self {
            left: padding.0,
            right: padding.1,
            top: padding.2,
            bottom: padding.3,
        }
    }

    /// Check if any padding is applied
    pub const fn is_zero(&self) -> bool {
        self.left == 0 && self.right == 0 && self.top == 0 && self.bottom == 0
    }
}

/// Apply replicate padding to a 4D tensor [B, C, H, W]
///
/// This function replicates edge pixels since Burn only supports constant padding.
/// Equivalent to PyTorch's F.pad with mode='replicate'.
///
/// # Arguments
/// * `tensor` - Input tensor of shape [B, C, H, W]
/// * `padding` - Padding configuration
///
/// # Returns
/// * Padded tensor
///
/// # Panics
/// Panics if the input tensor is not 4-dimensional.
fn pad_replicate<B: Backend>(tensor: Tensor<B, 4>, padding: Padding) -> Tensor<B, 4> {
    let [batch_size, channels, height, width] = tensor.dims();

    // If no padding needed, return original tensor
    if padding.is_zero() {
        return tensor;
    }

    let new_height = height + padding.top + padding.bottom;
    let new_width = width + padding.left + padding.right;

    // Create output tensor with zeros
    let device = tensor.device();
    let mut result = Tensor::<B, 4>::zeros([batch_size, channels, new_height, new_width], &device);

    // Copy original tensor to center
    result = result.slice_assign(
        s![
            ..,
            ..,
            padding.top..padding.top + height,
            padding.left..padding.left + width
        ],
        tensor.clone(),
    );

    // Replicate top edge
    if padding.top > 0 {
        let top_edge = tensor.clone().slice(s![.., .., 0..1, 0..width]);
        for i in 0..padding.top {
            result = result.slice_assign(
                s![.., .., i..i + 1, padding.left..padding.left + width],
                top_edge.clone(),
            );
        }
    }

    // Replicate bottom edge
    if padding.bottom > 0 {
        let bottom_edge = tensor.slice(s![.., .., height - 1..height, 0..width]);
        for i in 0..padding.bottom {
            result = result.slice_assign(
                s![
                    ..,
                    ..,
                    padding.top + height + i..padding.top + height + i + 1,
                    padding.left..padding.left + width
                ],
                bottom_edge.clone(),
            );
        }
    }

    // Replicate left edge (including padded top/bottom)
    if padding.left > 0 {
        let left_edge =
            result
                .clone()
                .slice(s![.., .., 0..new_height, padding.left..padding.left + 1]);
        for i in 0..padding.left {
            result = result.slice_assign(s![.., .., 0..new_height, i..i + 1], left_edge.clone());
        }
    }

    // Replicate right edge (including padded top/bottom)
    if padding.right > 0 {
        let right_edge = result.clone().slice(s![
            ..,
            ..,
            0..new_height,
            padding.left + width - 1..padding.left + width
        ]);
        for i in 0..padding.right {
            result = result.slice_assign(
                s![
                    ..,
                    ..,
                    0..new_height,
                    padding.left + width + i..padding.left + width + i + 1
                ],
                right_edge.clone(),
            );
        }
    }

    result
}

/// Apply mean blur equivalent to cv2.blur
///
/// # Arguments
/// * `x` - Input tensor of shape [B, C, H, W]
/// * `kernel_size` - Size of the blur kernel
///
/// # Returns
/// * Blurred tensor of the same shape as input
///
/// # Panics
/// Panics if kernel_size is 0.
fn mean_blur<B: Backend>(x: Tensor<B, 4>, kernel_size: usize) -> Tensor<B, 4> {
    assert!(kernel_size > 0, "Kernel size must be greater than 0");

    let padding = if kernel_size % 2 == 0 {
        let pad_l = kernel_size / 2 - 1;
        let pad_r = kernel_size / 2;
        let pad_t = kernel_size / 2 - 1;
        let pad_b = kernel_size / 2;
        Padding::from_tuple((pad_l, pad_r, pad_t, pad_b))
    } else {
        let pad = kernel_size / 2;
        Padding::uniform(pad)
    };

    let x_padded = pad_replicate(x, padding);

    // Use avg_pool2d with count_include_pad=false to match cv2.blur behavior
    avg_pool2d(
        x_padded,
        [kernel_size, kernel_size],
        [1, 1], // stride = 1
        [0, 0], // no additional padding
        false,  // count_include_pad = false
        false,  // ceil_mode = false
    )
}

/// Core blur fusion estimation algorithm
///
/// # Arguments
/// * `image` - Input image tensor [B, C, H, W]
/// * `fg` - Foreground tensor [B, C, H, W]
/// * `bg` - Background tensor [B, C, H, W]
/// * `alpha` - Alpha mask tensor [B, 1, H, W]
/// * `radius` - Blur radius
///
/// # Returns
/// * Tuple of (refined_foreground, refined_background)
///
/// # Panics
/// Panics if radius is 0 or if tensors have incompatible shapes.
fn blur_fusion_estimator<B: Backend>(
    image: Tensor<B, 4>,
    fg: Tensor<B, 4>,
    bg: Tensor<B, 4>,
    alpha: Tensor<B, 4>,
    radius: usize,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    assert!(radius > 0, "Blur radius must be greater than 0");

    // Apply blur operations
    let blurred_alpha = mean_blur(alpha.clone(), radius);
    let blurred_fga = mean_blur(fg * alpha.clone(), radius);
    let blurred_fg = blurred_fga / (blurred_alpha.clone() + EPSILON);

    let one_minus_alpha = Tensor::ones_like(&alpha) - alpha.clone();
    let blurred_b1a = mean_blur(bg * one_minus_alpha, radius);
    let one_minus_blurred_alpha = Tensor::ones_like(&blurred_alpha) - blurred_alpha.clone();
    let blurred_bg = blurred_b1a / (one_minus_blurred_alpha + EPSILON);

    // Refine foreground
    let refined_fg = blurred_fg.clone()
        + alpha.clone()
            * (image
                - alpha.clone() * blurred_fg
                - (Tensor::ones_like(&alpha) - alpha) * blurred_bg.clone());
    let refined_fg = refined_fg.clamp(0.0, 1.0);

    (refined_fg, blurred_bg)
}

/// Refine foreground using blur fusion algorithm
///
/// This is the core implementation that handles batched tensor inputs.
///
/// # Arguments
/// * `image` - Input image tensor [B, C, H, W], values in [0, 1]
/// * `mask` - Alpha mask tensor [B, 1, H, W], values in [0, 1]
/// * `radius` - Blur radius for refinement
///
/// # Returns
/// * Refined foreground tensor with same shape as input image
///
/// # Panics
/// Panics if radius is 0 or if tensors have incompatible dimensions.
///
/// # Examples
///
/// ```rust,ignore
/// use birefnet_util::foreground_refiner::refine_foreground_core;
/// use burn::{backend::cpu::Cpu, tensor::Tensor};
///
/// type Backend = Cpu<f32>;
/// let device = Default::default();
///
/// // Create sample image and mask tensors
/// let image = Tensor::<Backend, 4>::ones([1, 3, 64, 64], &device) * 0.8;
/// let mask = Tensor::<Backend, 4>::ones([1, 1, 64, 64], &device);
///
/// // Refine the foreground
/// let refined = refine_foreground_core(image, mask, 90);
/// assert_eq!(refined.dims(), [1, 3, 64, 64]);
/// ```
pub fn refine_foreground_core<B: Backend>(
    image: Tensor<B, 4>,
    mask: Tensor<B, 4>,
    radius: usize,
) -> Tensor<B, 4> {
    assert!(radius > 0, "Blur radius must be greater than 0");

    let image_dims = image.dims();
    let mask_dims = mask.dims();
    assert_eq!(image_dims[0], mask_dims[0], "Batch sizes must match");
    assert_eq!(mask_dims[1], 1, "Mask must have 1 channel");
    assert_eq!(image_dims[2], mask_dims[2], "Heights must match");
    assert_eq!(image_dims[3], mask_dims[3], "Widths must match");

    // First refinement pass
    let (fg_refined, bg_refined) = blur_fusion_estimator(
        image.clone(),
        image.clone(),
        image.clone(),
        mask.clone(),
        radius,
    );

    // Second refinement pass with smaller radius
    let (fg_final, _) =
        blur_fusion_estimator(image, fg_refined, bg_refined, mask, SECOND_PASS_RADIUS);

    fg_final
}

/// Refine foreground using blur fusion algorithm with automatic dimension handling
///
/// This function handles single image [C, H, W] inputs by converting to batch format internally.
///
/// # Arguments
/// * `image` - Input image tensor [C, H, W], values in [0, 1]
/// * `mask` - Alpha mask tensor [H, W], values in [0, 1]
/// * `radius` - Blur radius for refinement (default: 90)
///
/// # Returns
/// * Refined foreground tensor with same shape and dimensions as input image
///
/// # Panics
/// Panics if radius is Some(0) or if image and mask dimensions don't match.
///
/// # Examples
///
/// ```rust,ignore
/// use birefnet_util::foreground_refiner::refine_foreground;
/// use burn::{backend::cpu::Cpu, tensor::Tensor};
///
/// type Backend = Cpu<f32>;
/// let device = Default::default();
///
/// // Create sample image [C, H, W] and mask [H, W]
/// let image = Tensor::<Backend, 3>::ones([3, 256, 256], &device) * 0.7;
/// let mask = Tensor::<Backend, 2>::ones([256, 256], &device) * 0.9;
///
/// // Refine the foreground with custom radius
/// let refined = refine_foreground(image, mask, Some(45));
/// assert_eq!(refined.dims(), [3, 256, 256]);
/// ```
pub fn refine_foreground<B: Backend>(
    image: Tensor<B, 3>,
    mask: Tensor<B, 2>,
    radius: Option<usize>,
) -> Tensor<B, 3> {
    let radius = radius.unwrap_or(90);
    assert!(radius > 0, "Blur radius must be greater than 0");

    let image_dims = image.dims();
    let mask_dims = mask.dims();
    assert_eq!(
        image_dims[1], mask_dims[0],
        "Image height must match mask height"
    );
    assert_eq!(
        image_dims[2], mask_dims[1],
        "Image width must match mask width"
    );

    // Convert to 4D tensors
    let image_4d = image.unsqueeze_dim::<4>(0); // [1, C, H, W]
    let mask_4d = mask.unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0); // [1, 1, H, W]

    // Process with core function
    let refined_4d = refine_foreground_core(image_4d, mask_4d, radius);

    // Remove batch dimension
    refined_4d.squeeze()
}

/// Refine foreground for batch inputs
///
/// This function provides a convenient interface for processing multiple images at once.
///
/// # Arguments
/// * `images` - Batch of input image tensors [B, C, H, W], values in [0, 1]
/// * `masks` - Batch of alpha mask tensors [B, 1, H, W], values in [0, 1]
/// * `radius` - Blur radius for refinement (default: 90)
///
/// # Returns
/// * Batch of refined foreground tensors [B, C, H, W]
///
/// # Panics
/// Panics if radius is Some(0) or if tensors have incompatible dimensions.
///
/// # Examples
///
/// ```rust,ignore
/// use birefnet_util::foreground_refiner::refine_foreground_batch;
/// use burn::{backend::cpu::Cpu, tensor::Tensor};
///
/// type Backend = Cpu<f32>;
/// let device = Default::default();
///
/// // Create batch of images [B, C, H, W] and masks [B, 1, H, W]
/// let images = Tensor::<Backend, 4>::ones([4, 3, 128, 128], &device) * 0.6;
/// let masks = Tensor::<Backend, 4>::ones([4, 1, 128, 128], &device) * 0.8;
///
/// // Refine the batch with default radius
/// let refined = refine_foreground_batch(images, masks, None);
/// assert_eq!(refined.dims(), [4, 3, 128, 128]);
/// ```
pub fn refine_foreground_batch<B: Backend>(
    images: Tensor<B, 4>,
    masks: Tensor<B, 4>,
    radius: Option<usize>,
) -> Tensor<B, 4> {
    let radius = radius.unwrap_or(90);
    refine_foreground_core(images, masks, radius)
}

#[cfg(test)]
mod tests {
    use burn::backend::cpu::Cpu;

    use super::*;

    type TestBackend = Cpu<f32>;

    #[test]
    fn pad_replicate_uniform_padding_preserves_dimensions() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 4>::from_data([[[[1.0, 2.0], [3.0, 4.0]]]], &device);

        let padded = pad_replicate(tensor, Padding::uniform(1));
        let expected_shape = [1, 1, 4, 4];
        assert_eq!(padded.dims(), expected_shape);

        // Check that corners are replicated correctly
        let data = padded.to_data();
        let values = data.as_slice::<f32>().unwrap();

        // Top-left corner should be 1.0
        assert!((values[0] - 1.0).abs() < 1e-6);
        // Top-right corner should be 2.0
        assert!((values[3] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn mean_blur_uniform_input_stays_uniform() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 4>::from_data(
            [[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]],
            &device,
        );

        let blurred = mean_blur(tensor, 3);
        let expected_shape = [1, 1, 3, 3];
        assert_eq!(blurred.dims(), expected_shape);

        // For uniform input, output should also be uniform
        let data = blurred.to_data();
        let values = data.as_slice::<f32>().unwrap();
        for &val in values {
            assert!((val - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn refine_foreground_core_clamps_values_to_range() {
        let device = Default::default();
        let image = Tensor::<TestBackend, 4>::from_data([[[[0.5, 0.7], [0.3, 0.9]]]], &device);
        let mask = Tensor::<TestBackend, 4>::from_data([[[[1.0, 0.8], [0.2, 0.6]]]], &device);

        let refined = refine_foreground_core(image, mask, 3);
        let expected_shape = [1, 1, 2, 2];
        assert_eq!(refined.dims(), expected_shape);

        // Values should be clamped between 0 and 1
        let data = refined.to_data();
        let values = data.as_slice::<f32>().unwrap();
        for &val in values {
            assert!(val >= 0.0 && val <= 1.0);
        }
    }

    #[test]
    fn refine_foreground_single_image_clamps_to_range() {
        let device = Default::default();
        let image = Tensor::<TestBackend, 3>::from_data(
            [[[0.5, 0.7, 0.2], [0.3, 0.9, 0.4], [0.8, 0.1, 0.6]]],
            &device,
        );
        let mask = Tensor::<TestBackend, 2>::from_data(
            [[1.0, 0.8, 0.3], [0.2, 0.6, 0.9], [0.7, 0.4, 0.5]],
            &device,
        );

        let refined = refine_foreground(image, mask, Some(5));
        let expected_shape = [1, 3, 3];
        assert_eq!(refined.dims(), expected_shape);

        // Values should be clamped between 0 and 1
        let data = refined.to_data();
        let values = data.as_slice::<f32>().unwrap();
        for &val in values {
            assert!(val >= 0.0 && val <= 1.0);
        }
    }

    #[test]
    fn refine_foreground_batch_preserves_shapes() {
        let device = Default::default();
        let images = Tensor::<TestBackend, 4>::from_data(
            [[[[0.5, 0.7], [0.3, 0.9]]], [[[0.2, 0.8], [0.6, 0.4]]]],
            &device,
        );
        let masks = Tensor::<TestBackend, 4>::from_data(
            [[[[1.0, 0.8], [0.2, 0.6]]], [[[0.9, 0.3], [0.7, 0.5]]]],
            &device,
        );

        let refined = refine_foreground_batch(images, masks, Some(7));
        let expected_shape = [2, 1, 2, 2];
        assert_eq!(refined.dims(), expected_shape);

        // Values should be clamped between 0 and 1
        let data = refined.to_data();
        let values = data.as_slice::<f32>().unwrap();
        for &val in values {
            assert!(val >= 0.0 && val <= 1.0);
        }
    }

    #[test]
    fn pad_replicate_asymmetric_padding_replicates_edges_correctly() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 4>::from_data([[[[1.0, 2.0], [3.0, 4.0]]]], &device);

        let padded = pad_replicate(tensor, Padding::from_tuple((2, 1, 1, 2)));
        let expected_shape = [1, 1, 5, 5]; // Original 2x2 + padding (top=1, bottom=2, left=2, right=1)
        assert_eq!(padded.dims(), expected_shape);

        let data = padded.to_data();
        let values = data.as_slice::<f32>().unwrap();

        // With padding (left=2, right=1, top=1, bottom=2):
        // Original tensor [1.0, 2.0]  positioned at [1,2] and [1,3]
        //                [3.0, 4.0]  positioned at [2,2] and [2,3]

        // Check original values in center
        let center_1_1 = values[1 * 5 + 2]; // Row 1, Col 2 -> should be 1.0
        let center_1_2 = values[1 * 5 + 3]; // Row 1, Col 3 -> should be 2.0
        let center_2_1 = values[2 * 5 + 2]; // Row 2, Col 2 -> should be 3.0
        let center_2_2 = values[2 * 5 + 3]; // Row 2, Col 3 -> should be 4.0

        assert!((center_1_1 - 1.0).abs() < 1e-6);
        assert!((center_1_2 - 2.0).abs() < 1e-6);
        assert!((center_2_1 - 3.0).abs() < 1e-6);
        assert!((center_2_2 - 4.0).abs() < 1e-6);

        // Check edge replications
        // Top edge should replicate first row
        assert!((values[0 * 5 + 2] - 1.0).abs() < 1e-6); // Top row, original [0,0]
        assert!((values[0 * 5 + 3] - 2.0).abs() < 1e-6); // Top row, original [0,1]

        // Left edge should replicate first column
        assert!((values[1 * 5 + 0] - 1.0).abs() < 1e-6); // Left col, replicated from [1,2]
        assert!((values[1 * 5 + 1] - 1.0).abs() < 1e-6); // Left col, replicated from [1,2]
        assert!((values[2 * 5 + 0] - 3.0).abs() < 1e-6); // Left col, replicated from [2,2]
        assert!((values[2 * 5 + 1] - 3.0).abs() < 1e-6); // Left col, replicated from [2,2]
    }

    #[test]
    fn mean_blur_odd_and_even_kernels_produce_different_results() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 4>::from_data(
            [[[[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]]],
            &device,
        );

        // Test odd kernel size
        let blurred_3 = mean_blur(tensor.clone(), 3);
        assert_eq!(blurred_3.dims(), [1, 1, 3, 3]);

        // Test even kernel size
        let blurred_4 = mean_blur(tensor.clone(), 4);
        assert_eq!(blurred_4.dims(), [1, 1, 3, 3]);

        // Results should be different due to different padding
        let data_3 = blurred_3.to_data();
        let data_4 = blurred_4.to_data();
        let values_3 = data_3.as_slice::<f32>().unwrap();
        let values_4 = data_4.as_slice::<f32>().unwrap();

        // At least some values should be different
        let mut differences = 0;
        for (v3, v4) in values_3.iter().zip(values_4.iter()) {
            if (v3 - v4).abs() > 1e-6 {
                differences += 1;
            }
        }
        assert!(
            differences > 0,
            "Odd and even kernel sizes should produce different results"
        );
    }
}
