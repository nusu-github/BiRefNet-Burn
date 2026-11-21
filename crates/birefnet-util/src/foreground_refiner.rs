//! Multi-pass blur fusion foreground refinement.

use burn::tensor::{
    Tensor,
    module::avg_pool2d,
    ops::{AvgPoolOptions, PadMode},
};

const EPSILON: f32 = 1e-5;
const SECOND_PASS_RADIUS: usize = 6;
const DEFAULT_REFINE_RADIUS: usize = 90;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Padding {
    left: usize,
    right: usize,
    top: usize,
    bottom: usize,
}

impl Padding {
    const fn uniform(pad: usize) -> Self {
        Self {
            left: pad,
            right: pad,
            top: pad,
            bottom: pad,
        }
    }

    const fn from_tuple(padding: (usize, usize, usize, usize)) -> Self {
        Self {
            left: padding.0,
            right: padding.1,
            top: padding.2,
            bottom: padding.3,
        }
    }

    const fn is_zero(self) -> bool {
        self.left == 0 && self.right == 0 && self.top == 0 && self.bottom == 0
    }
}

/// Applies edge-replicated padding (`PadMode::Edge`, matching `F.pad(..., mode='replicate')`).
fn pad_replicate(tensor: Tensor<4>, padding: Padding) -> Tensor<4> {
    if padding.is_zero() {
        return tensor;
    }
    tensor.pad(
        [
            (0, 0),
            (0, 0),
            (padding.top, padding.bottom),
            (padding.left, padding.right),
        ],
        PadMode::Edge,
    )
}

fn mean_blur(x: Tensor<4>, kernel_size: usize) -> Tensor<4> {
    assert!(kernel_size > 0, "kernel size must be greater than 0");

    let padding = if kernel_size.is_multiple_of(2) {
        let pad_l = kernel_size / 2 - 1;
        let pad_r = kernel_size / 2;
        Padding::from_tuple((pad_l, pad_r, pad_l, pad_r))
    } else {
        Padding::uniform(kernel_size / 2)
    };

    let x_padded = pad_replicate(x, padding);
    avg_pool2d(
        x_padded,
        AvgPoolOptions::new([kernel_size, kernel_size])
            .with_stride([1, 1])
            .with_count_include_pad(false),
    )
}

fn blur_fusion_estimator(
    image: Tensor<4>,
    fg: Tensor<4>,
    bg: Tensor<4>,
    alpha: Tensor<4>,
    radius: usize,
) -> (Tensor<4>, Tensor<4>) {
    assert!(radius > 0, "blur radius must be greater than 0");

    let blurred_alpha = mean_blur(alpha.clone(), radius);
    let blurred_fga = mean_blur(fg * alpha.clone(), radius);
    let foreground = blurred_fga / (blurred_alpha.clone() + EPSILON);

    let one_minus_alpha = alpha.ones_like() - alpha.clone();
    let blurred_b1a = mean_blur(bg * one_minus_alpha.clone(), radius);
    let one_minus_blurred_alpha = blurred_alpha.ones_like() - blurred_alpha;
    let background = blurred_b1a / (one_minus_blurred_alpha + EPSILON);

    let refined_fg = foreground.clone()
        + alpha.clone() * (image - alpha * foreground - one_minus_alpha * background.clone());
    (refined_fg.clamp(0.0, 1.0), background)
}

/// Refines batched foreground colors using a two-pass blur fusion estimator.
///
/// # Panics
///
/// Panics if `radius == 0`, batch/spatial dimensions differ, or `mask` does not have 1 channel.
///
/// # Examples
///
/// ```rust
/// use birefnet_util::refine_foreground_core;
/// use burn::prelude::*;
///
/// let device = Device::flex();
/// let image = Tensor::<4>::ones([1, 3, 8, 8], &device) * 0.8;
/// let mask = Tensor::<4>::ones([1, 1, 8, 8], &device);
/// let refined = refine_foreground_core(image, mask, 2);
/// assert_eq!(refined.dims(), [1, 3, 8, 8]);
/// ```
#[must_use]
pub fn refine_foreground_core(image: Tensor<4>, mask: Tensor<4>, radius: usize) -> Tensor<4> {
    assert!(radius > 0, "blur radius must be greater than 0");

    let image_dims = image.dims();
    let mask_dims = mask.dims();
    assert_eq!(image_dims[0], mask_dims[0], "batch sizes must match");
    assert_eq!(mask_dims[1], 1, "mask must have 1 channel");
    assert_eq!(image_dims[2], mask_dims[2], "heights must match");
    assert_eq!(image_dims[3], mask_dims[3], "widths must match");

    let (fg_refined, bg_refined) = blur_fusion_estimator(
        image.clone(),
        image.clone(),
        image.clone(),
        mask.clone(),
        radius,
    );
    let (fg_final, _) = blur_fusion_estimator(
        image,
        fg_refined,
        bg_refined,
        mask,
        SECOND_PASS_RADIUS.min(radius),
    );
    fg_final
}

/// Refines foreground colors for a single CHW `[C, H, W]` image and HW `[H, W]` alpha mask.
///
/// # Panics
///
/// Panics if `radius == Some(0)` or spatial dimensions do not match.
///
/// # Examples
///
/// ```rust
/// use birefnet_util::refine_foreground;
/// use burn::prelude::*;
///
/// let device = Device::flex();
/// let image = Tensor::<3>::ones([3, 8, 8], &device) * 0.7;
/// let mask = Tensor::<2>::ones([8, 8], &device) * 0.9;
/// let refined = refine_foreground(image, mask, Some(2));
/// assert_eq!(refined.dims(), [3, 8, 8]);
/// ```
#[must_use]
pub fn refine_foreground(image: Tensor<3>, mask: Tensor<2>, radius: Option<usize>) -> Tensor<3> {
    let radius = radius.unwrap_or(DEFAULT_REFINE_RADIUS);
    assert!(radius > 0, "blur radius must be greater than 0");

    let image_dims = image.dims();
    let mask_dims = mask.dims();
    assert_eq!(
        image_dims[1], mask_dims[0],
        "image height must match mask height"
    );
    assert_eq!(
        image_dims[2], mask_dims[1],
        "image width must match mask width"
    );

    let image_4d = image.unsqueeze_dim::<4>(0);
    let mask_4d = mask.unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0);
    refine_foreground_core(image_4d, mask_4d, radius).squeeze_dim::<3>(0)
}

/// Refines foreground colors for a batch of `[B, C, H, W]` images and `[B, 1, H, W]` masks.
///
/// # Examples
///
/// ```rust
/// use birefnet_util::refine_foreground_batch;
/// use burn::prelude::*;
///
/// let device = Device::flex();
/// let images = Tensor::<4>::ones([2, 3, 8, 8], &device) * 0.6;
/// let masks = Tensor::<4>::ones([2, 1, 8, 8], &device) * 0.8;
/// let refined = refine_foreground_batch(images, masks, Some(2));
/// assert_eq!(refined.dims(), [2, 3, 8, 8]);
/// ```
#[must_use]
pub fn refine_foreground_batch(
    images: Tensor<4>,
    masks: Tensor<4>,
    radius: Option<usize>,
) -> Tensor<4> {
    refine_foreground_core(images, masks, radius.unwrap_or(DEFAULT_REFINE_RADIUS))
}
