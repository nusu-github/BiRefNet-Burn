//! ImageNet normalization and foreground alpha masking.

use burn::tensor::{Shape, Tensor, TensorData};

use crate::error::UtilError;

/// Standard RGB ImageNet channel means `[R, G, B]` in `[0.0, 1.0]`.
pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];

/// Standard RGB ImageNet channel standard deviations `[R, G, B]`.
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Applies ImageNet channel-wise standardization `(x - mean) / std` to a `[B, 3, H, W]` tensor.
///
/// The statistics are created in the input's float dtype.
///
/// # Errors
///
/// Returns [`UtilError`] if `tensor` does not have 3 channels (`dims[1] != 3`).
pub fn apply_imagenet_normalization(tensor: Tensor<4>) -> Result<Tensor<4>, UtilError> {
    let dims = tensor.dims();
    if dims[1] != 3 {
        return Err(UtilError::invalid_dimensions(format!(
            "ImageNet normalization expects 3 channels, got {} (tensor shape {dims:?})",
            dims[1]
        )));
    }

    let options = (&tensor.device(), tensor.dtype());
    let channel_stats = |values: [f32; 3]| {
        Tensor::<4>::from_data(
            TensorData::new(values.to_vec(), Shape::new([1, 3, 1, 1])),
            options,
        )
    };
    Ok((tensor - channel_stats(IMAGENET_MEAN)) / channel_stats(IMAGENET_STD))
}

/// Applies an alpha mask `[B, 1, H, W]` to an image `[B, C, H, W]`.
///
/// RGB images (`C = 3`) gain the mask as a fourth (alpha) channel; other images are multiplied
/// by the mask, broadcasting a single-channel mask across all channels.
///
/// # Errors
///
/// Returns [`UtilError`] if batch or spatial dimensions differ, or if `mask` channels cannot
/// broadcast to `image`.
pub fn apply_mask(image: Tensor<4>, mask: Tensor<4>) -> Result<Tensor<4>, UtilError> {
    let img_dims = image.dims();
    let mask_dims = mask.dims();
    if img_dims[0] != mask_dims[0] || img_dims[2..] != mask_dims[2..] {
        return Err(UtilError::invalid_dimensions(format!(
            "image shape {img_dims:?} does not match mask shape {mask_dims:?}"
        )));
    }

    match (img_dims[1], mask_dims[1]) {
        (3, 1) => Ok(Tensor::cat(vec![image, mask], 1)),
        (ic, mc) if ic == mc => Ok(image * mask),
        (_, 1) => Ok(image * mask),
        (ic, mc) => Err(UtilError::invalid_dimensions(format!(
            "incompatible channel counts: image has {ic}, mask has {mc}"
        ))),
    }
}
