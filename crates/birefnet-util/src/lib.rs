//! Tensor, morphological, distance-transform, filtering, and optional image I/O utilities for BiRefNet.

mod array_ops;
mod distance;
mod error;
mod filters;
mod foreground_refiner;
#[cfg(feature = "image-io")]
mod image;
mod morphology;
mod normalize;

#[doc(inline)]
pub use array_ops::{
    argwhere, count_nonzero, cumsum_1d, cumsum_2d_axis0, flip_1d, histogram, std_with_ddof,
};
#[doc(inline)]
pub use distance::{
    bwdist, euclidean_distance_transform, euclidean_distance_transform_simple,
    manhattan_distance_transform,
};
#[doc(inline)]
pub use error::UtilError;
#[doc(inline)]
pub use filters::{custom_filter, gaussian_filter_matlab, gaussian_kernel, matlab_gaussian_2d};
#[doc(inline)]
pub use foreground_refiner::{refine_foreground, refine_foreground_batch, refine_foreground_core};
#[cfg(feature = "image-io")]
#[doc(inline)]
pub use image::{
    common_image_extensions, decode_image_bytes, dynamic_image_to_tensor,
    dynamic_image_to_tensor_with_imagenet_normalization, from_raw_pixels,
    from_raw_pixels_with_normalization, is_extension_supported, is_supported_image_format,
    load_image, load_image_batch, load_image_batch_with_imagenet_normalization,
    load_image_from_reader, load_image_with_imagenet_normalization, resize_image_file,
    supported_formats, tensor_to_dynamic_image,
};
#[doc(inline)]
pub use morphology::{StructuringElement, closing, dilation, erosion, gradient, opening};
#[doc(inline)]
pub use normalize::{IMAGENET_MEAN, IMAGENET_STD, apply_imagenet_normalization, apply_mask};
