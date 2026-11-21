//! High-level `BiRefNet` inference pipeline and output conversion.

use std::path::Path;

use birefnet_model::BiRefNet;
use birefnet_util::{
    apply_imagenet_normalization, apply_mask, dynamic_image_to_tensor, load_image,
    refine_foreground_core, tensor_to_dynamic_image,
};
use burn::tensor::{Device, Tensor, activation::sigmoid, ops::InterpolateMode};
use image::{DynamicImage, GrayImage};

use crate::{
    error::InferenceError,
    postprocessing::{PostprocessConfig, mask_to_u8, postprocess_mask, resize_tensor},
};

const DEFAULT_INPUT_SIZE: (usize, usize) = (1024, 1024);
const DEFAULT_REFINE_RADIUS: usize = 90;

/// Configuration for [`Segmenter`].
#[derive(Debug, Clone, PartialEq)]
pub struct SegmenterConfig {
    /// Model inference `(height, width)` resolution before restoring original image size.
    pub input_size: (usize, usize),
    /// Whether to apply multi-pass blur-fusion foreground color refinement.
    pub refine_foreground: bool,
    /// Blur radius used when `refine_foreground` is enabled (values `0` skip refinement).
    pub foreground_refine_radius: usize,
    /// Optional mask postprocessing pipeline applied after resizing back to source size.
    pub postprocess: PostprocessConfig,
}

impl Default for SegmenterConfig {
    fn default() -> Self {
        Self {
            input_size: DEFAULT_INPUT_SIZE,
            refine_foreground: true,
            foreground_refine_radius: DEFAULT_REFINE_RADIUS,
            postprocess: PostprocessConfig::default(),
        }
    }
}

/// Segmentation predictions paired with source image context at its original resolution.
#[derive(Debug, Clone)]
pub struct SegmentationOutput {
    source_image: Tensor<4>,
    mask: Tensor<4>,
    foreground: Option<Tensor<4>>,
    original_size: (usize, usize),
}

impl SegmentationOutput {
    /// Returns the predicted alpha probability mask `[1, 1, H, W]` in `[0.0, 1.0]`.
    #[must_use]
    pub const fn mask(&self) -> &Tensor<4> {
        &self.mask
    }

    /// Returns the refined RGB foreground tensor `[1, 3, H, W]` if foreground refinement ran.
    #[must_use]
    pub const fn foreground(&self) -> Option<&Tensor<4>> {
        self.foreground.as_ref()
    }

    /// Returns the original source image `(height, width)`.
    #[must_use]
    pub const fn original_size(&self) -> (usize, usize) {
        self.original_size
    }

    /// Composites the foreground image with the predicted alpha mask as an RGBA [`DynamicImage`].
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError`] if mask application or tensor-to-image conversion fails.
    pub fn to_rgba_image(&self) -> Result<DynamicImage, InferenceError> {
        let rgb = self
            .foreground
            .as_ref()
            .unwrap_or(&self.source_image)
            .clone();
        let rgba = apply_mask(rgb, self.mask.clone())?;
        Ok(tensor_to_dynamic_image(rgba, false)?)
    }

    /// Converts the predicted probability mask into an 8-bit grayscale [`GrayImage`].
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError`] if extracting the mask tensor fails.
    pub fn to_mask_image(&self) -> Result<GrayImage, InferenceError> {
        mask_to_u8(&self.mask)
    }
}

/// High-level segmentation runner wrapping a [`BiRefNet`] model and preprocessing/postprocessing.
#[derive(Debug)]
pub struct Segmenter {
    model: BiRefNet,
    config: SegmenterConfig,
}

impl Segmenter {
    /// Wraps `model` with the given inference `config`.
    #[must_use]
    pub const fn new(model: BiRefNet, config: SegmenterConfig) -> Self {
        Self { model, config }
    }

    /// Returns a reference to the segmenter configuration.
    #[must_use]
    pub const fn config(&self) -> &SegmenterConfig {
        &self.config
    }

    /// Returns a reference to the underlying [`BiRefNet`] model.
    #[must_use]
    pub const fn model(&self) -> &BiRefNet {
        &self.model
    }

    /// Unwraps and returns the underlying [`BiRefNet`] model.
    #[must_use]
    pub fn into_model(self) -> BiRefNet {
        self.model
    }

    /// Segments an RGB float tensor `[1, 3, H, W]` with values in `[0.0, 1.0]`.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError`] if tensor shapes or `input_size` are invalid, or if the
    /// forward pass fails.
    pub fn segment_tensor(&self, image: Tensor<4>) -> Result<SegmentationOutput, InferenceError> {
        let [batch, channels, orig_h, orig_w] = image.dims();
        if batch != 1 || channels != 3 || orig_h == 0 || orig_w == 0 {
            return Err(InferenceError::invalid_input(format!(
                "segment_tensor expects non-empty [1, 3, H, W] RGB tensor, got [{batch}, {channels}, {orig_h}, {orig_w}]"
            )));
        }

        let (in_h, in_w) = self.config.input_size;
        if in_h == 0 || in_w == 0 {
            return Err(InferenceError::invalid_input(format!(
                "segmenter input_size must be positive, got {in_h}x{in_w}"
            )));
        }

        let normalized = apply_imagenet_normalization(image.clone())?;
        let model_input = resize_tensor(normalized, in_h, in_w, InterpolateMode::Bicubic);

        let logits = self.model.forward(model_input)?;
        let prob = sigmoid(logits);
        let restored = resize_tensor(prob, orig_h, orig_w, InterpolateMode::Bicubic);
        let mask = postprocess_mask(restored, &self.config.postprocess);

        let foreground = (self.config.refine_foreground
            && self.config.foreground_refine_radius > 0)
            .then(|| {
                refine_foreground_core(
                    image.clone(),
                    mask.clone(),
                    self.config.foreground_refine_radius,
                )
            });

        Ok(SegmentationOutput {
            source_image: image,
            mask,
            foreground,
            original_size: (orig_h, orig_w),
        })
    }

    /// Segments a decoded [`DynamicImage`] on `device`.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError`] if image conversion or model execution fails.
    pub fn segment_image(
        &self,
        image: &DynamicImage,
        device: &Device,
    ) -> Result<SegmentationOutput, InferenceError> {
        let tensor = dynamic_image_to_tensor(image.clone(), device)?;
        self.segment_tensor(tensor)
    }

    /// Loads an image file from `path` on disk and runs segmentation on `device`.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError`] if file decoding or segmentation fails.
    pub fn segment_file(
        &self,
        path: impl AsRef<Path>,
        device: &Device,
    ) -> Result<SegmentationOutput, InferenceError> {
        let tensor = load_image(path, device)?;
        self.segment_tensor(tensor)
    }
}
