//! High-level `BiRefNet` inference pipeline and output conversion.

use std::{
    io::{BufRead, Seek},
    num::NonZeroUsize,
    path::Path,
};

use birefnet_model::BiRefNet;
use birefnet_util::{
    foreground::refine_foreground_core,
    image::{
        decode_image_bytes, dynamic_image_to_tensor, load_image, load_image_from_reader,
        tensor_to_dynamic_image,
    },
    normalize::{apply_imagenet_normalization, apply_mask},
};
use burn::{
    module::Module,
    tensor::{Device, Tensor, activation::sigmoid, ops::InterpolateMode},
};
use image::{DynamicImage, GrayImage, ImageFormat};

use crate::{
    error::InferenceError,
    postprocess::{PostprocessConfig, mask_to_u8, postprocess_mask, resize_tensor},
};

/// Configuration for [`Segmenter`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SegmenterConfig {
    input_size: [NonZeroUsize; 2],
    foreground_refine_radius: Option<NonZeroUsize>,
    postprocess: PostprocessConfig,
}

impl Default for SegmenterConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// BiRefNet's training resolution.
const DEFAULT_INPUT_SIZE: [NonZeroUsize; 2] = [NonZeroUsize::new(1024).unwrap(); 2];
/// First-pass blur radius of the reference foreground refinement.
const DEFAULT_REFINE_RADIUS: NonZeroUsize = NonZeroUsize::new(90).unwrap();

impl SegmenterConfig {
    /// `1024 x 1024` model input, foreground refinement with radius 90, no postprocessing.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            input_size: DEFAULT_INPUT_SIZE,
            foreground_refine_radius: Some(DEFAULT_REFINE_RADIUS),
            postprocess: PostprocessConfig::new(),
        }
    }

    /// Sets the `[height, width]` the image is resized to before the forward pass.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError`] if either dimension is zero.
    pub fn with_input_size(self, [height, width]: [usize; 2]) -> Result<Self, InferenceError> {
        match (NonZeroUsize::new(height), NonZeroUsize::new(width)) {
            (Some(h), Some(w)) => Ok(Self {
                input_size: [h, w],
                ..self
            }),
            _ => Err(InferenceError::invalid_config(format!(
                "input size must be positive, got {height}x{width}"
            ))),
        }
    }

    /// Enables blur-fusion foreground refinement with `radius`, or disables it with `None`.
    #[must_use]
    pub const fn with_foreground_refinement(self, radius: Option<NonZeroUsize>) -> Self {
        Self {
            foreground_refine_radius: radius,
            ..self
        }
    }

    /// Sets the mask postprocessing applied at the source resolution.
    #[must_use]
    pub const fn with_postprocess(self, postprocess: PostprocessConfig) -> Self {
        Self {
            postprocess,
            ..self
        }
    }

    /// Returns the model input `[height, width]`.
    #[must_use]
    pub const fn input_size(&self) -> [usize; 2] {
        [self.input_size[0].get(), self.input_size[1].get()]
    }

    /// Returns the foreground refinement radius, or `None` if refinement is disabled.
    #[must_use]
    pub const fn foreground_refine_radius(&self) -> Option<NonZeroUsize> {
        self.foreground_refine_radius
    }

    /// Returns the mask postprocessing configuration.
    #[must_use]
    pub const fn postprocess(&self) -> &PostprocessConfig {
        &self.postprocess
    }
}

/// Segmentation predictions paired with source image context at its original resolution.
#[derive(Debug, Clone)]
pub struct SegmentationOutput {
    source_image: Tensor<4>,
    mask: Tensor<4>,
    foreground: Option<Tensor<4>>,
}

impl SegmentationOutput {
    /// Returns the predicted alpha probability mask `[1, 1, H, W]` in `[0, 1]`.
    #[must_use]
    pub const fn mask(&self) -> &Tensor<4> {
        &self.mask
    }

    /// Returns the refined RGB foreground `[1, 3, H, W]` if foreground refinement ran.
    #[must_use]
    pub const fn foreground(&self) -> Option<&Tensor<4>> {
        self.foreground.as_ref()
    }

    /// Returns the source image `[height, width]`.
    #[must_use]
    pub fn original_size(&self) -> [usize; 2] {
        let [.., height, width] = self.mask.dims();
        [height, width]
    }

    /// Composites the (refined) foreground with the predicted alpha as an RGBA image.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError`] if masking or tensor-to-image conversion fails.
    pub fn to_rgba_image(&self) -> Result<DynamicImage, InferenceError> {
        let rgb = self.foreground.as_ref().unwrap_or(&self.source_image);
        let rgba = apply_mask(rgb.clone(), self.mask.clone())?;
        Ok(tensor_to_dynamic_image(rgba, false)?)
    }

    /// Converts the predicted probability mask into an 8-bit [`GrayImage`].
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError`] if the mask cannot be read back from its device.
    pub fn to_mask_image(&self) -> Result<GrayImage, InferenceError> {
        mask_to_u8(&self.mask)
    }
}

/// High-level segmentation runner wrapping a [`BiRefNet`] model with pre- and postprocessing.
///
/// Cloning is cheap: model parameters are reference-counted device tensors.
#[derive(Debug, Clone)]
pub struct Segmenter {
    model: BiRefNet,
    config: SegmenterConfig,
}

impl Segmenter {
    /// Wraps `model` in validation mode (no autodiff, no dropout) with `config`.
    #[must_use]
    pub fn new(model: &BiRefNet, config: SegmenterConfig) -> Self {
        Self {
            model: model.valid(),
            config,
        }
    }

    /// Returns the segmenter configuration.
    #[must_use]
    pub const fn config(&self) -> &SegmenterConfig {
        &self.config
    }

    /// Returns the underlying [`BiRefNet`] model.
    #[must_use]
    pub const fn model(&self) -> &BiRefNet {
        &self.model
    }

    /// Unwraps and returns the underlying [`BiRefNet`] model.
    #[must_use]
    pub fn into_model(self) -> BiRefNet {
        self.model
    }

    /// Segments an RGB tensor `[1, 3, H, W]` with values in `[0, 1]`.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError`] if the tensor shape is invalid or the forward pass or
    /// postprocessing fails.
    pub fn segment_tensor(&self, image: Tensor<4>) -> Result<SegmentationOutput, InferenceError> {
        let [batch, channels, orig_h, orig_w] = image.dims();
        if batch != 1 || channels != 3 || orig_h == 0 || orig_w == 0 {
            return Err(InferenceError::invalid_input(format!(
                "segment_tensor expects a non-empty [1, 3, H, W] RGB tensor, got [{batch}, {channels}, {orig_h}, {orig_w}]"
            )));
        }
        // The model runs in validation mode; keep inputs off the autodiff graph as well.
        let image = image.without_autodiff();

        let [in_h, in_w] = self.config.input_size();
        let normalized = apply_imagenet_normalization(image.clone())?;
        let model_input = resize_tensor(normalized, in_h, in_w, InterpolateMode::Bicubic);

        let prob = sigmoid(self.model.forward(model_input)?);
        let restored = resize_tensor(prob, orig_h, orig_w, InterpolateMode::Bicubic);
        let mask = postprocess_mask(restored, &self.config.postprocess)?;

        let foreground = self
            .config
            .foreground_refine_radius
            .map(|radius| refine_foreground_core(image.clone(), mask.clone(), radius.get()));

        Ok(SegmentationOutput {
            source_image: image,
            mask,
            foreground,
        })
    }

    /// Segments a decoded [`DynamicImage`] on `device`.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError`] if segmentation fails.
    pub fn segment_image(
        &self,
        image: DynamicImage,
        device: &Device,
    ) -> Result<SegmentationOutput, InferenceError> {
        self.segment_tensor(dynamic_image_to_tensor(image, device))
    }

    /// Decodes an image from `reader` (format guessed unless given) and segments it.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError`] if decoding or segmentation fails.
    pub fn segment_reader<R: BufRead + Seek>(
        &self,
        reader: R,
        format: Option<ImageFormat>,
        device: &Device,
    ) -> Result<SegmentationOutput, InferenceError> {
        self.segment_tensor(load_image_from_reader(reader, format, device)?)
    }

    /// Decodes encoded image bytes and segments them.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError`] if decoding or segmentation fails.
    pub fn segment_bytes(
        &self,
        bytes: &[u8],
        device: &Device,
    ) -> Result<SegmentationOutput, InferenceError> {
        self.segment_tensor(decode_image_bytes(bytes, device)?)
    }

    /// Loads an image file and segments it.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError`] if file decoding or segmentation fails.
    pub fn segment_file(
        &self,
        path: impl AsRef<Path>,
        device: &Device,
    ) -> Result<SegmentationOutput, InferenceError> {
        self.segment_tensor(load_image(path, device)?)
    }
}

const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Segmenter>();
    assert_send_sync::<SegmentationOutput>();
};
