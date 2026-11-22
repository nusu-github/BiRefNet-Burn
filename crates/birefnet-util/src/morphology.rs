//! Binary morphology (`erosion`, `dilation`, `opening`, `closing`, `gradient`) on the device.
//!
//! Inputs are binarized at `0.5`, borders are edge-replicated, and outputs are `{0, 1}` maps in
//! the input float dtype. Rectangular elements run as min/max pooling; other shapes count hits
//! with a depth-wise convolution.

use burn::tensor::{
    DType, Shape, Tensor, TensorData,
    module::{conv2d, max_pool2d},
    ops::{ConvOptions, MaxPoolOptions, PadMode},
};

use crate::error::UtilError;

/// Binary structuring element: an `[height, width]` mask and its `(row, col)` anchor.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StructuringElement {
    mask: Vec<bool>,
    dims: [usize; 2],
    anchor: (usize, usize),
}

impl StructuringElement {
    /// Creates a structuring element from a row-major `[height, width]` mask.
    ///
    /// # Errors
    ///
    /// Returns [`UtilError`] if `mask.len() != height * width`, a dimension is zero, or the
    /// anchor lies outside the mask.
    pub fn new(
        mask: Vec<bool>,
        dims: [usize; 2],
        anchor: (usize, usize),
    ) -> Result<Self, UtilError> {
        let [height, width] = dims;
        if height == 0 || width == 0 || mask.len() != height * width {
            return Err(UtilError::invalid_dimensions(format!(
                "structuring element mask of length {} does not match positive dims {dims:?}",
                mask.len()
            )));
        }
        if anchor.0 >= height || anchor.1 >= width {
            return Err(UtilError::invalid_input(format!(
                "anchor {anchor:?} lies outside a {height}x{width} structuring element"
            )));
        }
        Ok(Self { mask, dims, anchor })
    }

    /// Creates an all-ones `[height, width]` rectangle anchored at its center.
    ///
    /// # Panics
    ///
    /// Panics if `height == 0` or `width == 0`.
    #[must_use]
    pub fn rectangle(height: usize, width: usize) -> Self {
        assert!(
            height > 0 && width > 0,
            "StructuringElement::rectangle dimensions must be positive, got {height}x{width}"
        );
        Self {
            mask: vec![true; height * width],
            dims: [height, width],
            anchor: (height / 2, width / 2),
        }
    }

    /// Creates a `(2 * radius + 1)`-wide disk containing every offset with `dy² + dx² <= radius²`.
    #[must_use]
    pub fn disk(radius: usize) -> Self {
        let size = 2 * radius + 1;
        let r_sq = radius * radius;
        let mask = (0..size * size)
            .map(|idx| {
                let dy = (idx / size).abs_diff(radius);
                let dx = (idx % size).abs_diff(radius);
                dy * dy + dx * dx <= r_sq
            })
            .collect();
        Self {
            mask,
            dims: [size, size],
            anchor: (radius, radius),
        }
    }

    /// Returns the row-major binary mask.
    #[must_use]
    pub fn mask(&self) -> &[bool] {
        &self.mask
    }

    /// Returns the `(anchor_y, anchor_x)` coordinate within the mask.
    #[must_use]
    pub const fn anchor(&self) -> (usize, usize) {
        self.anchor
    }

    /// Returns the `[height, width]` dimensions of the mask.
    #[must_use]
    pub const fn dims(&self) -> [usize; 2] {
        self.dims
    }

    fn active_count(&self) -> usize {
        self.mask.iter().filter(|&&on| on).count()
    }

    fn is_full(&self) -> bool {
        self.mask.iter().all(|&on| on)
    }

    /// `(top, bottom, left, right)` edge padding that centres windows on the anchor.
    const fn padding(&self) -> [(usize, usize); 2] {
        let [height, width] = self.dims;
        let (ay, ax) = self.anchor;
        [(ay, height - 1 - ay), (ax, width - 1 - ax)]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Op {
    Erode,
    Dilate,
}

fn morph_op(image: Tensor<4>, se: &StructuringElement, op: Op) -> Tensor<4> {
    let [batch, channels, height, width] = image.dims();
    let dtype = image.dtype();
    if height == 0 || width == 0 {
        return image;
    }
    let active = se.active_count();
    if active == 0 {
        // Empty element: no neighbor can be set (dilation) and every "all" test is vacuous.
        return match op {
            Op::Dilate => image.zeros_like(),
            Op::Erode => image.ones_like(),
        };
    }

    // Counts are small integers; f32 represents them exactly even for half-precision inputs.
    let binary = image.greater_equal_elem(0.5).float().cast(DType::F32);
    let [pad_y, pad_x] = se.padding();
    let padded = binary.pad([(0, 0), (0, 0), pad_y, pad_x], PadMode::Edge);

    let result = if se.is_full() {
        let pool = |x: Tensor<4>| max_pool2d(x, MaxPoolOptions::new(se.dims).with_stride([1, 1]));
        match op {
            Op::Dilate => pool(padded),
            Op::Erode => pool(padded.neg()).neg(),
        }
    } else {
        let [kh, kw] = se.dims;
        let weights: Vec<f32> = se.mask.iter().map(|&on| f32::from(u8::from(on))).collect();
        let kernel = Tensor::<4>::from_data(
            TensorData::new(weights, Shape::new([1, 1, kh, kw])),
            (&padded.device(), DType::F32),
        )
        .repeat_dim(0, channels);
        let hits = conv2d(
            padded,
            kernel,
            None,
            ConvOptions::new([1, 1], [0, 0], [1, 1], channels),
        );
        match op {
            Op::Dilate => hits.greater_elem(0.5).float(),
            Op::Erode => {
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "structuring elements have far fewer than 2^24 active cells"
                )]
                let needed = active as f32 - 0.5;
                hits.greater_equal_elem(needed).float()
            }
        }
    };

    debug_assert_eq!(result.dims(), [batch, channels, height, width]);
    result.cast(dtype)
}

/// Binary erosion: a pixel stays set only if every active neighbor is `>= 0.5`.
#[must_use]
pub fn erosion(image: Tensor<4>, structuring_element: &StructuringElement) -> Tensor<4> {
    morph_op(image, structuring_element, Op::Erode)
}

/// Binary dilation: a pixel is set if any active neighbor is `>= 0.5`.
#[must_use]
pub fn dilation(image: Tensor<4>, structuring_element: &StructuringElement) -> Tensor<4> {
    morph_op(image, structuring_element, Op::Dilate)
}

/// Morphological opening (`dilation(erosion(image))`), removing small foreground specks.
#[must_use]
pub fn opening(image: Tensor<4>, structuring_element: &StructuringElement) -> Tensor<4> {
    dilation(erosion(image, structuring_element), structuring_element)
}

/// Morphological closing (`erosion(dilation(image))`), filling small background holes.
#[must_use]
pub fn closing(image: Tensor<4>, structuring_element: &StructuringElement) -> Tensor<4> {
    erosion(dilation(image, structuring_element), structuring_element)
}

/// Morphological gradient (`dilation - erosion`), marking object boundaries.
#[must_use]
pub fn gradient(image: Tensor<4>, structuring_element: &StructuringElement) -> Tensor<4> {
    let dilated = dilation(image.clone(), structuring_element);
    dilated - erosion(image, structuring_element)
}
