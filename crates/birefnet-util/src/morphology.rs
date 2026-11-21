//! Morphological image operations (`erosion`, `dilation`, `opening`, `closing`, `gradient`).

use burn::tensor::{Device, Shape, Tensor, TensorData};

/// Binary structuring element kernel paired with its anchor offset.
#[derive(Debug, Clone)]
pub struct StructuringElement {
    kernel: Tensor<2>,
    anchor: (usize, usize),
}

impl StructuringElement {
    /// Creates a structuring element from a binary 2D kernel and `(anchor_y, anchor_x)`.
    #[must_use]
    pub const fn new(kernel: Tensor<2>, anchor: (usize, usize)) -> Self {
        Self { kernel, anchor }
    }

    /// Creates an all-ones rectangular structuring element of shape `[height, width]`.
    ///
    /// # Panics
    ///
    /// Panics if `height == 0` or `width == 0`.
    #[must_use]
    pub fn rectangle(height: usize, width: usize, device: &Device) -> Self {
        assert!(
            height > 0 && width > 0,
            "StructuringElement::rectangle dimensions must be positive, got {height}x{width}"
        );
        let kernel = Tensor::<2>::ones([height, width], device);
        Self::new(kernel, (height / 2, width / 2))
    }

    /// Creates a discrete circular (disk) structuring element of radius `radius`.
    #[must_use]
    pub fn disk(radius: usize, device: &Device) -> Self {
        let size = 2 * radius + 1;
        #[expect(
            clippy::cast_precision_loss,
            reason = "structuring element radii are small kernel dimensions"
        )]
        let center = radius as f32;
        let mut kernel_data = Vec::with_capacity(size * size);

        for row in 0..size {
            for col in 0..size {
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "kernel coordinates fit within exact float precision"
                )]
                let dy = row as f32 - center;
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "kernel coordinates fit within exact float precision"
                )]
                let dx = col as f32 - center;
                kernel_data.push(if dy.hypot(dx) <= center { 1.0_f32 } else { 0.0 });
            }
        }

        let kernel = Tensor::<2>::from_data(
            TensorData::new(kernel_data, Shape::new([size, size])),
            device,
        );
        Self::new(kernel, (radius, radius))
    }

    /// Returns a reference to the underlying binary structuring kernel `[kH, kW]`.
    #[must_use]
    pub const fn kernel(&self) -> &Tensor<2> {
        &self.kernel
    }

    /// Returns the `(anchor_y, anchor_x)` coordinate within the structuring kernel.
    #[must_use]
    pub const fn anchor(&self) -> (usize, usize) {
        self.anchor
    }

    /// Returns the `[height, width]` dimensions of the structuring kernel.
    #[must_use]
    pub fn dims(&self) -> [usize; 2] {
        self.kernel.dims()
    }
}

fn active_kernel_offsets(se: &StructuringElement) -> Vec<(isize, isize)> {
    let [kernel_h, kernel_w] = se.dims();
    let (anchor_y, anchor_x) = se.anchor;
    let kernel_vals = se
        .kernel
        .clone()
        .into_data()
        .try_to_vec::<f32>()
        .expect("structuring element kernel must convert to f32");

    let ay = isize::try_from(anchor_y).expect("anchor_y exceeds isize::MAX");
    let ax = isize::try_from(anchor_x).expect("anchor_x exceeds isize::MAX");
    let mut offsets = Vec::new();
    for ky in 0..kernel_h {
        let dy = isize::try_from(ky).expect("ky exceeds isize::MAX") - ay;
        for kx in 0..kernel_w {
            if kernel_vals[ky * kernel_w + kx] > 0.5 {
                let dx = isize::try_from(kx).expect("kx exceeds isize::MAX") - ax;
                offsets.push((dy, dx));
            }
        }
    }
    offsets
}

fn morph_op(image: Tensor<4>, se: &StructuringElement, is_dilation: bool) -> Tensor<4> {
    let [batch_size, channels, height, width] = image.dims();
    let device = image.device();
    if height == 0 || width == 0 {
        return image;
    }

    let offsets = active_kernel_offsets(se);
    if offsets.is_empty() {
        return if is_dilation {
            Tensor::zeros([batch_size, channels, height, width], &device)
        } else {
            Tensor::ones([batch_size, channels, height, width], &device)
        };
    }

    let values = image
        .into_data()
        .try_to_vec::<f32>()
        .expect("morphological input image must convert to f32");

    let plane_len = height * width;
    let max_y = isize::try_from(height - 1).expect("height exceeds isize::MAX");
    let max_x = isize::try_from(width - 1).expect("width exceeds isize::MAX");
    let mut output = Vec::with_capacity(values.len());

    for plane in values.chunks_exact(plane_len) {
        for y in 0..height {
            let iy = isize::try_from(y).expect("y exceeds isize::MAX");
            for x in 0..width {
                let ix = isize::try_from(x).expect("x exceeds isize::MAX");
                let hit = if is_dilation {
                    offsets.iter().any(|&(dy, dx)| {
                        #[expect(clippy::cast_sign_loss, reason = "clamped to non-negative bounds")]
                        let sy = (iy + dy).clamp(0, max_y) as usize;
                        #[expect(clippy::cast_sign_loss, reason = "clamped to non-negative bounds")]
                        let sx = (ix + dx).clamp(0, max_x) as usize;
                        plane[sy * width + sx] >= 0.5
                    })
                } else {
                    offsets.iter().all(|&(dy, dx)| {
                        #[expect(clippy::cast_sign_loss, reason = "clamped to non-negative bounds")]
                        let sy = (iy + dy).clamp(0, max_y) as usize;
                        #[expect(clippy::cast_sign_loss, reason = "clamped to non-negative bounds")]
                        let sx = (ix + dx).clamp(0, max_x) as usize;
                        plane[sy * width + sx] >= 0.5
                    })
                };
                output.push(if hit { 1.0_f32 } else { 0.0 });
            }
        }
    }

    Tensor::<4>::from_data(
        TensorData::new(output, Shape::new([batch_size, channels, height, width])),
        &device,
    )
}

/// Applies binary morphological erosion to a `[B, C, H, W]` tensor using edge-replicated borders.
#[must_use]
pub fn erosion(image: Tensor<4>, structuring_element: &StructuringElement) -> Tensor<4> {
    morph_op(image, structuring_element, false)
}

/// Applies binary morphological dilation to a `[B, C, H, W]` tensor using edge-replicated borders.
#[must_use]
pub fn dilation(image: Tensor<4>, structuring_element: &StructuringElement) -> Tensor<4> {
    morph_op(image, structuring_element, true)
}

/// Applies morphological opening (`dilation(erosion(image))`) to remove small bright artifacts.
#[must_use]
pub fn opening(image: Tensor<4>, structuring_element: &StructuringElement) -> Tensor<4> {
    let eroded = erosion(image, structuring_element);
    dilation(eroded, structuring_element)
}

/// Applies morphological closing (`erosion(dilation(image))`) to fill small dark holes.
#[must_use]
pub fn closing(image: Tensor<4>, structuring_element: &StructuringElement) -> Tensor<4> {
    let dilated = dilation(image, structuring_element);
    erosion(dilated, structuring_element)
}

/// Computes the morphological gradient (`dilation - erosion`) highlighting object boundaries.
#[must_use]
pub fn gradient(image: Tensor<4>, structuring_element: &StructuringElement) -> Tensor<4> {
    let dilated = dilation(image.clone(), structuring_element);
    let eroded = erosion(image, structuring_element);
    dilated - eroded
}
