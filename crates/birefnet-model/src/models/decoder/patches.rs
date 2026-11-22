//! Image-to-patch rearrangement for the decoder's split image-pyramid inputs.

use burn::prelude::*;

/// Splits `image` into a grid of `patch_h x patch_w` tiles stacked along channels.
///
/// For a `[B, C, H, W]` image this returns `[B, cols * rows * C, patch_h, patch_w]`, ordered
/// column-major (all rows of column 0, then column 1, ...) like the upstream
/// `torch.split(..., dim=-1)` / `torch.split(..., dim=-2)` loop. Trailing pixels that do not
/// fill a whole tile are dropped. This is a single pixel-unshuffle (`reshape` + `permute`)
/// instead of `rows * cols` slices and a concatenation.
pub(super) fn patches_batch(image: Tensor<4>, patch_h: usize, patch_w: usize) -> Tensor<4> {
    let [batch, channels, height, width] = image.dims();
    let rows = height / patch_h;
    let cols = width / patch_w;
    let cropped = if rows * patch_h == height && cols * patch_w == width {
        image
    } else {
        image.slice(s![.., .., 0..rows * patch_h, 0..cols * patch_w])
    };

    // [B, C, rows, ph, cols, pw] -> [B, cols, rows, C, ph, pw]
    cropped
        .reshape([batch, channels, rows, patch_h, cols, patch_w])
        .permute([0, 4, 2, 1, 3, 5])
        .reshape([batch, cols * rows * channels, patch_h, patch_w])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference implementation mirroring the upstream double loop.
    fn naive_patches_batch(image: &Tensor<4>, patch_h: usize, patch_w: usize) -> Tensor<4> {
        let [_, _, height, width] = image.dims();
        let mut patches = Vec::new();
        for col in 0..width / patch_w {
            for row in 0..height / patch_h {
                patches.push(image.clone().slice(s![
                    ..,
                    ..,
                    row * patch_h..(row + 1) * patch_h,
                    col * patch_w..(col + 1) * patch_w
                ]));
            }
        }
        Tensor::cat(patches, 1)
    }

    #[test]
    fn pixel_unshuffle_matches_sliced_concatenation() {
        let device = Device::flex();
        // Non-square tiles and a non-divisible width exercise ordering and cropping.
        let image = Tensor::<1, Int>::arange(0..2 * 3 * 8 * 13, &device)
            .float()
            .reshape([2, 3, 8, 13]);

        let fast = patches_batch(image.clone(), 2, 4);
        let naive = naive_patches_batch(&image, 2, 4);

        assert_eq!(fast.dims(), [2, 3 * 4 * 3, 2, 4]);
        fast.into_data().assert_eq(&naive.into_data(), true);
    }
}
