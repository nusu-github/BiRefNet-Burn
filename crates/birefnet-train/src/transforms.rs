//! Geometric and filter transforms used by the training augmentations.

use image::{GrayImage, RgbImage};

fn round_clamped_u8(value: f64) -> u8 {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the value is clamped to [0, 255] before rounding"
    )]
    let byte = value.clamp(0.0, 255.0).round() as u8;
    byte
}

/// Inverse rotation about the image center, stepping source coordinates incrementally.
///
/// Calls `write(dst_index, src_x, src_y)` for every destination pixel in row-major order.
fn for_each_source(
    width: u32,
    height: u32,
    angle_rad: f32,
    mut write: impl FnMut(usize, f64, f64),
) {
    let (sin_a, cos_a) = f64::from(angle_rad).sin_cos();
    let cx = (f64::from(width) - 1.0) * 0.5;
    let cy = (f64::from(height) - 1.0) * 0.5;
    let mut dst = 0_usize;
    for dy in 0..height {
        let rel_y = f64::from(dy) - cy;
        // Source of (dx = 0, dy); each step right adds (cos, -sin).
        let mut src_x = (-cx).mul_add(cos_a, rel_y * sin_a) + cx;
        let mut src_y = cx.mul_add(sin_a, rel_y * cos_a) + cy;
        for _ in 0..width {
            write(dst, src_x, src_y);
            dst += 1;
            src_x += cos_a;
            src_y -= sin_a;
        }
    }
}

/// Converts a coordinate to an in-bounds index, or `None` outside `[0, limit)`.
fn index_in(value: f64, limit: u32) -> Option<usize> {
    (value >= 0.0 && value < f64::from(limit)).then(|| {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "the value is checked to lie in [0, limit)"
        )]
        let idx = value as usize;
        idx
    })
}

/// Rotates an 8-bit mask about its center with nearest-neighbor sampling and a zero border.
pub(crate) fn rotate_about_center_luma8(img: &GrayImage, angle_rad: f32) -> GrayImage {
    let (width, height) = img.dimensions();
    let src = img.as_raw();
    let mut out = GrayImage::new(width, height);
    let dst = out.as_mut();
    for_each_source(width, height, angle_rad, |idx, x, y| {
        if let (Some(sx), Some(sy)) = (index_in(x.round(), width), index_in(y.round(), height)) {
            dst[idx] = src[sy * width as usize + sx];
        }
    });
    out
}

/// Rotates an 8-bit RGB image about its center with bilinear sampling and a zero border.
pub(crate) fn rotate_about_center_rgb8(img: &RgbImage, angle_rad: f32) -> RgbImage {
    let (width, height) = img.dimensions();
    let src = img.as_raw();
    let stride = width as usize * 3;
    let texel = |x: f64, y: f64| -> [f64; 3] {
        match (index_in(x, width), index_in(y, height)) {
            (Some(ix), Some(iy)) => {
                let base = iy * stride + ix * 3;
                [src[base], src[base + 1], src[base + 2]].map(f64::from)
            }
            _ => [0.0; 3],
        }
    };

    let mut out = RgbImage::new(width, height);
    let dst = out.as_mut();
    for_each_source(width, height, angle_rad, |idx, x, y| {
        if x <= -1.0 || y <= -1.0 || x >= f64::from(width) || y >= f64::from(height) {
            return;
        }
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let corners = [
            (texel(x0, y0), (1.0 - fx) * (1.0 - fy)),
            (texel(x0 + 1.0, y0), fx * (1.0 - fy)),
            (texel(x0, y0 + 1.0), (1.0 - fx) * fy),
            (texel(x0 + 1.0, y0 + 1.0), fx * fy),
        ];
        for ch in 0..3 {
            let value = corners.iter().map(|(p, w)| p[ch] * w).sum::<f64>();
            dst[idx * 3 + ch] = round_clamped_u8(value);
        }
    });
    out
}

/// Blends a 3x3 Laplacian sharpen with the original (`factor = 0` is the identity); the
/// one-pixel border is left unchanged.
pub(crate) fn sharpen_rgb(img: &RgbImage, factor: f32) -> RgbImage {
    let (width, height) = img.dimensions();
    let mut out = img.clone();
    if width < 3 || height < 3 {
        return out;
    }

    let blend = f64::from(factor);
    let keep = 1.0 - blend;
    let stride = width as usize * 3;
    let rows: Vec<&[u8]> = img.as_raw().chunks_exact(stride).collect();
    let dst = out.as_mut();
    for (y, [top, mid, bottom]) in rows.array_windows::<3>().enumerate() {
        let dst_row = &mut dst[(y + 1) * stride..(y + 2) * stride];
        for i in 3..stride - 3 {
            let center = f64::from(mid[i]);
            let neighbors = f64::from(top[i])
                + f64::from(bottom[i])
                + f64::from(mid[i - 3])
                + f64::from(mid[i + 3]);
            let sharp = 5.0f64.algebraic_mul(center).algebraic_sub(neighbors);
            dst_row[i] = round_clamped_u8(keep.algebraic_mul(center).algebraic_add(blend * sharp));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use image::{Luma, Rgb};

    use super::*;

    #[test]
    fn rotation_by_zero_is_the_identity() {
        let rgb = RgbImage::from_fn(9, 5, |x, y| Rgb([u8::try_from(x * 20 + y).unwrap(), 7, 9]));
        assert_eq!(rotate_about_center_rgb8(&rgb, 0.0), rgb);
        let gray = GrayImage::from_fn(9, 5, |x, y| Luma([u8::try_from(x * 20 + y).unwrap()]));
        assert_eq!(rotate_about_center_luma8(&gray, 0.0), gray);
    }

    #[test]
    fn quarter_turn_moves_pixels_to_the_rotated_position() {
        // At 90° the destination pixel (x, y) samples the source pixel (y, W - 1 - x).
        let mut gray = GrayImage::new(5, 5);
        gray.put_pixel(4, 1, Luma([200]));
        let rotated = rotate_about_center_luma8(&gray, std::f32::consts::FRAC_PI_2);
        let hits: Vec<_> = rotated
            .enumerate_pixels()
            .filter(|(.., p)| p.0[0] == 200)
            .map(|(x, y, _)| (x, y))
            .collect();
        assert_eq!(hits, vec![(3, 4)]);
    }

    #[test]
    fn sharpening_keeps_flat_regions_and_amplifies_edges() {
        let flat = RgbImage::from_pixel(6, 6, Rgb([90, 90, 90]));
        assert_eq!(sharpen_rgb(&flat, 1.0), flat);

        let edge = RgbImage::from_fn(
            6,
            6,
            |x, _| if x < 3 { Rgb([50; 3]) } else { Rgb([150; 3]) },
        );
        let sharpened = sharpen_rgb(&edge, 1.0);
        // The dark side of the edge gets darker, the bright side brighter; borders are unchanged.
        assert!(sharpened.get_pixel(2, 2).0[0] < 50);
        assert!(sharpened.get_pixel(3, 2).0[0] > 150);
        assert_eq!(sharpened.get_pixel(0, 0), edge.get_pixel(0, 0));
        assert_eq!(sharpen_rgb(&edge, 0.0), edge);
    }
}
