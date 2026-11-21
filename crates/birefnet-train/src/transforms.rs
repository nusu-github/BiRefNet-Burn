//! Self-contained image geometric and filter transforms for training augmentations.

use image::{GrayImage, Luma, Rgb, RgbImage};

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "values are clamped into [0.0, 255.0] before casting to u8"
)]
fn round_clamped_u8(value: f64) -> u8 {
    value.clamp(0.0, 255.0).round() as u8
}

#[expect(
    clippy::cast_sign_loss,
    reason = "caller checks value >= 0 before casting"
)]
fn checked_coord(value: isize, limit: u32) -> Option<u32> {
    if value < 0 {
        return None;
    }
    let unsigned = value as usize;
    u32::try_from(unsigned).ok().filter(|&coord| coord < limit)
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "image dimensions fit well within isize range"
)]
fn floor_isize(value: f64) -> isize {
    value.floor() as isize
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "image dimensions fit well within isize range"
)]
fn round_isize(value: f64) -> isize {
    value.round() as isize
}

fn sample_rgb_zero_padded(img: &RgbImage, ix: isize, iy: isize) -> [f64; 3] {
    let Some(ux) = checked_coord(ix, img.width()) else {
        return [0.0; 3];
    };
    let Some(uy) = checked_coord(iy, img.height()) else {
        return [0.0; 3];
    };
    let [r, g, b] = img.get_pixel(ux, uy).0;
    [f64::from(r), f64::from(g), f64::from(b)]
}

/// Rotates an 8-bit grayscale mask about its center using nearest-neighbor sampling.
pub(crate) fn rotate_about_center_luma8(img: &GrayImage, angle_rad: f32) -> GrayImage {
    let (width, height) = img.dimensions();
    if width == 0 || height == 0 {
        return img.clone();
    }

    let (sin_a, cos_a) = f64::from(angle_rad).sin_cos();
    let cx = (f64::from(width) - 1.0) * 0.5;
    let cy = (f64::from(height) - 1.0) * 0.5;

    let mut out = GrayImage::new(width, height);
    for dy in 0..height {
        let rel_y = f64::from(dy) - cy;
        for dx in 0..width {
            let rel_x = f64::from(dx) - cx;
            let src_x = round_isize(rel_x.mul_add(cos_a, rel_y * sin_a) + cx);
            let src_y = round_isize((-rel_x).mul_add(sin_a, rel_y * cos_a) + cy);

            let value = match (checked_coord(src_x, width), checked_coord(src_y, height)) {
                (Some(ux), Some(uy)) => *img.get_pixel(ux, uy),
                _ => Luma([0u8]),
            };
            out.put_pixel(dx, dy, value);
        }
    }

    out
}

/// Rotates an 8-bit RGB image about its center using bilinear interpolation and zero border fill.
pub(crate) fn rotate_about_center_rgb8(img: &RgbImage, angle_rad: f32) -> RgbImage {
    let (width, height) = img.dimensions();
    if width == 0 || height == 0 {
        return img.clone();
    }

    let (sin_a, cos_a) = f64::from(angle_rad).sin_cos();
    let w_f = f64::from(width);
    let h_f = f64::from(height);
    let cx = (w_f - 1.0) * 0.5;
    let cy = (h_f - 1.0) * 0.5;

    let mut out = RgbImage::new(width, height);
    for dy in 0..height {
        let rel_y = f64::from(dy) - cy;
        for dx in 0..width {
            let rel_x = f64::from(dx) - cx;
            let src_x = rel_x.mul_add(cos_a, rel_y * sin_a) + cx;
            let src_y = (-rel_x).mul_add(sin_a, rel_y * cos_a) + cy;

            if src_x <= -1.0 || src_y <= -1.0 || src_x >= w_f || src_y >= h_f {
                out.put_pixel(dx, dy, Rgb([0u8; 3]));
                continue;
            }

            let x0_f = src_x.floor();
            let y0_f = src_y.floor();
            let x0 = floor_isize(x0_f);
            let y0 = floor_isize(y0_f);
            let fx = src_x - x0_f;
            let fy = src_y - y0_f;

            let p00 = sample_rgb_zero_padded(img, x0, y0);
            let p10 = sample_rgb_zero_padded(img, x0 + 1, y0);
            let p01 = sample_rgb_zero_padded(img, x0, y0 + 1);
            let p11 = sample_rgb_zero_padded(img, x0 + 1, y0 + 1);

            let w00 = (1.0 - fx) * (1.0 - fy);
            let w10 = fx * (1.0 - fy);
            let w01 = (1.0 - fx) * fy;
            let w11 = fx * fy;

            let mut rgb = [0u8; 3];
            for ch in 0..3 {
                let interpolated = p00[ch].mul_add(
                    w00,
                    p10[ch].mul_add(w10, p01[ch].mul_add(w01, p11[ch] * w11)),
                );
                rgb[ch] = round_clamped_u8(interpolated);
            }
            out.put_pixel(dx, dy, Rgb(rgb));
        }
    }

    out
}

/// Applies a 3x3 Laplacian sharpen blended with the original RGB image, preserving borders.
pub(crate) fn sharpen_rgb(img: &RgbImage, factor: f32) -> RgbImage {
    let (width, height) = img.dimensions();
    let mut out = img.clone();
    if width < 3 || height < 3 {
        return out;
    }

    let blend = f64::from(factor);
    let keep = 1.0 - blend;

    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let center = img.get_pixel(x, y).0;
            let top = img.get_pixel(x, y - 1).0;
            let bottom = img.get_pixel(x, y + 1).0;
            let left = img.get_pixel(x - 1, y).0;
            let right = img.get_pixel(x + 1, y).0;

            let mut sharpened_pixel = [0u8; 3];
            for ch in 0..3 {
                let c = f64::from(center[ch]);
                let sharp = 5.0 * c
                    - f64::from(top[ch])
                    - f64::from(bottom[ch])
                    - f64::from(left[ch])
                    - f64::from(right[ch]);
                let blended = keep.mul_add(c, blend * sharp);
                sharpened_pixel[ch] = round_clamped_u8(blended);
            }
            out.put_pixel(x, y, Rgb(sharpened_pixel));
        }
    }

    out
}
