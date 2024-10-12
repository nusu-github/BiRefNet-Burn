use std::ops::{Div, Mul};

use image::{GenericImageView, ImageBuffer, Luma, LumaA, Pixel, Primitive};
use num_traits::ToPrimitive;

pub trait ClipMinimumBorder {
    fn clip_minimum_border(self, iterations: usize, threshold: u8) -> Self;
}

impl<P, S> ClipMinimumBorder for ImageBuffer<P, Vec<S>>
where
    P: Pixel<Subpixel = S> + 'static,
    S: Primitive + 'static,
{
    fn clip_minimum_border(mut self, iterations: usize, threshold: u8) -> Self {
        for i in 0..iterations {
            let corners = self.extract_corners();
            let background = &corners[i % 4];
            let [x, y, w, h] = self.find_content_bounds(background, threshold);

            if w == 0 || h == 0 {
                break;
            }

            self = self.view(x, y, w, h).to_image();
        }
        self
    }
}

trait ImageProcessing<P: Pixel> {
    fn extract_corners(&self) -> [Luma<P::Subpixel>; 4];
    fn find_content_bounds(&self, background: &Luma<P::Subpixel>, threshold: u8) -> [u32; 4];
    fn calculate_pixel_difference(&self, pixel: &P, background: &Luma<P::Subpixel>, max: f32)
        -> u8;
}

impl<P: Pixel> ImageProcessing<P> for ImageBuffer<P, Vec<P::Subpixel>>
where
    <P as Pixel>::Subpixel: 'static,
{
    fn extract_corners(&self) -> [Luma<P::Subpixel>; 4] {
        let (width, height) = self.dimensions();
        [
            merge_alpha(self.get_pixel(0, 0).to_luma_alpha()),
            merge_alpha(self.get_pixel(width.saturating_sub(1), 0).to_luma_alpha()),
            merge_alpha(self.get_pixel(0, height.saturating_sub(1)).to_luma_alpha()),
            merge_alpha(
                self.get_pixel(width.saturating_sub(1), height.saturating_sub(1))
                    .to_luma_alpha(),
            ),
        ]
    }

    fn find_content_bounds(&self, background: &Luma<P::Subpixel>, threshold: u8) -> [u32; 4] {
        let max = P::Subpixel::DEFAULT_MAX_VALUE.to_f32().unwrap();
        let (width, height) = self.dimensions();
        let mut bounds = [width, height, 0, 0]; // [x1, y1, x2, y2]

        for (x, y, pixel) in self.enumerate_pixels() {
            let diff = self.calculate_pixel_difference(pixel, background, max);
            if diff > threshold {
                update_bounds(&mut bounds, x, y);
            }
        }

        [
            bounds[0],
            bounds[1],
            bounds[2].saturating_sub(bounds[0]),
            bounds[3].saturating_sub(bounds[1]),
        ]
    }

    fn calculate_pixel_difference(
        &self,
        pixel: &P,
        background: &Luma<P::Subpixel>,
        max: f32,
    ) -> u8 {
        let pixel_value = merge_alpha(pixel.to_luma_alpha())[0]
            .to_f32()
            .unwrap()
            .div(max)
            .mul(255.0);
        let background_value = background[0].to_f32().unwrap().div(max).mul(255.0);
        pixel_value
            .to_u8()
            .unwrap()
            .abs_diff(background_value.to_u8().unwrap())
    }
}

fn merge_alpha<S>(image: LumaA<S>) -> Luma<S>
where
    LumaA<S>: Pixel<Subpixel = S>,
    Luma<S>: Pixel<Subpixel = S>,
    S: Primitive + 'static,
{
    unsafe {
        // SAFETY: u8,u16,f32 to f32 is safe
        let max = S::DEFAULT_MAX_VALUE.to_f32().unwrap_unchecked();
        let LumaA([l, a]) = image;
        let l = l.to_f32().unwrap_unchecked();
        let a = a.to_f32().unwrap_unchecked() / max;
        let l = S::from(l * a).unwrap_unchecked();
        Luma([l])
    }
}

fn update_bounds(bounds: &mut [u32; 4], x: u32, y: u32) {
    bounds[0] = bounds[0].min(x);
    bounds[1] = bounds[1].min(y);
    bounds[2] = bounds[2].max(x);
    bounds[3] = bounds[3].max(y);
}
