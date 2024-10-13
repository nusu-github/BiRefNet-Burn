use anyhow::{anyhow, ensure, Result};
use image::{ImageBuffer, Luma, Pixel, Primitive, Rgb, Rgba};

use crate::imageops_ai::Image;

pub trait AlphaMaskImage<S>
where
    Rgba<S>: Pixel<Subpixel = S>,
    S: Primitive + 'static,
{
    fn add_alpha_mask<SM>(&self, mask: &Image<Luma<SM>>) -> Result<Image<Rgba<S>>>
    where
        SM: Primitive + 'static;
}

impl<S> AlphaMaskImage<S> for Image<Rgb<S>>
where
    Rgb<S>: Pixel<Subpixel = S>,
    Rgba<S>: Pixel<Subpixel = S>,
    S: Primitive + 'static,
{
    fn add_alpha_mask<SM>(&self, mask: &Image<Luma<SM>>) -> Result<Image<Rgba<S>>>
    where
        SM: Primitive + 'static,
    {
        ensure!(
            self.dimensions() == mask.dimensions(),
            "Image and mask dimensions do not match"
        );

        let si_max = S::DEFAULT_MAX_VALUE.to_f32().unwrap();
        let sm_max = SM::DEFAULT_MAX_VALUE.to_f32().unwrap();

        let processed_pixels = self
            .pixels()
            .zip(mask.pixels())
            .flat_map(|(&image_pixel, mask_pixel)| {
                let Rgb([red, green, blue]) = image_pixel;
                let Luma([alpha]) = mask_pixel;
                let alpha = alpha.to_f32().unwrap() / sm_max * si_max;
                [red, green, blue, S::from(alpha).unwrap()]
            })
            .collect();

        ImageBuffer::from_raw(self.width(), self.height(), processed_pixels)
            .ok_or_else(|| anyhow!("Failed to create ImageBuffer from processed pixels"))
    }
}
