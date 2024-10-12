use anyhow::{anyhow, ensure, Result};
use image::{GenericImageView, ImageBuffer, Luma, Pixel, Primitive, Rgb, Rgba};

pub trait AlphaMaskImage<S>
where
    Rgba<S>: Pixel<Subpixel = S>,
    S: Primitive + 'static,
{
    fn add_alpha_mask<I, SM>(&self, mask: &I) -> Result<ImageBuffer<Rgba<S>, Vec<S>>>
    where
        I: GenericImageView<Pixel = Luma<SM>>,
        SM: Primitive + 'static;
}

impl<S> AlphaMaskImage<S> for ImageBuffer<Rgb<S>, Vec<S>>
where
    Rgb<S>: Pixel<Subpixel = S>,
    Rgba<S>: Pixel<Subpixel = S>,
    S: Primitive + 'static,
{
    fn add_alpha_mask<I, SM>(&self, mask: &I) -> Result<ImageBuffer<Rgba<S>, Vec<S>>>
    where
        I: GenericImageView<Pixel = Luma<SM>>,
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
                let Luma([alpha]) = mask_pixel.2;
                let alpha = S::from(alpha.to_f32().unwrap() / sm_max * si_max).unwrap();
                [red, green, blue, alpha]
            })
            .collect();

        ImageBuffer::from_raw(self.width(), self.height(), processed_pixels)
            .ok_or_else(|| anyhow!("Failed to create ImageBuffer from processed pixels"))
    }
}
