use image::{ImageBuffer, Luma, LumaA, Pixel, Primitive, Rgb, Rgba};

pub trait MargeAlpha {
    type Output;
    fn marge_alpha(&self) -> Self::Output;
}

impl<S> MargeAlpha for ImageBuffer<LumaA<S>, Vec<S>>
where
    LumaA<S>: Pixel<Subpixel = S>,
    Luma<S>: Pixel<Subpixel = S>,
    S: Primitive + 'static,
{
    type Output = ImageBuffer<Luma<S>, Vec<S>>;

    fn marge_alpha(&self) -> Self::Output {
        let max = S::DEFAULT_MAX_VALUE.to_f32().unwrap();
        ImageBuffer::from_fn(self.width(), self.height(), |x, y| {
            let LumaA([l, a]) = self.get_pixel(x, y);
            let l_f32 = l.to_f32().unwrap();
            let a_f32 = a.to_f32().unwrap() / max;
            Luma([S::from(l_f32 * a_f32).unwrap()])
        })
    }
}

impl<S> MargeAlpha for ImageBuffer<Rgba<S>, Vec<S>>
where
    Rgba<S>: Pixel<Subpixel = S>,
    Rgb<S>: Pixel<Subpixel = S>,
    S: Primitive + 'static,
{
    type Output = ImageBuffer<Rgb<S>, Vec<S>>;

    fn marge_alpha(&self) -> Self::Output {
        let max = S::DEFAULT_MAX_VALUE.to_f32().unwrap();
        ImageBuffer::from_fn(self.width(), self.height(), |x, y| {
            let Rgba([r, g, b, a]) = self.get_pixel(x, y);
            let a_f32 = a.to_f32().unwrap() / max;
            let merged = |channel: &S| S::from(channel.to_f32().unwrap() * a_f32).unwrap();
            Rgb([merged(r), merged(g), merged(b)])
        })
    }
}
