use anyhow::Result;
use burn::{
    prelude::*,
    tensor::{
        module::interpolate,
        ops::{InterpolateMode, InterpolateOptions},
    },
};
use image::{flat::SampleLayout, ImageBuffer, Luma, Pixel, Primitive, RgbImage};

#[derive(Debug, Clone)]
pub struct PreprocessConfig {
    pub image_size: u32,
    pub maintain_aspect_ratio: bool,
    pub padding_color: [f32; 3],
}

impl Default for PreprocessConfig {
    fn default() -> Self {
        Self {
            image_size: 1024,
            maintain_aspect_ratio: true,
            padding_color: [0.0, 0.0, 0.0], // Black padding
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct PostprocessConfig {
    pub apply_morphology: bool,
    pub smooth_edges: bool,
    pub remove_noise: bool,
}

pub trait IntoTensor3<B: Backend> {
    type Out;

    fn into_tensor3(self, device: &Device<B>) -> Self::Out;
}

impl<B: Backend, P> IntoTensor3<B> for ImageBuffer<P, Vec<P::Subpixel>>
where
    P: Pixel + 'static,
    f32: From<P::Subpixel>,
{
    type Out = Tensor<B, 3>;

    fn into_tensor3(self, device: &Device<B>) -> Self::Out {
        let SampleLayout {
            channels,
            height,
            width,
            ..
        } = self.sample_layout();
        let shape = Vec::from([height as usize, width as usize, channels as usize]);
        Self::Out::from_data(
            TensorData::new(self.iter().map(|x| f32::from(*x)).collect(), shape)
                .convert::<B::FloatElem>(),
            device,
        )
    }
}

pub fn preprocess<B: Backend, P>(
    image: &ImageBuffer<P, Vec<P::Subpixel>>,
    image_size: u32,
    device: &Device<B>,
) -> Tensor<B, 3>
where
    P: Pixel + 'static,
    f32: From<P::Subpixel>,
{
    let tensor = image.clone().into_tensor3(device)
        .permute([2, 0, 1]) // [C, H, W]
        / f32::from(P::Subpixel::DEFAULT_MAX_VALUE);

    let tensor = interpolate(
        tensor.unsqueeze(),
        [image_size as usize, image_size as usize],
        InterpolateOptions::new(InterpolateMode::Bilinear),
    );

    tensor.squeeze(0)
}

pub fn preprocess_with_config<B: Backend, P>(
    image: &ImageBuffer<P, Vec<P::Subpixel>>,
    config: &PreprocessConfig,
    device: &Device<B>,
) -> Tensor<B, 3>
where
    P: Pixel + 'static,
    f32: From<P::Subpixel>,
{
    let tensor = image.clone().into_tensor3(device)
        .permute([2, 0, 1]) // [C, H, W]
        / f32::from(P::Subpixel::DEFAULT_MAX_VALUE);

    let tensor = interpolate(
        tensor.unsqueeze(),
        [config.image_size as usize, config.image_size as usize],
        InterpolateOptions::new(InterpolateMode::Bilinear),
    );

    tensor.squeeze(0)
}

pub fn postprocess_mask<B: Backend>(
    mask: Tensor<B, 3>,
    width: u32,
    height: u32,
) -> ImageBuffer<Luma<f32>, Vec<f32>> {
    let mask: Tensor<B, 3> = interpolate(
        mask.unsqueeze(),
        [height as usize, width as usize],
        InterpolateOptions::new(InterpolateMode::Bilinear),
    )
    .squeeze(0);

    let mask = mask.into_data().convert::<f32>();
    let mask = mask.to_vec::<f32>().unwrap();

    ImageBuffer::<Luma<f32>, Vec<f32>>::from_raw(width, height, mask).unwrap()
}

pub fn postprocess_mask_with_config<B: Backend>(
    mask: Tensor<B, 3>,
    width: u32,
    height: u32,
    _config: &PostprocessConfig,
) -> ImageBuffer<Luma<f32>, Vec<f32>> {
    let mask: Tensor<B, 3> = interpolate(
        mask.unsqueeze(),
        [height as usize, width as usize],
        InterpolateOptions::new(InterpolateMode::Bilinear),
    )
    .squeeze(0);

    let mask = mask.into_data().convert::<f32>();
    let mask = mask.to_vec::<f32>().unwrap();

    ImageBuffer::<Luma<f32>, Vec<f32>>::from_raw(width, height, mask).unwrap()
}

pub fn apply_threshold_mask(
    mask: &ImageBuffer<Luma<f32>, Vec<f32>>,
    threshold: Option<f32>,
) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    ImageBuffer::<Luma<u8>, Vec<u8>>::from_fn(mask.width(), mask.height(), |x, y| {
        let pixel = mask.get_pixel(x, y);
        let value = match threshold {
            Some(t) => {
                if pixel.0[0] > t {
                    255
                } else {
                    0
                }
            }
            None => (pixel.0[0].clamp(0.0, 1.0) * 255.0) as u8,
        };
        Luma([value])
    })
}

const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

pub struct Normalizer<B: Backend> {
    pub mean: Tensor<B, 4>,
    pub std: Tensor<B, 4>,
}

impl<B: Backend> Normalizer<B> {
    pub fn new(device: &Device<B>) -> Self {
        let mean = Tensor::<B, 1>::from_floats(MEAN, device).reshape([1, 3, 1, 1]);
        let std = Tensor::<B, 1>::from_floats(STD, device).reshape([1, 3, 1, 1]);
        Self { mean, std }
    }

    pub fn normalize(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        (input - self.mean.clone()) / self.std.clone()
    }

    pub fn denormalize(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        input * self.std.clone() + self.mean.clone()
    }
}

pub trait ImageProcessingExt {
    fn resize_maintain_aspect(&self, target_size: u32) -> RgbImage;
    fn to_square(&self, target_size: u32, fill_color: [u8; 3]) -> RgbImage;
}

impl ImageProcessingExt for RgbImage {
    fn resize_maintain_aspect(&self, target_size: u32) -> RgbImage {
        let (width, height) = self.dimensions();
        let scale = (target_size as f32 / width.max(height) as f32).min(1.0);
        let new_width = (width as f32 * scale) as u32;
        let new_height = (height as f32 * scale) as u32;

        image::imageops::resize(
            self,
            new_width,
            new_height,
            image::imageops::FilterType::Lanczos3,
        )
    }

    fn to_square(&self, target_size: u32, fill_color: [u8; 3]) -> RgbImage {
        let resized = self.resize_maintain_aspect(target_size);
        let (width, height) = resized.dimensions();

        let mut result = Self::from_pixel(target_size, target_size, image::Rgb(fill_color));
        let x_offset = (target_size - width) / 2;
        let y_offset = (target_size - height) / 2;

        image::imageops::overlay(&mut result, &resized, x_offset as i64, y_offset as i64);
        result
    }
}

pub trait TensorToImage<B: Backend> {
    fn to_image_rgb(&self) -> Result<RgbImage>;
    fn to_image_gray(&self) -> Result<ImageBuffer<Luma<u8>, Vec<u8>>>;
}

impl<B: Backend> TensorToImage<B> for Tensor<B, 3> {
    fn to_image_rgb(&self) -> Result<RgbImage> {
        let [c, h, w] = self.dims();
        if c != 3 {
            return Err(anyhow::anyhow!(
                "Expected 3 channels for RGB image, got {}",
                c
            ));
        }

        let data = self.clone().permute([1, 2, 0]).into_data().convert::<f32>();
        let pixels: Vec<u8> = data
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&x| (x.clamp(0.0, 1.0) * 255.0) as u8)
            .collect();

        RgbImage::from_raw(w as u32, h as u32, pixels)
            .ok_or_else(|| anyhow::anyhow!("Failed to create RGB image from tensor"))
    }

    fn to_image_gray(&self) -> Result<ImageBuffer<Luma<u8>, Vec<u8>>> {
        let [c, h, w] = self.dims();
        if c != 1 {
            return Err(anyhow::anyhow!(
                "Expected 1 channel for grayscale image, got {}",
                c
            ));
        }

        let data = self.clone().squeeze::<2>(0).into_data().convert::<f32>();
        let pixels: Vec<u8> = data
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&x| (x.clamp(0.0, 1.0) * 255.0) as u8)
            .collect();

        ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(w as u32, h as u32, pixels)
            .ok_or_else(|| anyhow::anyhow!("Failed to create grayscale image from tensor"))
    }
}

pub fn create_batch_from_images<B: Backend>(
    images: &[RgbImage],
    config: &PreprocessConfig,
    device: &Device<B>,
) -> Result<Tensor<B, 4>> {
    let mut batch_tensors = Vec::new();

    for image in images {
        let tensor = preprocess_with_config(image, config, device);
        batch_tensors.push(tensor);
    }

    let batch = Tensor::stack(batch_tensors, 0);
    Ok(batch)
}

pub fn process_batch_masks<B: Backend>(
    masks: Tensor<B, 4>,
    original_sizes: &[(u32, u32)],
    config: &PostprocessConfig,
) -> Result<Vec<ImageBuffer<Luma<f32>, Vec<f32>>>> {
    let batch_size = masks.dims()[0];
    let mut results = Vec::new();

    for i in 0..batch_size {
        let mask = masks.clone().slice([i..i + 1]).squeeze::<3>(0);
        let (width, height) = original_sizes[i];
        let processed = postprocess_mask_with_config(mask, width, height, config);
        results.push(processed);
    }

    Ok(results)
}
