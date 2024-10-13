use burn::tensor::module::interpolate;
use burn::tensor::ops::{InterpolateMode, InterpolateOptions};
use burn::{prelude::*, record::Recorder, tensor::Element};
use image::{
    buffer::ConvertBuffer,
    imageops::{self, FilterType},
    GenericImageView, ImageBuffer, Luma, RgbImage,
};

fn to_tensor<B: Backend, T: Element>(
    data: Vec<T>,
    shape: [usize; 3],
    device: &Device<B>,
) -> Tensor<B, 3> {
    Tensor::<B, 3>::from_data(
        TensorData::new(data, shape).convert::<B::FloatElem>(),
        device,
    )
    .permute([2, 0, 1]) // [C, H, W]
}

pub fn preprocess<B: Backend>(
    image: RgbImage,
    image_size: u32,
    device: &Device<B>,
) -> Tensor<B, 3> {
    let width = image.width();
    let height = image.height();
    let tensor = to_tensor(
        image.into_raw(),
        [height as usize, width as usize, 3],
        device,
    );

    let tensor = interpolate(
        tensor.unsqueeze(),
        [image_size as usize, image_size as usize],
        InterpolateOptions::new(InterpolateMode::Bilinear),
    );

    tensor.squeeze(0) / 255.0
}

fn to_image<B: Backend>(
    tensor: Tensor<B, 3>,
    width: u32,
    height: u32,
) -> ImageBuffer<Luma<f32>, Vec<f32>> {
    let tensor = tensor.permute([1, 2, 0]); // [H, W, C]
    let tensor = tensor.into_data().convert::<f32>();
    let tensor = tensor.to_vec::<f32>().unwrap();
    ImageBuffer::<Luma<f32>, Vec<f32>>::from_raw(width, height, tensor).unwrap()
}

pub fn postprocess_mask<B: Backend>(
    mask: Tensor<B, 3>,
    image_size: u32,
    width: u32,
    height: u32,
) -> ImageBuffer<Luma<f32>, Vec<f32>> {
    let mask = to_image(mask, image_size, image_size);
    imageops::resize(&mask, width, height, FilterType::Lanczos3)
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
}
