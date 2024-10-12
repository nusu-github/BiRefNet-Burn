use crate::imageops_ai::{AlphaMaskImage, Padding};
use birefnet_burn::{BiRefNetConfig, ModelConfig};
use burn::{
    backend::cuda_jit::{Cuda, CudaDevice},
    nn::Sigmoid,
    prelude::*,
    record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
    tensor::Element,
};
use image::imageops::FilterType;
use image::{
    buffer::ConvertBuffer, imageops, GenericImageView, GrayImage, Pixel, Primitive, Rgb, RgbImage,
};

mod imageops_ai;

fn to_tensor<B: Backend, T: Element>(
    data: Vec<T>,
    shape: [usize; 3],
    device: &Device<B>,
) -> Tensor<B, 3> {
    Tensor::<B, 3>::from_data(TensorData::new(data, shape).convert::<B::FloatElem>(), device)
        .permute([2, 0, 1]) // [C, H, W]
        / 255 // normalize between [0, 1]
}

fn preprocess<B: Backend>(
    image: &RgbImage,
    image_size: u32,
    device: &Device<B>,
) -> (Tensor<B, 3>, [u32; 4]) {
    let image = imageops::resize(image, image_size, image_size, FilterType::Lanczos3);
    let (w, h) = image.dimensions();
    let (image, (x, y)) = image.padding_square(Rgb([0, 0, 0]));

    let tensor = to_tensor(
        image.into_raw(),
        [image_size as usize, image_size as usize, 3],
        device,
    );

    (tensor, [x, y, w, h])
}

fn to_image<B: Backend>(tensor: Tensor<B, 3>, width: usize, height: usize) -> GrayImage {
    let tensor = tensor.permute([1, 2, 0]); // [H, W, C]
    let tensor = tensor.to_data().convert::<f32>();
    let tensor = tensor.to_vec::<f32>().unwrap();
    let tensor = tensor.into_iter().map(|x| (x * 255.0) as u8).collect();
    GrayImage::from_raw(width as u32, height as u32, tensor).unwrap()
}

fn postprocess_mask<B: Backend>(
    mask: Tensor<B, 3>,
    image_size: u32,
    crop: [u32; 4],
    width: u32,
    height: u32,
) -> GrayImage {
    let [x, y, w, h] = crop;
    let mask = to_image(mask, image_size as usize, image_size as usize);
    let mask = mask.view(x, y, w, h).to_image();
    imageops::resize(&mask, width, height, FilterType::Lanczos3)
}

const HEIGHT: usize = 1024;
const WIDTH: usize = 1024;
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

fn main() -> anyhow::Result<()> {
    type Backend = Cuda;

    let device = CudaDevice::default();
    let record = NamedMpkFileRecorder::<FullPrecisionSettings>::default()
        .load("ckpt/BiRefNet-general-epoch_244".into(), &device)
        .expect("Should decode state successfully");

    let model = BiRefNetConfig::new(
        ModelConfig::load("config.json").unwrap_or_else(|_| ModelConfig::new()),
        true,
    )
    .init::<Backend>(&device)
    .load_record(record);

    let img = image::open("test.jpg")?.into_rgb8();
    let size = img.dimensions();

    let (img_tensor, crop) = preprocess(&img, WIDTH as u32, &device);
    let img_tensor = img_tensor.unsqueeze::<4>();

    let x = Normalizer::new(&device).normalize(img_tensor);

    let mask = model.forward(x);
    let mask = Sigmoid::new().forward(mask).squeeze::<3>(0);

    let mask = postprocess_mask(mask, WIDTH as u32, crop, size.0, size.1);
    let mask_img = img.add_alpha_mask(&mask)?;

    mask_img.save("mask.png")?;

    Ok(())
}
