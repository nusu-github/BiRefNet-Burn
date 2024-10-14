use burn::{
    prelude::*,
    record::Recorder,
    tensor::{
        module::interpolate,
        ops::{InterpolateMode, InterpolateOptions},
    },
};
use image::{
    buffer::ConvertBuffer, flat::SampleLayout, GenericImageView, ImageBuffer, Luma, Pixel,
    Primitive,
};
use num_traits::ToPrimitive;

pub trait IntoTensor3<B: Backend> {
    type Out;

    fn into_tensor3(self, device: &Device<B>) -> Self::Out;
}

impl<B: Backend, P> IntoTensor3<B> for ImageBuffer<P, Vec<P::Subpixel>>
where
    P: Pixel + 'static,
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
            TensorData::new(
                self.into_iter().map(|x| x.to_f32().unwrap()).collect(),
                shape,
            )
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
{
    let tensor = image.clone().into_tensor3(device)
        .permute([2, 0, 1]) // [C, H, W]
        / P::Subpixel::DEFAULT_MAX_VALUE.to_f32().unwrap();

    let tensor = interpolate(
        tensor.unsqueeze(),
        [image_size as usize, image_size as usize],
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
