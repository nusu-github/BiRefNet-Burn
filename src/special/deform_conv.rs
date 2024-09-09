use burn::prelude::*;

pub fn deform_conv2d<B: Backend>(
    input: Tensor<B, 4, Float>,
    offsets: Tensor<B, 4, Float>,
    weights: Tensor<B, 4, Float>,
    bias: Option<Tensor<B, 1, Float>>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    mask: Option<Tensor<B, 4, Float>>,
) -> Tensor<B, 4, Float> {
    todo!()
}
