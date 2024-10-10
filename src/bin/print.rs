use birefnet_burn::{BiRefNetConfig, ModelConfig};
use burn::backend::{wgpu::WgpuDevice, Wgpu};

type B = Wgpu<f32, i32>;
fn main() {
    let device = WgpuDevice::default();
    let model = BiRefNetConfig::new(ModelConfig::new(), false).init::<B>(&device);

    println!("{}", model);
}
