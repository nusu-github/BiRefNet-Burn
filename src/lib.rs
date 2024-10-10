mod config;
mod models;
mod special;

pub use config::{Backbone, ModelConfig};
pub use models::{BiRefNet, BiRefNetConfig};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{swin_v1_l, SwinTransformer};
    use burn::{
        backend::{wgpu::WgpuDevice, Wgpu},
        prelude::*,
    };

    type B = Wgpu<f32, i32>;

    #[test]
    fn test_bi_ref_net() -> anyhow::Result<()> {
        let device = &WgpuDevice::IntegratedGpu(1);

        let bi_ref_net = BiRefNetConfig::new(
            ModelConfig::load("config.json").unwrap_or_else(|_| ModelConfig::new()),
            true,
        )
        .init::<B>(&device);

        let x = Tensor::<B, 4>::zeros([1, 3, 1024, 1024], device);
        let _y = bi_ref_net.forward(x.clone());

        Ok(())
    }

    #[test]
    fn test_swin_v1_l() -> anyhow::Result<()> {
        let device = &Default::default();

        let swin_v1_l: SwinTransformer<B> = swin_v1_l(device);

        let x = Tensor::<B, 4, Float>::zeros([1, 3, 1024, 1024], device);

        let y = swin_v1_l.forward(x);

        println!("{:?}", y);

        Ok(())
    }
}
