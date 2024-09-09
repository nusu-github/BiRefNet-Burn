mod models;
mod special;

pub use models::{Backbone, BiRefNet, BiRefNetConfig, SqueezeBlockConfig, SqueezeBlockEnum};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{swin_v1_l, SwinTransformer};
    use burn::{backend::Wgpu, prelude::*};

    type B = Wgpu;

    #[test]
    fn test_bi_ref_net() -> anyhow::Result<()> {
        let device = &Default::default();

        let bi_ref_net: BiRefNet<B> = BiRefNetConfig::new(
            Backbone::SwinV1L,
            true,
            SqueezeBlockConfig::new(SqueezeBlockEnum::BasicDecBlk, 1),
        )
        .init(device);

        let x = Tensor::<B, 4, Float>::zeros([1, 3, 1024, 1024], device);

        let start = std::time::Instant::now();
        let mut result = Vec::new();
        for _ in 0..20 {
            let start_ = std::time::Instant::now();
            let y = bi_ref_net.forward(x.clone());
            result.push(start_.elapsed());
        }
        println!(
            "Total time: {:?}, Speed: {:?}",
            start.elapsed(),
            20.0 / start.elapsed().as_secs_f32()
        );
        println!("{:?}", result);

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
