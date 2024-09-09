mod models;
mod special;

pub use models::{Backbone, BiRefNet, BiRefNetConfig, SqueezeBlockConfig, SqueezeBlockEnum};

#[cfg(test)]
mod tests {
    use burn::{backend::Wgpu, prelude::*};

    use super::*;

    type B = Wgpu;

    #[test]
    fn it_works() -> anyhow::Result<()> {
        let device = &Default::default();

        let bi_ref_net: BiRefNet<B> = BiRefNetConfig::new(
            Backbone::SwinV1L,
            true,
            SqueezeBlockConfig::new(SqueezeBlockEnum::BasicDecBlk, 1),
        )
        .init(device);

        let x = Tensor::<B, 4, Float>::zeros([1, 3, 1024, 1024], device);

        let y = bi_ref_net.forward(x);

        Ok(())
    }
}
