use birefnet_burn::{BiRefNetConfig, ModelConfig};
use burn::{backend::NdArray, prelude::*};

type B = NdArray;

fn main() -> anyhow::Result<()> {
    let device = &Default::default();

    let bi_ref_net = BiRefNetConfig::new(ModelConfig::new(), true).init::<B>(&device);

    let start = std::time::Instant::now();
    let mut result = Vec::new();
    for _ in 0..20 {
        let start_ = std::time::Instant::now();
        let x = Tensor::<B, 4>::zeros([1, 3, 1024, 1024], device);
        let _y = bi_ref_net.forward(x);
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
