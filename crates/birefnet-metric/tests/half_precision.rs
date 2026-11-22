//! `F16` model outputs are read back without dtype errors.
//!
//! Device float settings lock on first use, so this precision gets its own test binary.

use approx::assert_abs_diff_eq;
use birefnet_metric::Sample;
use burn::tensor::{Device, FloatDType, Tensor};

#[test]
fn half_precision_batches_are_evaluated() {
    let mut device = Device::flex();
    device
        .configure(FloatDType::F16)
        .expect("Flex supports F16");

    let mut mask = [[0.0_f32; 8]; 8];
    for row in &mut mask[2..6] {
        row[2..6].fill(1.0);
    }
    let targets = Tensor::<4>::from_floats([[mask]], &device);
    let predictions = targets.clone().mul_scalar(0.75);

    let samples = Sample::batch_from_tensors(&predictions, &targets).unwrap();
    assert_eq!(samples.len(), 1);
    // Min-max normalization maps the 0.75 plateau to 1.0, a perfect prediction.
    assert_abs_diff_eq!(samples[0].mae(), 0.0);
    assert_abs_diff_eq!(samples[0].f_measure(0.3).adaptive, 1.0, epsilon = 1e-12);
}
