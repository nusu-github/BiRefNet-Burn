//! `F16` masks flow through postprocessing and export without dtype errors.
//!
//! Device float settings lock on first use, so this precision gets its own test binary.

use birefnet_inference::postprocess::{PostprocessConfig, mask_to_u8, postprocess_mask};
use burn::tensor::{DType, Device, FloatDType, Tensor};

#[test]
fn postprocessing_preserves_f16_masks() {
    let mut device = Device::flex();
    device
        .configure(FloatDType::F16)
        .expect("Flex supports F16");

    let mut grid = [[0.0_f32; 7]; 7];
    for (y, row) in grid.iter_mut().enumerate().skip(1).take(5) {
        for (x, value) in row.iter_mut().enumerate().skip(1).take(5) {
            *value = if (y, x) == (3, 3) { 0.0 } else { 0.9 };
        }
    }
    grid[0][6] = 0.9; // speckle
    let mask = Tensor::<4>::from_floats([[grid]], &device);

    let config = PostprocessConfig::new()
        .with_threshold(0.5)
        .unwrap()
        .with_min_component_size(2)
        .with_fill_holes(true);
    let cleaned = postprocess_mask(mask, &config).unwrap();
    assert_eq!(cleaned.dtype(), DType::F16);

    let bytes = mask_to_u8(&cleaned).unwrap().into_raw();
    assert_eq!(bytes[6], 0, "speckle removed");
    assert_eq!(bytes[3 * 7 + 3], 255, "hole filled");
}
