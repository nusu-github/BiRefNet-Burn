//! Half-precision (`F16`) regression tests.
//!
//! Device float settings lock on first use, so this precision gets its own test binary.

use approx::assert_abs_diff_eq;
use birefnet_util::{
    distance::{euclidean_distance_transform, manhattan_distance_transform},
    filters::{custom_filter, gaussian_filter_matlab, gaussian_kernel},
    foreground::refine_foreground_core,
    morphology::{StructuringElement, dilation, erosion},
    normalize::apply_imagenet_normalization,
};
use burn::tensor::{DType, Device, FloatDType, Tensor};

fn f16_device() -> Device {
    let mut device = Device::flex();
    device
        .configure(FloatDType::F16)
        .expect("Flex supports F16");
    device
}

fn values(tensor: Tensor<4>) -> Vec<f32> {
    tensor.try_into_vec_as::<f32>().unwrap()
}

#[test]
fn utilities_accept_and_preserve_f16_tensors() {
    let device = f16_device();
    let mut grid = [[1.0_f32; 5]; 5];
    grid[2][2] = 0.0;
    let mask = Tensor::<4>::from_floats([[grid]], &device);
    assert_eq!(mask.dtype(), DType::F16);

    let edt = euclidean_distance_transform(&mask).unwrap();
    assert_eq!(edt.dtype(), DType::F16);
    assert_abs_diff_eq!(values(edt)[0], 8.0_f32.sqrt(), epsilon = 1e-2);

    let l1 = manhattan_distance_transform(&mask).unwrap();
    assert_abs_diff_eq!(values(l1)[0], 4.0);

    let inverted = mask.neg().add_scalar(1.0);
    let disk = StructuringElement::disk(1);
    let grown = dilation(inverted.clone(), &disk);
    assert_eq!(grown.dtype(), DType::F16);
    assert_abs_diff_eq!(values(grown).iter().sum::<f32>(), 5.0);
    assert_abs_diff_eq!(values(erosion(inverted, &disk)).iter().sum::<f32>(), 0.0);

    let image = Tensor::<4>::full([1, 3, 8, 8], 0.5, &device);
    let normalized = apply_imagenet_normalization(image.clone()).unwrap();
    assert_eq!(normalized.dtype(), DType::F16);

    let blurred = gaussian_filter_matlab(image.clone(), (3, 3), 1.0);
    assert_eq!(blurred.dtype(), DType::F16);
    let filtered = custom_filter(image.clone(), gaussian_kernel(3, 1.0, &device));
    for v in values(blurred).into_iter().chain(values(filtered)) {
        assert_abs_diff_eq!(v, 0.5, epsilon = 1e-3);
    }

    let refined = refine_foreground_core(image, Tensor::ones([1, 1, 8, 8], &device), 2);
    assert_eq!(refined.dtype(), DType::F16);
}
