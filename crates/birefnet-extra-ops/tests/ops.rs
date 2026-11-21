//! Integration tests for `birefnet-extra-ops`.

use approx::assert_abs_diff_eq;
use birefnet_extra_ops::{DropPath, DropPathConfig, Erfinv, erfinv, trunc_normal};
use burn::prelude::*;

#[test]
fn erfinv_matches_signed_values_and_round_trips_erf() {
    let device = Device::flex();
    let inputs = [-0.95_f32, -0.8, -0.5, -0.1, 0.0, 0.1, 0.5, 0.8, 0.95];
    let x = Tensor::<1>::from_data(inputs, &device);
    let inv = x.clone().erfinv();
    let free_inv = erfinv(x);

    let inv_vals = inv.clone().into_data().try_to_vec::<f32>().unwrap();
    let free_vals = free_inv.into_data().try_to_vec::<f32>().unwrap();
    assert_eq!(inv_vals, free_vals);

    let round_trip = inv.erf().into_data().try_to_vec::<f32>().unwrap();
    for (&expected, &actual) in inputs.iter().zip(&round_trip) {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-4);
    }

    // Odd symmetry: erfinv(-y) == -erfinv(y)
    for i in 0..4 {
        assert_abs_diff_eq!(inv_vals[i], -inv_vals[8 - i], epsilon = 1e-5);
    }
    assert_abs_diff_eq!(inv_vals[4], 0.0, epsilon = 1e-6);
}

#[test]
fn erfinv_handles_boundary_and_out_of_domain_values() {
    let device = Device::flex();
    let x = Tensor::<1>::from_data([-1.5_f32, -1.0, 1.0, 1.5, f32::NAN], &device);
    let out = x.erfinv().into_data().try_to_vec::<f32>().unwrap();

    assert!(out[0].is_nan(), "expected NaN for y = -1.5, got {}", out[0]);
    assert_eq!(out[1], f32::NEG_INFINITY);
    assert_eq!(out[2], f32::INFINITY);
    assert!(out[3].is_nan(), "expected NaN for y = 1.5, got {}", out[3]);
    assert!(out[4].is_nan(), "expected NaN for y = NaN, got {}", out[4]);
}

#[test]
fn trunc_normal_respects_bounds_and_symmetry() {
    let device = Device::flex();
    device.seed(42);
    let samples = trunc_normal::<2>([200, 200], 0.5, 0.25, 0.0, 1.0, &device);
    let vals = samples.into_data().try_to_vec::<f32>().unwrap();

    assert_eq!(vals.len(), 40_000);
    for &v in &vals {
        assert!((0.0..=1.0).contains(&v), "sample out of bounds: {v}");
    }
    let count = f64::from(u32::try_from(vals.len()).unwrap());
    let mean = vals.iter().map(|&v| f64::from(v)).sum::<f64>() / count;
    assert_abs_diff_eq!(mean, 0.5, epsilon = 0.02);
}

#[test]
fn trunc_normal_handles_degenerate_bounds() {
    let device = Device::flex();
    let vals = trunc_normal::<1>([8], 0.0, 1.0, 0.3, 0.3, &device)
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();
    for v in vals {
        assert_abs_diff_eq!(v, 0.3, epsilon = 1e-6);
    }
}

#[test]
#[should_panic(expected = "std must be finite and positive")]
fn trunc_normal_rejects_non_positive_std() {
    let device = Device::flex();
    let _ = trunc_normal::<1>([4], 0.0, 0.0, -1.0, 1.0, &device);
}

#[test]
fn droppath_scales_surviving_samples_and_propagates_gradients() {
    let ad_device = Device::flex().autodiff();
    ad_device.seed(123);

    let layer = DropPath::new(0.5, true);
    let x = Tensor::<4>::ones([200, 2, 2, 2], &ad_device).require_grad();
    let out = layer.forward(x.clone());
    assert_eq!(out.dims(), [200, 2, 2, 2]);

    let out_vals = out.clone().into_data().try_to_vec::<f32>().unwrap();
    let mut dropped = 0_usize;
    let mut kept = 0_usize;
    for chunk in out_vals.as_chunks::<8>().0 {
        if chunk.iter().all(|&v| v == 0.0) {
            dropped += 1;
        } else if chunk.iter().all(|&v| (v - 2.0).abs() < 1e-5) {
            kept += 1;
        } else {
            panic!("sample neither all-zero nor scaled by 1/(1 - 0.5) = 2.0: {chunk:?}");
        }
    }
    assert_eq!(dropped + kept, 200);
    assert!(
        (60..=140).contains(&dropped),
        "expected roughly 50% dropped out of 200, got {dropped}"
    );

    // Gradient flows only through kept samples with scale factor 2.0.
    let grads = out.sum().backward();
    let x_grad = x.grad(&grads).unwrap();
    let grad_vals = x_grad.into_data().try_to_vec::<f32>().unwrap();
    assert_eq!(grad_vals.len(), 200 * 8);
    assert!(grad_vals.iter().any(|&g| (g - 2.0).abs() < 1e-5));
    assert!(grad_vals.contains(&0.0));
}

#[test]
fn droppath_full_drop_zeros_on_autodiff_and_is_identity_on_valid_or_plain_device() {
    let ad_device = Device::flex().autodiff();
    let plain_device = Device::flex();
    let layer = DropPathConfig::new().with_drop_prob(1.0).init();
    assert_eq!(DropPathConfig::default(), DropPathConfig::new());

    let x_ad = Tensor::<2>::ones([4, 3], &ad_device);
    let dropped = layer
        .forward(x_ad.clone())
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();
    assert!(dropped.iter().all(|&v| v == 0.0));

    let valid_layer = DropPath::new(0.5, true).valid();
    let valid_out = valid_layer
        .forward(x_ad)
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();
    assert!(valid_out.iter().all(|&v| (v - 1.0).abs() < 1e-6));

    let x_plain = Tensor::<2>::ones([4, 3], &plain_device);
    let plain_out = layer
        .forward(x_plain)
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();
    assert!(plain_out.iter().all(|&v| (v - 1.0).abs() < 1e-6));
}

#[test]
#[should_panic(expected = "drop_prob must be in [0.0, 1.0]")]
fn droppath_rejects_invalid_probability() {
    let _ = DropPath::new(1.2, true);
}
