//! Integration tests for `birefnet-loss`.

use approx::assert_abs_diff_eq;
use birefnet_loss::{
    BiRefNetLossConfig, ClassificationLoss, ContourLoss, IoULoss, LossWeightsConfig, MaeLoss,
    PatchIoULossConfig, PixLossConfig, SSIMLossConfig, StructureLossConfig,
    ThresholdRegularizationLoss,
};
use burn::{nn::loss::Reduction, prelude::*};

#[test]
fn ssim_loss_is_zero_for_identical_inputs_and_sensitive_to_2d_structure() {
    let device = Device::flex();
    let ssim = SSIMLossConfig::new().with_window_size(5).init();

    // Checkerboard differs spatially in both H and W while keeping equal pixel sums locally in 1x1!
    let mut vals_a = vec![0.0_f32; 16 * 16];
    let mut vals_b = vec![0.0_f32; 16 * 16];
    for row in 0..16 {
        for col in 0..16 {
            let idx = row * 16 + col;
            vals_a[idx] = if (row + col) % 2 == 0 { 1.0 } else { 0.0 };
            vals_b[idx] = if (row + col) % 2 == 0 { 0.0 } else { 1.0 };
        }
    }
    let a = Tensor::<1>::from_floats(vals_a.as_slice(), &device).reshape([1, 1, 16, 16]);
    let b = Tensor::<1>::from_floats(vals_b.as_slice(), &device).reshape([1, 1, 16, 16]);

    let identical = ssim
        .forward(a.clone(), a.clone(), Reduction::Mean)
        .into_scalar::<f64>();
    assert_abs_diff_eq!(identical, 0.0, epsilon = 1e-5);

    let inverted = ssim.forward(a, b, Reduction::Mean).into_scalar::<f64>();
    assert!(
        inverted > 0.25,
        "expected significant structural loss for inverted checkerboard, got {inverted}"
    );
}

#[test]
fn structure_loss_matches_analytical_weighted_bce_and_iou() {
    let device = Device::flex();
    let structure = StructureLossConfig::new()
        .with_kernel_size(3)
        .with_edge_factor(5.0)
        .with_eps(1e-8)
        .init();

    // Constant targets -> avg_pool(targets) == targets -> uniform weight = 1.0 everywhere.
    let logits = Tensor::<4>::zeros([2, 1, 8, 8], &device);
    let targets = Tensor::<4>::ones([2, 1, 8, 8], &device);

    // At logits = 0, sigmoid = 0.5:
    // weighted BCE = ln(2) ~= 0.693147
    // weighted IoU: inter = 0.5 * 64 = 32, union = (0.5 + 1.0) * 64 = 96, iou = 32 / (96 - 32) = 0.5, loss = 0.5
    let actual = structure
        .forward(logits, targets, Reduction::Mean)
        .into_scalar::<f64>();
    assert_abs_diff_eq!(actual, 2.0_f64.ln() + 0.5, epsilon = 1e-5);
}

#[test]
fn individual_losses_match_analytical_and_batch_shapes() {
    let device = Device::flex();

    // IoU: p = 0.5 everywhere, y = 1.0 everywhere -> iou = 0.5 -> loss = 0.5.
    let iou = IoULoss::new();
    let p_half = Tensor::<4>::full([2, 1, 4, 4], 0.5, &device);
    let y_one = Tensor::<4>::ones([2, 1, 4, 4], &device);
    assert_abs_diff_eq!(
        iou.forward(p_half.clone(), y_one.clone(), Reduction::Mean)
            .into_scalar::<f64>(),
        0.5,
        epsilon = 1e-5
    );

    // PatchIoU with 4x4 patches over 4x8 across batch = 2: 2 patches * 2 samples * 0.5 = 2.0.
    let patch_iou = PatchIoULossConfig::new()
        .with_patch_height(4)
        .with_patch_width(4)
        .init();
    let p_patch = Tensor::<4>::full([2, 1, 4, 8], 0.5, &device);
    let y_patch = Tensor::<4>::ones([2, 1, 4, 8], &device);
    assert_abs_diff_eq!(
        patch_iou
            .forward(&p_patch, &y_patch, Reduction::Mean)
            .into_scalar::<f64>(),
        2.0,
        epsilon = 1e-4
    );

    // ContourLoss supports batch > 1 cleanly.
    let contour = ContourLoss::new();
    let contour_per_item = contour.forward_no_reduction(p_half, y_one.clone());
    assert_eq!(contour_per_item.dims(), [2]);
    let contour_mean = contour
        .forward(y_one.clone(), y_one.clone(), Reduction::Mean)
        .into_scalar::<f64>();
    assert!(contour_mean < 1e-3);

    // MAE and ThresholdRegularization.
    let mae = MaeLoss::new();
    assert_abs_diff_eq!(
        mae.forward(
            Tensor::<4>::full([2, 1, 4, 4], 0.25, &device),
            y_one.clone(),
            Reduction::Mean
        )
        .into_scalar::<f64>(),
        0.75,
        epsilon = 1e-5
    );

    let thr = ThresholdRegularizationLoss::new();
    assert_abs_diff_eq!(
        thr.forward(Tensor::<2>::full([2, 2], 0.5, &device), Reduction::Mean)
            .into_scalar::<f64>(),
        0.5,
        epsilon = 1e-5
    );
    assert_abs_diff_eq!(
        thr.forward(y_one, Reduction::Mean).into_scalar::<f64>(),
        0.0,
        epsilon = 1e-6
    );
}

#[test]
fn classification_loss_handles_batch_and_multiple_levels() {
    let device = Device::flex();
    let cls = ClassificationLoss::new(&device);
    let preds = vec![
        Tensor::<2>::from_floats([[4.0, 0.0], [0.0, 4.0]], &device),
        Tensor::<2>::from_floats([[4.0, 0.0], [0.0, 4.0]], &device),
    ];
    let targets = Tensor::<1, Int>::from_ints([0, 1], &device);
    let val = cls
        .forward(&preds, &targets, Reduction::Mean)
        .into_scalar::<f64>();
    assert!(val > 0.0 && val < 0.1);
}

#[test]
fn pixel_and_birefnet_loss_support_negative_logits_soft_matting_batch_and_autodiff() {
    let device = Device::flex().autodiff();
    let weights = LossWeightsConfig::new()
        .with_bce(1.0)
        .with_iou(1.0)
        .with_iou_patch(0.5)
        .with_mae(1.0)
        .with_mse(1.0)
        .with_reg(1.0)
        .with_ssim(1.0)
        .with_cnt(1.0)
        .with_structure(1.0);
    let pix = PixLossConfig::new(weights).init(&device);

    // Spatial dims 64x64 with batch = 2, negative logits, and continuous soft target = 0.3.
    let pred_logits = Tensor::<4>::full([2, 1, 64, 64], -2.0, &device).require_grad();
    let soft_target = Tensor::<4>::full([2, 1, 64, 64], 0.3, &device);

    let loss_fast = pix
        .forward(std::slice::from_ref(&pred_logits), &soft_target, 1.0)
        .unwrap();
    let (loss_detailed, breakdown) = pix
        .forward_detailed(std::slice::from_ref(&pred_logits), &soft_target, 1.0)
        .unwrap();

    let fast_scalar = loss_fast.clone().into_scalar::<f64>();
    let detailed_scalar = loss_detailed.into_scalar::<f64>();
    assert!(fast_scalar.is_finite() && fast_scalar > 0.0);
    assert_abs_diff_eq!(fast_scalar, detailed_scalar, epsilon = 1e-4);

    for key in [
        "bce",
        "iou",
        "iou_patch",
        "mae",
        "mse",
        "reg",
        "ssim",
        "cnt",
        "structure",
    ] {
        let entry = *breakdown.get(key).expect("missing loss key");
        assert!(
            entry.is_finite() && entry >= 0.0,
            "bad component {key}: {entry}"
        );
    }

    // Backprop through all 9 losses simultaneously.
    let grads = loss_fast.backward();
    let grad = pred_logits
        .grad(&grads)
        .expect("expected non-empty gradient");
    let total_abs_grad = grad.abs().sum().into_scalar::<f64>();
    assert!(
        total_abs_grad.is_finite() && total_abs_grad > 0.1,
        "expected significant summed gradient, got {total_abs_grad}"
    );

    // BiRefNetLoss handles presets and empty prediction error cleanly.
    let full_loss = BiRefNetLossConfig::matting_optimized().init(&device);
    let err = full_loss.forward(&[], &soft_target).unwrap_err();
    assert!(err.is_empty_predictions());

    let (total, map) = full_loss
        .forward_detailed(&[pred_logits], &soft_target)
        .unwrap();
    assert!(total.into_scalar::<f64>() > 0.0);
    assert!(map.contains_key("total"));
}
