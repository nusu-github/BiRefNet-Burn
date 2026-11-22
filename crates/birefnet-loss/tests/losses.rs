//! Integration tests for `birefnet-loss`.

use approx::assert_abs_diff_eq;
use birefnet_loss::{
    BiRefNetLossConfig, ClassificationLoss, ClassificationLossConfig, ContourLoss, IouLoss,
    LossWeightsConfig, MaeLoss, PatchIouLossConfig, PixLossConfig, SsimLossConfig,
    StructureLossConfig, ThresholdRegularizationLoss,
};
use burn::{
    nn::loss::Reduction,
    prelude::*,
    tensor::{Distribution, module::conv2d, ops::ConvOptions},
};

fn to_vec(tensor: Tensor<1>) -> Vec<f32> {
    tensor.into_data().try_to_vec::<f32>().unwrap()
}

/// Weights with only the named components enabled at weight `1.0`.
fn only(components: &[&str]) -> LossWeightsConfig {
    let w = |name: &str| if components.contains(&name) { 1.0 } else { 0.0 };
    LossWeightsConfig::new()
        .with_bce(w("bce"))
        .with_iou(w("iou"))
        .with_iou_patch(w("iou_patch"))
        .with_mae(w("mae"))
        .with_mse(w("mse"))
        .with_reg(w("reg"))
        .with_ssim(w("ssim"))
        .with_cnt(w("cnt"))
        .with_structure(w("structure"))
}

#[test]
fn ssim_loss_is_zero_for_identical_inputs_and_sensitive_to_2d_structure() {
    let device = Device::flex();
    let ssim = SsimLossConfig::new().with_window_size(5).init();

    // Complementary checkerboards differ at every pixel while having the same local mean.
    let mut vals_a = vec![0.0_f32; 16 * 16];
    let mut vals_b = vec![0.0_f32; 16 * 16];
    for row in 0..16 {
        for col in 0..16 {
            let idx = row * 16 + col;
            vals_a[idx] = if (row + col) % 2 == 0 { 1.0 } else { 0.0 };
            vals_b[idx] = 1.0 - vals_a[idx];
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
fn ssim_separable_window_matches_direct_2d_window() {
    let device = Device::flex();
    device.seed(7);
    let a = Tensor::<4>::random([2, 1, 12, 12], Distribution::Uniform(0.0, 1.0), &device);
    let b = Tensor::<4>::random([2, 1, 12, 12], Distribution::Uniform(0.0, 1.0), &device);
    let ssim = SsimLossConfig::new().with_window_size(5).init();
    let separable = to_vec(ssim.forward_no_reduction(a.clone(), b.clone()));

    // Reference: a single 5x5 outer-product Gaussian window with sigma = 1.5.
    let taps: Vec<f32> = [-2.0_f32, -1.0, 0.0, 1.0, 2.0]
        .iter()
        .map(|offset| (-offset * offset / 4.5).exp())
        .collect();
    let norm: f32 = taps.iter().sum();
    let window: Vec<f32> = taps
        .iter()
        .flat_map(|row| taps.iter().map(move |col| row * col / (norm * norm)))
        .collect();
    let window = Tensor::<1>::from_floats(window.as_slice(), &device).reshape([1, 1, 5, 5]);
    let blur = |t: Tensor<4>| {
        conv2d(
            t,
            window.clone(),
            None,
            ConvOptions::new([1, 1], [2, 2], [1, 1], 1),
        )
    };
    let (mu1, mu2) = (blur(a.clone()), blur(b.clone()));
    let s1 = blur(a.clone() * a.clone()) - mu1.clone() * mu1.clone();
    let s2 = blur(b.clone() * b.clone()) - mu2.clone() * mu2.clone();
    let s12 = blur(a * b) - mu1.clone() * mu2.clone();
    let num = (mu1.clone() * mu2.clone() * 2.0 + 1e-4) * (s12 * 2.0 + 9e-4);
    let den = (mu1.clone() * mu1 + mu2.clone() * mu2 + 1e-4) * (s1 + s2 + 9e-4);
    let reference = to_vec(
        (num / den)
            .flatten::<2>(1, 3)
            .mean_dim(1)
            .squeeze_dim::<1>(1)
            .neg()
            .add_scalar(1.0)
            .div_scalar(2.0),
    );

    for (sep, reference) in separable.iter().zip(&reference) {
        assert_abs_diff_eq!(sep, reference, epsilon = 1e-4);
    }
}

#[test]
fn structure_loss_matches_analytical_weighted_bce_and_iou() {
    let device = Device::flex();
    let structure = StructureLossConfig::new()
        .with_kernel_size(3)
        .with_edge_factor(5.0)
        .with_eps(1e-8)
        .init();

    // Constant targets -> avg_pool(targets) == targets in the interior; the padded border
    // still sees zeros, so use an all-zero target where pooling is exact everywhere.
    let logits = Tensor::<4>::zeros([2, 1, 8, 8], &device);
    let targets = Tensor::<4>::zeros([2, 1, 8, 8], &device);

    // At logits = 0, sigmoid = 0.5 and y = 0:
    // weighted BCE = ln(2); IoU: inter = 0, union = 32 -> iou = eps / (32 + eps) ~= 0 -> loss 1.
    let actual = structure
        .forward(logits, targets, Reduction::Mean)
        .into_scalar::<f64>();
    assert_abs_diff_eq!(actual, 2.0_f64.ln() + 1.0, epsilon = 1e-5);
}

#[test]
fn individual_losses_match_analytical_and_batch_shapes() {
    let device = Device::flex();

    // IoU: p = 0.5 everywhere, y = 1.0 everywhere -> iou = 0.5 -> loss = 0.5.
    let iou = IouLoss::new();
    let p_half = Tensor::<4>::full([2, 1, 4, 4], 0.5, &device);
    let y_one = Tensor::<4>::ones([2, 1, 4, 4], &device);
    assert_abs_diff_eq!(
        iou.forward(p_half.clone(), y_one.clone(), Reduction::Mean)
            .into_scalar::<f64>(),
        0.5,
        epsilon = 1e-5
    );

    // ContourLoss produces one value per sample and vanishes for a perfect, flat prediction.
    let contour = ContourLoss::new();
    assert_eq!(
        contour.forward_no_reduction(p_half, y_one.clone()).dims(),
        [2]
    );
    let contour_mean = contour
        .forward(y_one.clone(), y_one.clone(), Reduction::Mean)
        .into_scalar::<f64>();
    assert!(contour_mean < 1e-3);

    let mae = MaeLoss::new();
    assert_abs_diff_eq!(
        mae.forward(
            Tensor::<4>::full([2, 1, 4, 4], 0.25, &device),
            y_one,
            Reduction::Mean
        )
        .into_scalar::<f64>(),
        0.75,
        epsilon = 1e-5
    );

    // 2 p (1 - p): 0.5 at p = 0.5, 0.32 at p = 0.2, 0 at p = 1.
    let thr = ThresholdRegularizationLoss::new();
    for (p, expected) in [(0.5, 0.5), (0.2, 0.32), (1.0, 0.0)] {
        assert_abs_diff_eq!(
            thr.forward(Tensor::<2>::full([2, 2], p, &device), Reduction::Mean)
                .into_scalar::<f64>(),
            expected,
            epsilon = 1e-5
        );
    }
}

#[test]
fn iou_loss_treats_two_empty_masks_as_a_perfect_match() {
    let device = Device::flex();
    let empty = Tensor::<4>::zeros([2, 1, 8, 8], &device);
    let loss = IouLoss::new()
        .forward(empty.clone(), empty, Reduction::Sum)
        .into_scalar::<f64>();
    assert_abs_diff_eq!(loss, 0.0, epsilon = 1e-6);
}

#[test]
fn patch_iou_reductions_and_clipped_edges() {
    let device = Device::flex();
    let patch_iou = PatchIouLossConfig::new()
        .with_patch_height(4)
        .with_patch_width(4)
        .init();

    // Sample 0: p = 0.5 on a ones target -> every patch loss is 0.5, two 4x4 patches.
    // Sample 1: perfect prediction -> 0.
    let preds = Tensor::cat(
        vec![
            Tensor::<4>::full([1, 1, 4, 8], 0.5, &device),
            Tensor::<4>::ones([1, 1, 4, 8], &device),
        ],
        0,
    );
    let targets = Tensor::<4>::ones([2, 1, 4, 8], &device);

    let per_sample = to_vec(patch_iou.forward_no_reduction(preds.clone(), targets.clone()));
    assert_eq!(per_sample.len(), 2);
    assert_abs_diff_eq!(per_sample[0], 1.0, epsilon = 1e-5);
    assert_abs_diff_eq!(per_sample[1], 0.0, epsilon = 1e-5);

    let mean = patch_iou.forward(preds.clone(), targets.clone(), Reduction::Mean);
    let sum = patch_iou.forward(preds, targets, Reduction::Sum);
    assert_abs_diff_eq!(mean.into_scalar::<f64>(), 0.5, epsilon = 1e-5);
    assert_abs_diff_eq!(sum.into_scalar::<f64>(), 1.0, epsilon = 1e-5);

    // 6x6 with 4x4 patches: one full patch, two edge strips, and one corner -> 4 patches.
    let p = Tensor::<4>::full([1, 1, 6, 6], 0.5, &device);
    let y = Tensor::<4>::ones([1, 1, 6, 6], &device);
    let clipped = patch_iou.forward(p, y, Reduction::Sum).into_scalar::<f64>();
    assert_abs_diff_eq!(clipped, 4.0 * 0.5, epsilon = 1e-5);
}

#[test]
fn classification_loss_is_per_sample_cross_entropy() {
    let device = Device::flex();
    let cls = ClassificationLossConfig::new().with_weight(2.0).init();
    let preds = [Tensor::<2>::from_floats([[4.0, 0.0], [0.0, 0.0]], &device)];
    let targets = Tensor::<1, Int>::from_ints([0, 1], &device);

    // ln(1 + e^-4) for the confident sample and ln(2) for the uniform one, both doubled.
    let per_sample = to_vec(cls.forward_no_reduction(&preds, &targets));
    let confident = 2.0 * (1.0 + (-4.0_f32).exp()).ln();
    let uniform = 2.0 * 2.0_f32.ln();
    assert_abs_diff_eq!(per_sample[0], confident, epsilon = 1e-5);
    assert_abs_diff_eq!(per_sample[1], uniform, epsilon = 1e-5);

    let mean = cls
        .forward(&preds, &targets, Reduction::Mean)
        .into_scalar::<f32>();
    let sum = cls
        .forward(&preds, &targets, Reduction::Sum)
        .into_scalar::<f32>();
    assert_abs_diff_eq!(mean, f32::midpoint(confident, uniform), epsilon = 1e-5);
    assert_abs_diff_eq!(sum, confident + uniform, epsilon = 1e-5);

    // Levels are summed.
    let two_levels = [preds[0].clone(), preds[0].clone()];
    let doubled = ClassificationLoss::new()
        .forward(&two_levels, &targets, Reduction::Sum)
        .into_scalar::<f32>();
    assert_abs_diff_eq!(doubled, confident + uniform, epsilon = 1e-5);
}

/// Central finite difference of `f` at `x` with step `h`.
fn finite_difference(f: impl Fn(f32) -> f64, x: f32, h: f32) -> f64 {
    (f(x + h) - f(x - h)) / (2.0 * f64::from(h))
}

#[test]
fn bce_gradient_is_smooth_at_zero_logits() {
    let pix = PixLossConfig::new(only(&["bce"])).init();
    let device = Device::flex().autodiff();
    let plain = Device::flex();

    for target in [0.0_f32, 0.3, 1.0] {
        let logits = Tensor::<4>::zeros([1, 1, 1, 1], &device).require_grad();
        let gt = Tensor::<4>::full([1, 1, 1, 1], target, &device);
        let grads = pix
            .forward(std::slice::from_ref(&logits), &gt, 1.0)
            .unwrap()
            .backward();
        let analytic = logits.grad(&grads).unwrap().into_scalar::<f64>();

        let eval = |x: f32| {
            let logits = Tensor::<4>::full([1, 1, 1, 1], x, &plain);
            let gt = Tensor::<4>::full([1, 1, 1, 1], target, &plain);
            pix.forward(&[logits], &gt, 1.0)
                .unwrap()
                .into_scalar::<f64>()
        };
        let numeric = finite_difference(eval, 0.0, 1e-2);

        // d/dx BCE(x, y) = sigmoid(x) - y = 0.5 - y at x = 0.
        assert_abs_diff_eq!(analytic, 0.5 - f64::from(target), epsilon = 1e-5);
        assert_abs_diff_eq!(analytic, numeric, epsilon = 1e-3);
    }
}

#[test]
fn pixel_and_birefnet_loss_support_negative_logits_soft_matting_batch_and_autodiff() {
    let device = Device::flex().autodiff();
    let mut weights = only(&[
        "bce",
        "iou",
        "iou_patch",
        "mae",
        "mse",
        "reg",
        "ssim",
        "cnt",
        "structure",
    ]);
    weights.iou_patch = 0.5;
    let pix = PixLossConfig::new(weights).init();

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

    for (name, value) in breakdown.components() {
        if name == "classification" {
            assert_abs_diff_eq!(value, 0.0);
        } else {
            assert!(
                value.is_finite() && value > 0.0,
                "bad component {name}: {value}"
            );
        }
    }
    let component_sum: f64 = breakdown.components().iter().map(|(_, v)| v).sum();
    assert_abs_diff_eq!(component_sum, breakdown.total, epsilon = 1e-4);
    assert_abs_diff_eq!(breakdown.total, detailed_scalar, epsilon = 1e-4);

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
    let full_loss = BiRefNetLossConfig::matting_optimized().init();
    let err = full_loss.forward(&[], &soft_target).unwrap_err();
    assert!(err.is_empty_predictions());

    let (total, breakdown) = full_loss
        .forward_detailed(&[pred_logits], &soft_target)
        .unwrap();
    assert_abs_diff_eq!(total.into_scalar::<f64>(), breakdown.total, epsilon = 1e-4);
    assert_abs_diff_eq!(breakdown.iou, 0.0);
}

#[test]
fn breakdown_sums_components_across_scales_and_includes_classification() {
    let device = Device::flex();
    let loss = BiRefNetLossConfig::new()
        .with_pixel_weights(only(&["mae"]))
        .with_classification_weight(1.0)
        .init();
    // Two scales with MAE 0.75 and 0.25 against an all-ones target.
    let preds = [
        Tensor::<4>::full([1, 1, 8, 8], -10.0, &device),
        Tensor::<4>::full([1, 1, 4, 4], 10.0, &device),
    ];
    let targets = Tensor::<4>::ones([1, 1, 8, 8], &device);
    let class_preds = [Tensor::<2>::from_floats([[0.0, 0.0]], &device)];
    let class_targets = Tensor::<1, Int>::from_ints([1], &device);

    let (total, breakdown) = loss
        .forward_detailed_with_classification(
            &preds,
            &targets,
            Some(&class_preds),
            Some(&class_targets),
        )
        .unwrap();
    // MAE: sigmoid(-10) ~ 0 -> 1.0 at scale 0, sigmoid(10) ~ 1 -> 0.0 at scale 1.
    assert_abs_diff_eq!(breakdown.mae, 1.0, epsilon = 1e-3);
    assert_abs_diff_eq!(breakdown.classification, 2.0_f64.ln(), epsilon = 1e-5);
    assert_abs_diff_eq!(
        breakdown.total,
        breakdown.mae + breakdown.classification,
        epsilon = 1e-5
    );
    assert_abs_diff_eq!(total.into_scalar::<f64>(), breakdown.total, epsilon = 1e-5);
}

#[test]
fn presets_enable_only_upstream_components() {
    let device = Device::flex();
    let preds = [Tensor::<4>::full([1, 1, 64, 64], -1.0, &device)];
    let targets = Tensor::<4>::ones([1, 1, 64, 64], &device);

    for (preset, enabled) in [
        (
            BiRefNetLossConfig::dis_optimized(),
            &["bce", "iou", "ssim"][..],
        ),
        (
            BiRefNetLossConfig::general_optimized(),
            &["bce", "iou", "mae", "ssim"][..],
        ),
        (
            BiRefNetLossConfig::matting_optimized(),
            &["bce", "mae", "ssim"][..],
        ),
    ] {
        let (_, breakdown) = preset.init().forward_detailed(&preds, &targets).unwrap();
        for (name, value) in breakdown.components() {
            assert_eq!(
                value != 0.0,
                enabled.contains(&name),
                "component {name} = {value}"
            );
        }
    }
}
