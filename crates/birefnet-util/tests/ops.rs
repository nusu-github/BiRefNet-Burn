//! Integration tests for pure-tensor operations in `birefnet-util`.

use approx::assert_abs_diff_eq;
use birefnet_util::{
    IMAGENET_MEAN, IMAGENET_STD, StructuringElement, apply_imagenet_normalization, apply_mask,
    argwhere, bwdist, closing, count_nonzero, cumsum_1d, cumsum_2d_axis0, dilation, erosion,
    euclidean_distance_transform, flip_1d, gaussian_filter_matlab, gaussian_kernel, gradient,
    histogram, manhattan_distance_transform, matlab_gaussian_2d, opening, refine_foreground,
    refine_foreground_batch, refine_foreground_core, std_with_ddof,
};
use burn::prelude::*;

#[test]
fn array_ops_handle_finite_and_edge_inputs() {
    let device = Device::flex();

    // Histogram skips NaN/Inf and counts finite values accurately.
    let vals = Tensor::<1>::from_floats([0.1, 0.9, 1.5, f32::NAN, f32::INFINITY], &device);
    let hist = histogram(vals, 2, Some((0.0, 2.0)))
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();
    assert_eq!(hist, vec![2.0, 1.0]);

    // Constant range does not produce NaN.
    let constant = Tensor::<1>::from_floats([3.0, 3.0, 3.0], &device);
    let const_hist = histogram(constant, 4, None)
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();
    assert_eq!(const_hist, vec![3.0, 0.0, 0.0, 0.0]);

    // Cumsum, flip, count_nonzero, argwhere, std_with_ddof.
    let seq = Tensor::<1>::from_floats([1.0, 2.0, 3.0], &device);
    assert_eq!(
        cumsum_1d(seq.clone())
            .into_data()
            .try_to_vec::<f32>()
            .unwrap(),
        vec![1.0, 3.0, 6.0]
    );
    assert_eq!(
        flip_1d(seq).into_data().try_to_vec::<f32>().unwrap(),
        vec![3.0, 2.0, 1.0]
    );

    let mat = Tensor::<2>::from_floats([[1.0, 0.0], [0.0, 2.0]], &device);
    assert_eq!(
        cumsum_2d_axis0(mat.clone())
            .into_data()
            .try_to_vec::<f32>()
            .unwrap(),
        vec![1.0, 0.0, 1.0, 2.0]
    );
    assert_abs_diff_eq!(count_nonzero(mat.clone()), 2.0, epsilon = 1e-6);
    assert_eq!(argwhere(mat), vec![[0, 0], [1, 1]]);

    let sample = Tensor::<1>::from_floats([1.0, 2.0, 3.0, 4.0, 5.0], &device);
    assert_abs_diff_eq!(std_with_ddof(sample, 1), 2.5_f64.sqrt(), epsilon = 1e-5);
}

#[test]
fn euclidean_and_manhattan_distance_transforms_match_exact_geometry() {
    let device = Device::flex();

    // Single foreground pixel (0.0) at (1, 2) in a 3x4 mask.
    let mut grid = [[1.0_f32; 4]; 3];
    grid[1][2] = 0.0;
    let mask = Tensor::<4>::from_floats([[grid]], &device);

    let (edt, (idx_y, idx_x)) = euclidean_distance_transform(mask.clone());
    let edt_vals = edt.into_data().try_to_vec::<f32>().unwrap();
    let row_vals = idx_y.into_data().try_to_vec::<f32>().unwrap();
    let col_vals = idx_x.into_data().try_to_vec::<f32>().unwrap();

    for y in 0_u16..3 {
        for x in 0_u16..4 {
            let i = usize::from(y) * 4 + usize::from(x);
            let expected = (f32::from(y) - 1.0).hypot(f32::from(x) - 2.0);
            assert_abs_diff_eq!(edt_vals[i], expected, epsilon = 1e-5);
            assert_abs_diff_eq!(row_vals[i], 1.0, epsilon = 1e-6);
            assert_abs_diff_eq!(col_vals[i], 2.0, epsilon = 1e-6);
        }
    }

    let l1_vals = manhattan_distance_transform(mask.clone())
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();
    assert_abs_diff_eq!(l1_vals[0], 3.0, epsilon = 1e-6);
    assert_abs_diff_eq!(l1_vals[6], 0.0, epsilon = 1e-6);

    let (bw_dist, bw_idx) = bwdist(mask, false);
    assert!(bw_idx.is_none());
    assert_abs_diff_eq!(
        bw_dist.into_data().try_to_vec::<f32>().unwrap()[0],
        5.0_f32.sqrt(),
        epsilon = 1e-5
    );
}

#[test]
fn morphology_operations_match_expected_binary_patterns() {
    let device = Device::flex();
    let se_rect = StructuringElement::rectangle(3, 3, &device);
    let se_disk = StructuringElement::disk(1, &device);

    // Isolated center dot vanishes under erosion and expands into a cross under disk dilation.
    let mut dot = [[0.0_f32; 5]; 5];
    dot[2][2] = 1.0;
    let img = Tensor::<4>::from_floats([[dot]], &device);

    assert_abs_diff_eq!(
        erosion(img.clone(), &se_disk).sum().into_scalar::<f32>(),
        0.0,
        epsilon = 1e-6
    );
    assert_abs_diff_eq!(
        dilation(img.clone(), &se_disk).sum().into_scalar::<f32>(),
        5.0,
        epsilon = 1e-6
    );
    assert_abs_diff_eq!(
        opening(img.clone(), &se_rect).sum().into_scalar::<f32>(),
        0.0,
        epsilon = 1e-6
    );
    assert_abs_diff_eq!(
        closing(img.clone(), &se_rect).sum().into_scalar::<f32>(),
        1.0,
        epsilon = 1e-6
    );
    assert_abs_diff_eq!(
        gradient(img, &se_disk).sum().into_scalar::<f32>(),
        5.0,
        epsilon = 1e-6
    );
}

#[test]
fn gaussian_filters_and_foreground_refinement_preserve_constants_and_shapes() {
    let device = Device::flex();
    let k1 = gaussian_kernel(5, 1.0, &device);
    let k2 = matlab_gaussian_2d((5, 5), 1.0, &device);
    assert_abs_diff_eq!(k1.sum().into_scalar::<f32>(), 1.0, epsilon = 1e-5);
    assert_abs_diff_eq!(k2.sum().into_scalar::<f32>(), 1.0, epsilon = 1e-5);

    let img = Tensor::<4>::full([1, 2, 8, 8], 0.5, &device);
    let filtered = gaussian_filter_matlab(img, (3, 3), 1.0)
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();
    for v in filtered {
        assert_abs_diff_eq!(v, 0.5, epsilon = 1e-5);
    }

    let c3 = refine_foreground(
        Tensor::<3>::full([3, 8, 8], 0.4, &device),
        Tensor::<2>::ones([8, 8], &device),
        Some(3),
    );
    assert_eq!(c3.dims(), [3, 8, 8]);

    let core = refine_foreground_core(
        Tensor::<4>::full([1, 3, 8, 8], 0.5, &device),
        Tensor::<4>::ones([1, 1, 8, 8], &device),
        2,
    );
    assert_eq!(core.dims(), [1, 3, 8, 8]);

    let batch = refine_foreground_batch(
        Tensor::<4>::full([2, 3, 8, 8], 0.5, &device),
        Tensor::<4>::ones([2, 1, 8, 8], &device),
        Some(2),
    );
    assert_eq!(batch.dims(), [2, 3, 8, 8]);
}

#[test]
fn normalize_and_apply_mask_work_without_image_io() {
    let device = Device::flex();
    let rgb = Tensor::<4>::zeros([1, 3, 2, 2], &device);
    let normed = apply_imagenet_normalization(rgb.clone()).unwrap();
    let norm_vals = normed.into_data().try_to_vec::<f32>().unwrap();

    for c in 0..3 {
        let expected = -IMAGENET_MEAN[c] / IMAGENET_STD[c];
        for p in 0..4 {
            assert_abs_diff_eq!(norm_vals[c * 4 + p], expected, epsilon = 1e-5);
        }
    }

    let mask = Tensor::<4>::full([1, 1, 2, 2], 0.75, &device);
    let rgba = apply_mask(rgb, mask).unwrap();
    assert_eq!(rgba.dims(), [1, 4, 2, 2]);
}
