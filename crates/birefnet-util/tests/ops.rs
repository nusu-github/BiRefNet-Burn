//! Integration tests for tensor utilities in `birefnet-util` (no image codecs required).

use approx::assert_abs_diff_eq;
use birefnet_util::{
    distance::{
        euclidean_distance_transform, euclidean_distance_transform_with_indices,
        manhattan_distance_transform, nearest_sites,
    },
    filters::{custom_filter, gaussian_filter_matlab, gaussian_kernel, matlab_gaussian_2d},
    foreground::{refine_foreground, refine_foreground_batch, refine_foreground_core},
    morphology::{StructuringElement, closing, dilation, erosion, gradient, opening},
    normalize::{IMAGENET_MEAN, IMAGENET_STD, apply_imagenet_normalization, apply_mask},
};
use burn::{prelude::*, tensor::Distribution};

fn values<const D: usize>(tensor: Tensor<D>) -> Vec<f32> {
    tensor.into_data().try_to_vec::<f32>().unwrap()
}

fn mask_from_rows<const H: usize, const W: usize>(
    rows: [[f32; W]; H],
    device: &Device,
) -> Tensor<4> {
    Tensor::<4>::from_floats([[rows]], device)
}

#[test]
fn euclidean_and_manhattan_distance_transforms_match_exact_geometry() {
    let device = Device::flex();

    // Single site (0.0) at (1, 2) in a 3x4 mask.
    let mut grid = [[1.0_f32; 4]; 3];
    grid[1][2] = 0.0;
    let mask = mask_from_rows(grid, &device);

    let (edt, (idx_y, idx_x)) = euclidean_distance_transform_with_indices(&mask).unwrap();
    let (edt_vals, row_vals, col_vals) = (values(edt), values(idx_y), values(idx_x));
    for y in 0_u16..3 {
        for x in 0_u16..4 {
            let i = usize::from(y) * 4 + usize::from(x);
            let expected = (f32::from(y) - 1.0).hypot(f32::from(x) - 2.0);
            assert_abs_diff_eq!(edt_vals[i], expected, epsilon = 1e-5);
            assert_abs_diff_eq!(row_vals[i], 1.0);
            assert_abs_diff_eq!(col_vals[i], 2.0);
        }
    }

    let dist_only = values(euclidean_distance_transform(&mask).unwrap());
    assert_eq!(dist_only, edt_vals);

    let l1 = values(manhattan_distance_transform(&mask).unwrap());
    for y in 0_u16..3 {
        for x in 0_u16..4 {
            let i = usize::from(y) * 4 + usize::from(x);
            let expected = (f32::from(y) - 1.0).abs() + (f32::from(x) - 2.0).abs();
            assert_abs_diff_eq!(l1[i], expected);
        }
    }
}

#[test]
fn euclidean_transform_matches_brute_force_on_random_masks() {
    let device = Device::flex();
    device.seed(5);
    let mask = Tensor::<4>::random([2, 1, 9, 13], Distribution::Uniform(0.0, 1.0), &device)
        .greater_elem(0.15)
        .float();
    let mask_vals = values(mask.clone());
    let edt = values(euclidean_distance_transform(&mask).unwrap());

    for (plane_idx, plane) in mask_vals.as_chunks::<{ 9 * 13 }>().0.iter().enumerate() {
        let sites: Vec<(f32, f32)> = plane
            .iter()
            .enumerate()
            .filter(|&(_, &v)| v < 0.5)
            .map(|(i, _)| {
                (
                    f32::from(u16::try_from(i / 13).unwrap()),
                    f32::from(u16::try_from(i % 13).unwrap()),
                )
            })
            .collect();
        for (i, &got) in edt[plane_idx * 9 * 13..][..9 * 13].iter().enumerate() {
            let (y, x) = (
                f32::from(u16::try_from(i / 13).unwrap()),
                f32::from(u16::try_from(i % 13).unwrap()),
            );
            let expected = sites
                .iter()
                .map(|&(sy, sx)| (y - sy).hypot(x - sx))
                .fold(f32::INFINITY, f32::min);
            let expected = if sites.is_empty() { 22.0 } else { expected };
            assert_abs_diff_eq!(got, expected, epsilon = 1e-4);
        }
    }
}

#[test]
fn nearest_site_ties_resolve_like_scipy() {
    // Pixel (0, 2) is at squared distance 4 from both sites (0, 0) and (2, 2); SciPy's
    // `distance_transform_edt(..., return_indices=True)` reports (0, 0).
    let mut sites = vec![false; 9];
    sites[0] = true;
    sites[8] = true;
    let result = nearest_sites(&sites, [3, 3]);
    assert_eq!(result.nearest[2], 0);
    assert_abs_diff_eq!(result.distances[2], 2.0);
    assert_eq!(result.nearest[8], 8);
}

#[test]
fn distance_transform_without_sites_fills_height_plus_width() {
    let device = Device::flex();
    let all_background = Tensor::<4>::ones([1, 1, 3, 5], &device);
    for v in values(euclidean_distance_transform(&all_background).unwrap()) {
        assert_abs_diff_eq!(v, 8.0);
    }
}

#[test]
fn morphology_operations_match_expected_binary_patterns() {
    let device = Device::flex();
    let rect = StructuringElement::rectangle(3, 3);
    let disk = StructuringElement::disk(1);

    // Isolated center dot vanishes under erosion and expands into a cross under disk dilation.
    let mut dot = [[0.0_f32; 5]; 5];
    dot[2][2] = 1.0;
    let img = mask_from_rows(dot, &device);

    let cross = values(dilation(img.clone(), &disk));
    let expected_cross: Vec<f32> = (0..25)
        .map(|i: usize| {
            let (y, x) = (i / 5, i % 5);
            if y.abs_diff(2) + x.abs_diff(2) <= 1 {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    assert_eq!(cross, expected_cross);

    let sum = |t: Tensor<4>| t.sum().into_scalar::<f32>();
    assert_abs_diff_eq!(sum(erosion(img.clone(), &disk)), 0.0);
    assert_abs_diff_eq!(sum(dilation(img.clone(), &rect)), 9.0);
    assert_abs_diff_eq!(sum(opening(img.clone(), &rect)), 0.0);
    assert_abs_diff_eq!(sum(closing(img.clone(), &rect)), 1.0);
    assert_abs_diff_eq!(sum(gradient(img, &disk)), 5.0);
}

#[test]
fn morphology_replicates_edges_and_honours_off_center_anchors() {
    let device = Device::flex();
    // Two filled top rows: with edge replication, erosion by a vertical 3x1 bar keeps row 0
    // (its replicated upper neighbor is also set) but not row 1.
    let mut img = [[0.0_f32; 4]; 4];
    img[0] = [1.0; 4];
    img[1] = [1.0; 4];
    let img = mask_from_rows(img, &device);
    let bar = StructuringElement::new(vec![true; 3], [3, 1], (1, 0)).unwrap();
    let eroded = values(erosion(img, &bar));
    assert_eq!(&eroded[..4], &[1.0; 4]);
    assert_eq!(&eroded[4..8], &[0.0; 4]);

    // Anchor at the top of a 2x1 element: dilation copies each set pixel one row up.
    let shift = StructuringElement::new(vec![true, true], [2, 1], (0, 0)).unwrap();
    let mut single = [[0.0_f32; 3]; 3];
    single[2][1] = 1.0;
    let dilated = values(dilation(mask_from_rows(single, &device), &shift));
    assert_eq!(dilated, vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]);

    let bad_len = StructuringElement::new(vec![true; 3], [2, 2], (0, 0)).unwrap_err();
    assert!(bad_len.is_invalid_dimensions());
    let bad_anchor = StructuringElement::new(vec![true; 4], [2, 2], (2, 0)).unwrap_err();
    assert!(bad_anchor.is_invalid_input());
}

#[test]
fn disk_contains_exactly_the_integer_lattice_points_within_the_radius() {
    for radius in 0..6_usize {
        let disk = StructuringElement::disk(radius);
        let size = 2 * radius + 1;
        assert_eq!(disk.dims(), [size, size]);
        let expected = (0..size * size)
            .filter(|i| {
                let (dy, dx) = ((i / size).abs_diff(radius), (i % size).abs_diff(radius));
                dy * dy + dx * dx <= radius * radius
            })
            .count();
        assert_eq!(disk.mask().iter().filter(|&&on| on).count(), expected);
    }
}

#[test]
fn separable_gaussian_filter_matches_the_direct_2d_kernel() {
    let device = Device::flex();
    device.seed(9);
    let img = Tensor::<4>::random([1, 2, 11, 7], Distribution::Uniform(0.0, 1.0), &device);

    let separable = values(gaussian_filter_matlab(img.clone(), (5, 3), 1.2));
    let direct = values(custom_filter(img, matlab_gaussian_2d((5, 3), 1.2, &device)));
    for (s, d) in separable.iter().zip(&direct) {
        assert_abs_diff_eq!(s, d, epsilon = 1e-5);
    }

    for kernel in [
        gaussian_kernel(5, 1.0, &device),
        matlab_gaussian_2d((5, 5), 1.0, &device),
    ] {
        let taps = values(kernel.flatten::<1>(0, 1));
        assert_abs_diff_eq!(taps.iter().sum::<f32>(), 1.0, epsilon = 1e-5);
        // Symmetric and peaked at the centre.
        assert_abs_diff_eq!(taps[0], taps[24], epsilon = 1e-7);
        assert!(taps.iter().all(|&t| t <= taps[12]));
    }
}

#[test]
fn foreground_refinement_recovers_colors_under_soft_alpha() {
    let device = Device::flex();
    // A constant-color image is a fixed point of blur fusion for any alpha.
    let image = Tensor::<4>::full([2, 3, 12, 12], 0.4, &device);
    let alpha = Tensor::<4>::random([2, 1, 12, 12], Distribution::Uniform(0.1, 0.9), &device);
    for v in values(refine_foreground_batch(image, alpha, Some(4))) {
        assert_abs_diff_eq!(v, 0.4, epsilon = 1e-3);
    }

    // Composite a red foreground over a blue background: refinement under the true alpha
    // pulls interior pixels back towards red.
    let alpha_row: [f32; 12] = [0.0, 0.0, 0.0, 0.3, 0.7, 1.0, 1.0, 0.7, 0.3, 0.0, 0.0, 0.0];
    let alpha = Tensor::<1>::from_floats(alpha_row, &device)
        .reshape([1, 1, 1, 12])
        .repeat_dim(2, 12);
    let red = Tensor::<1>::from_floats([1.0, 0.0, 0.0], &device).reshape([1, 3, 1, 1]);
    let blue = Tensor::<1>::from_floats([0.0, 0.0, 1.0], &device).reshape([1, 3, 1, 1]);
    let composite = alpha.clone() * red + alpha.clone().neg().add_scalar(1.0) * blue;
    let refined = refine_foreground_core(composite.clone(), alpha, 3);
    let red_channel = |t: Tensor<4>| values(t.slice(s![.., 0..1, 6..7, 3..4]))[0];
    assert!(red_channel(refined.clone()) > red_channel(composite) + 0.2);
    assert_eq!(refined.dims(), [1, 3, 12, 12]);

    let single = refine_foreground(
        Tensor::<3>::full([3, 8, 8], 0.4, &device),
        Tensor::<2>::ones([8, 8], &device),
        Some(3),
    );
    for v in values(single) {
        assert_abs_diff_eq!(v, 0.4, epsilon = 1e-3);
    }
}

#[test]
fn normalize_and_apply_mask_work_without_image_io() {
    let device = Device::flex();
    let rgb = Tensor::<4>::zeros([1, 3, 2, 2], &device);
    let norm_vals = values(apply_imagenet_normalization(rgb.clone()).unwrap());
    for c in 0..3 {
        let expected = -IMAGENET_MEAN[c] / IMAGENET_STD[c];
        for p in 0..4 {
            assert_abs_diff_eq!(norm_vals[c * 4 + p], expected, epsilon = 1e-5);
        }
    }
    assert!(
        apply_imagenet_normalization(Tensor::zeros([1, 1, 2, 2], &device))
            .unwrap_err()
            .is_invalid_dimensions()
    );

    let mask = Tensor::<4>::full([1, 1, 2, 2], 0.75, &device);
    let rgba = apply_mask(rgb.clone() + 1.0, mask.clone()).unwrap();
    assert_eq!(rgba.dims(), [1, 4, 2, 2]);
    let rgba_vals = values(rgba);
    assert!(rgba_vals[..12].iter().all(|&v| (v - 1.0).abs() < 1e-6));
    assert!(rgba_vals[12..].iter().all(|&v| (v - 0.75).abs() < 1e-6));

    let gray = Tensor::<4>::ones([1, 2, 2, 2], &device);
    for v in values(apply_mask(gray, mask).unwrap()) {
        assert_abs_diff_eq!(v, 0.75);
    }
    assert!(
        apply_mask(rgb, Tensor::zeros([1, 1, 3, 2], &device))
            .unwrap_err()
            .is_invalid_dimensions()
    );
}
