//! Linear-time Euclidean and Manhattan distance transforms for `[B, C, H, W]` masks.

use burn::tensor::{Shape, Tensor, TensorData};

const INF_SQ: f32 = 1.0e20;
const INF_THRESHOLD: f32 = 1.0e18;

fn usize_to_f32(value: usize) -> f32 {
    #[expect(
        clippy::cast_precision_loss,
        reason = "image dimensions fit comfortably within exact 24-bit mantissa"
    )]
    {
        value as f32
    }
}

/// Exact O(n) 1D squared Euclidean distance transform via lower envelope of parabolas
/// (Felzenszwalb & Huttenlocher, 2012), returning `(min_sq_dist, argmin_index)`.
fn dt1d_with_arg(costs: &[f32]) -> (Vec<f32>, Vec<usize>) {
    let len = costs.len();
    if len == 0 {
        return (Vec::new(), Vec::new());
    }

    let Some(first_site) = costs.iter().position(|&val| val < INF_THRESHOLD) else {
        return (vec![INF_SQ; len], vec![0; len]);
    };

    let mut parabola_indices = vec![0_usize; len];
    let mut boundaries = vec![0.0_f32; len + 1];
    let mut hull_len = 0_usize;
    parabola_indices[0] = first_site;
    boundaries[0] = f32::NEG_INFINITY;
    boundaries[1] = f32::INFINITY;

    for idx in (first_site + 1)..len {
        if costs[idx] >= INF_THRESHOLD {
            continue;
        }
        let site_cost = costs[idx];
        let site_pos = usize_to_f32(idx);
        let site_sq = site_pos * site_pos;
        let intersect = loop {
            let prev_idx = parabola_indices[hull_len];
            let prev_pos = usize_to_f32(prev_idx);
            let candidate = ((site_cost + site_sq) - (costs[prev_idx] + prev_pos * prev_pos))
                / (2.0 * (site_pos - prev_pos));
            if hull_len > 0 && candidate <= boundaries[hull_len] {
                hull_len -= 1;
            } else {
                break candidate;
            }
        };
        hull_len += 1;
        parabola_indices[hull_len] = idx;
        boundaries[hull_len] = intersect;
        boundaries[hull_len + 1] = f32::INFINITY;
    }

    let mut sq_dists = vec![0.0_f32; len];
    let mut argmin = vec![0_usize; len];
    let mut envelope_idx = 0_usize;
    for idx in 0..len {
        let pos = usize_to_f32(idx);
        while boundaries[envelope_idx + 1] < pos {
            envelope_idx += 1;
        }
        let site = parabola_indices[envelope_idx];
        let diff = pos - usize_to_f32(site);
        sq_dists[idx] = diff.mul_add(diff, costs[site]);
        argmin[idx] = site;
    }

    (sq_dists, argmin)
}

fn edt_plane(plane: &[f32], height: usize, width: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let hw = height * width;
    let mut dist = vec![0.0_f32; hw];
    let mut idx_y = vec![0.0_f32; hw];
    let mut idx_x = vec![0.0_f32; hw];

    if height == 0 || width == 0 {
        return (dist, idx_y, idx_x);
    }

    if !plane.iter().any(|&val| val < 0.5) {
        let fallback = usize_to_f32(height + width);
        dist.fill(fallback);
        return (dist, idx_y, idx_x);
    }

    let mut row_sq = vec![INF_SQ; hw];
    let mut row_nx = vec![0_usize; hw];
    let mut row_in = vec![0.0_f32; width];

    for y in 0..height {
        let row_offset = y * width;
        for (x, dst) in row_in.iter_mut().enumerate() {
            *dst = if plane[row_offset + x] < 0.5 {
                0.0
            } else {
                INF_SQ
            };
        }
        let (d1, a1) = dt1d_with_arg(&row_in);
        row_sq[row_offset..row_offset + width].copy_from_slice(&d1);
        row_nx[row_offset..row_offset + width].copy_from_slice(&a1);
    }

    let mut col_in = vec![0.0_f32; height];
    for x in 0..width {
        for y in 0..height {
            col_in[y] = row_sq[y * width + x];
        }
        let (d2, a2) = dt1d_with_arg(&col_in);
        for y in 0..height {
            let best_y = a2[y];
            let best_x = row_nx[best_y * width + x];
            let offset = y * width + x;
            dist[offset] = d2[y].sqrt();
            idx_y[offset] = usize_to_f32(best_y);
            idx_x[offset] = usize_to_f32(best_x);
        }
    }

    (dist, idx_y, idx_x)
}

fn manhattan_plane(plane: &[f32], height: usize, width: usize) -> Vec<f32> {
    let hw = height * width;
    let mut dist = vec![0.0_f32; hw];
    if height == 0 || width == 0 {
        return dist;
    }

    if !plane.iter().any(|&val| val < 0.5) {
        dist.fill(usize_to_f32(height + width));
        return dist;
    }

    let inf = usize_to_f32(height + width) * 2.0;
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if plane[idx] < 0.5 {
                dist[idx] = 0.0;
            } else {
                let top = if y > 0 {
                    dist[(y - 1) * width + x] + 1.0
                } else {
                    inf
                };
                let left = if x > 0 { dist[idx - 1] + 1.0 } else { inf };
                dist[idx] = top.min(left);
            }
        }
    }

    for y in (0..height).rev() {
        for x in (0..width).rev() {
            let idx = y * width + x;
            let bottom = if y + 1 < height {
                dist[(y + 1) * width + x] + 1.0
            } else {
                inf
            };
            let right = if x + 1 < width {
                dist[idx + 1] + 1.0
            } else {
                inf
            };
            dist[idx] = dist[idx].min(bottom).min(right);
        }
    }

    dist
}

/// Computes the exact 2D Euclidean distance transform of a `[B, C, H, W]` binary mask
/// (`< 0.5` = foreground object, `>= 0.5` = background), returning `(distances, (indices_y, indices_x))`.
///
/// # Panics
///
/// Panics if `binary_image` data cannot be converted to `f32`.
#[must_use]
pub fn euclidean_distance_transform(
    binary_image: Tensor<4>,
) -> (Tensor<4>, (Tensor<4>, Tensor<4>)) {
    let [batch_size, channels, height, width] = binary_image.dims();
    let device = binary_image.device();
    let values = binary_image
        .into_data()
        .try_to_vec::<f32>()
        .expect("distance transform input must be convertible to f32");

    let plane_len = height * width;
    let total = batch_size * channels * plane_len;
    let mut dist_out = Vec::with_capacity(total);
    let mut row_indices_out = Vec::with_capacity(total);
    let mut col_indices_out = Vec::with_capacity(total);

    for plane in values.chunks_exact(plane_len) {
        let (dists, row_indices, col_indices) = edt_plane(plane, height, width);
        dist_out.extend_from_slice(&dists);
        row_indices_out.extend_from_slice(&row_indices);
        col_indices_out.extend_from_slice(&col_indices);
    }

    let shape = Shape::new([batch_size, channels, height, width]);
    let distances = Tensor::<4>::from_data(TensorData::new(dist_out, shape.clone()), &device);
    let indices_y =
        Tensor::<4>::from_data(TensorData::new(row_indices_out, shape.clone()), &device);
    let indices_x = Tensor::<4>::from_data(TensorData::new(col_indices_out, shape), &device);
    (distances, (indices_y, indices_x))
}

/// Computes only the Euclidean distance map without tracking nearest-foreground coordinate tensors.
///
/// # Panics
///
/// Panics if `binary_image` data cannot be converted to `f32`.
#[must_use]
pub fn euclidean_distance_transform_simple(binary_image: Tensor<4>) -> Tensor<4> {
    let [batch_size, channels, height, width] = binary_image.dims();
    let device = binary_image.device();
    let values = binary_image
        .into_data()
        .try_to_vec::<f32>()
        .expect("distance transform input must be convertible to f32");

    let plane_len = height * width;
    let mut dist_out = Vec::with_capacity(batch_size * channels * plane_len);
    for plane in values.chunks_exact(plane_len) {
        let (dists, _, _) = edt_plane(plane, height, width);
        dist_out.extend_from_slice(&dists);
    }

    Tensor::<4>::from_data(
        TensorData::new(dist_out, Shape::new([batch_size, channels, height, width])),
        &device,
    )
}

/// Computes the 2D Manhattan (L1) distance transform of a `[B, C, H, W]` binary mask
/// (`< 0.5` = foreground, `>= 0.5` = background).
///
/// # Panics
///
/// Panics if `binary_image` data cannot be converted to `f32`.
#[must_use]
pub fn manhattan_distance_transform(binary_image: Tensor<4>) -> Tensor<4> {
    let [batch_size, channels, height, width] = binary_image.dims();
    let device = binary_image.device();
    let values = binary_image
        .into_data()
        .try_to_vec::<f32>()
        .expect("distance transform input must be convertible to f32");

    let plane_len = height * width;
    let mut dist_out = Vec::with_capacity(batch_size * channels * plane_len);
    for plane in values.chunks_exact(plane_len) {
        dist_out.extend_from_slice(&manhattan_plane(plane, height, width));
    }

    Tensor::<4>::from_data(
        TensorData::new(dist_out, Shape::new([batch_size, channels, height, width])),
        &device,
    )
}

/// SciPy-compatible `bwdist` wrapper returning distances and optional nearest `(y, x)` coordinates.
#[must_use]
pub fn bwdist(
    binary_image: Tensor<4>,
    return_indices: bool,
) -> (Tensor<4>, Option<(Tensor<4>, Tensor<4>)>) {
    if return_indices {
        let (dist, indices) = euclidean_distance_transform(binary_image);
        (dist, Some(indices))
    } else {
        let dist = euclidean_distance_transform_simple(binary_image);
        (dist, None)
    }
}
