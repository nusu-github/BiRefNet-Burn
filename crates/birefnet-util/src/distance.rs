//! Exact Euclidean and Manhattan distance transforms of `[B, C, H, W]` binary masks.
//!
//! Pixels `< 0.5` are *sites* (distance `0`); every other pixel receives its distance to the
//! nearest site. A plane without any site is filled with `H + W`. Transforms run on the host
//! and return tensors on the input device in the input float dtype.

use burn::tensor::{Shape, Tensor, TensorData};

use crate::error::UtilError;

/// Squared distance used for "no site yet"; far above any image-sized squared distance.
const INF_SQ: f32 = 1.0e20;
/// Costs at or above this value are treated as "no site".
const INF_THRESHOLD: f32 = 1.0e18;

fn usize_to_f32(value: usize) -> f32 {
    #[expect(
        clippy::cast_precision_loss,
        reason = "image coordinates stay far below 2^24 and convert exactly"
    )]
    let value = value as f32;
    value
}

/// Reusable buffers for the 1D lower-envelope transform, sized to the longest image axis.
struct Dt1d {
    sites: Vec<usize>,
    bounds: Vec<f32>,
    sq: Vec<f32>,
    arg: Vec<usize>,
}

impl Dt1d {
    fn with_capacity(len: usize) -> Self {
        Self {
            sites: vec![0; len],
            bounds: vec![0.0; len + 1],
            sq: vec![0.0; len],
            arg: vec![0; len],
        }
    }

    /// Exact O(n) 1D squared distance transform via the lower envelope of parabolas
    /// (Felzenszwalb & Huttenlocher, 2012); fills `sq[..n]` and `arg[..n]`.
    fn run(&mut self, costs: &[f32]) {
        let len = costs.len();
        let (sq, arg) = (&mut self.sq[..len], &mut self.arg[..len]);
        let Some(first_site) = costs.iter().position(|&val| val < INF_THRESHOLD) else {
            sq.fill(INF_SQ);
            arg.fill(0);
            return;
        };

        let (sites, bounds) = (&mut self.sites, &mut self.bounds);
        let mut hull_len = 0_usize;
        sites[0] = first_site;
        bounds[0] = f32::NEG_INFINITY;
        bounds[1] = f32::INFINITY;

        for (idx, &site_cost) in costs.iter().enumerate().skip(first_site + 1) {
            if site_cost >= INF_THRESHOLD {
                continue;
            }
            let site_pos = usize_to_f32(idx);
            let site_sq = site_pos * site_pos;
            let intersect = loop {
                let prev_idx = sites[hull_len];
                let prev_pos = usize_to_f32(prev_idx);
                let candidate = ((site_cost + site_sq)
                    - prev_pos.mul_add(prev_pos, costs[prev_idx]))
                    / (2.0 * (site_pos - prev_pos));
                if hull_len > 0 && candidate <= bounds[hull_len] {
                    hull_len -= 1;
                } else {
                    break candidate;
                }
            };
            hull_len += 1;
            sites[hull_len] = idx;
            bounds[hull_len] = intersect;
            bounds[hull_len + 1] = f32::INFINITY;
        }

        let mut envelope_idx = 0_usize;
        for (idx, (dst_sq, dst_arg)) in sq.iter_mut().zip(arg.iter_mut()).enumerate() {
            let pos = usize_to_f32(idx);
            while bounds[envelope_idx + 1] < pos {
                envelope_idx += 1;
            }
            let site = sites[envelope_idx];
            let diff = pos - usize_to_f32(site);
            *dst_sq = diff.mul_add(diff, costs[site]);
            *dst_arg = site;
        }
    }
}

/// Per-plane outputs of the Euclidean transform; `nearest` is filled only when requested.
struct EdtOutput {
    dist: Vec<f32>,
    /// Flat `row * width + col` index of the nearest site within its plane.
    nearest: Vec<usize>,
}

/// Computes the 2D EDT of one `height x width` plane, appending results to `out`.
fn edt_plane(
    is_site: impl Fn(usize) -> bool,
    [height, width]: [usize; 2],
    scratch: &mut EdtScratch,
    out: &mut EdtOutput,
) {
    let plane_len = height * width;
    let track_indices = scratch.track_indices;
    if !(0..plane_len).any(&is_site) {
        let fallback = usize_to_f32(height + width);
        out.dist.extend(std::iter::repeat_n(fallback, plane_len));
        if track_indices {
            out.nearest.extend(std::iter::repeat_n(0, plane_len));
        }
        return;
    }

    // Pass 1 runs down columns and pass 2 along rows; with ties resolved towards the earlier
    // site this reproduces SciPy's nearest-site choice exactly.
    for col in 0..width {
        for (row, dst) in scratch.line[..height].iter_mut().enumerate() {
            *dst = if is_site(row * width + col) {
                0.0
            } else {
                INF_SQ
            };
        }
        scratch.dt.run(&scratch.line[..height]);
        for row in 0..height {
            scratch.col_sq[row * width + col] = scratch.dt.sq[row];
            if track_indices {
                scratch.col_arg[row * width + col] = scratch.dt.arg[row];
            }
        }
    }

    let base = out.dist.len();
    out.dist.resize(base + plane_len, 0.0);
    if track_indices {
        out.nearest.resize(base + plane_len, 0);
    }
    for row in 0..height {
        let offset = row * width;
        scratch.dt.run(&scratch.col_sq[offset..offset + width]);
        for col in 0..width {
            out.dist[base + offset + col] = scratch.dt.sq[col].sqrt();
            if track_indices {
                let best_col = scratch.dt.arg[col];
                out.nearest[base + offset + col] =
                    scratch.col_arg[offset + best_col] * width + best_col;
            }
        }
    }
}

/// Buffers shared by every plane of one transform call.
struct EdtScratch {
    track_indices: bool,
    dt: Dt1d,
    line: Vec<f32>,
    /// Column-pass squared distances, row-major.
    col_sq: Vec<f32>,
    /// Column-pass nearest-site rows, row-major.
    col_arg: Vec<usize>,
}

impl EdtScratch {
    fn new([height, width]: [usize; 2], track_indices: bool) -> Self {
        Self {
            track_indices,
            dt: Dt1d::with_capacity(height.max(width)),
            line: vec![0.0; height],
            col_sq: vec![0.0; height * width],
            col_arg: if track_indices {
                vec![0; height * width]
            } else {
                Vec::new()
            },
        }
    }
}

/// Reads a `[B, C, H, W]` tensor to host `f32` values regardless of its float dtype.
fn read_planes(tensor: &Tensor<4>) -> Result<Vec<f32>, UtilError> {
    tensor
        .try_to_vec_as::<f32>()
        .map_err(UtilError::tensor_read)
}

/// Uploads `values` with the shape, device, and float dtype of `like`.
fn upload_like(values: Vec<f32>, dims: [usize; 4], like: &Tensor<4>) -> Tensor<4> {
    Tensor::from_data(
        TensorData::new(values, Shape::new(dims)),
        (&like.device(), like.dtype()),
    )
}

fn run_edt(binary_image: &Tensor<4>, track_indices: bool) -> Result<EdtOutput, UtilError> {
    let [_, _, height, width] = binary_image.dims();
    let values = read_planes(binary_image)?;
    let mut out = EdtOutput {
        dist: Vec::with_capacity(values.len()),
        nearest: Vec::with_capacity(if track_indices { values.len() } else { 0 }),
    };
    if height == 0 || width == 0 {
        return Ok(out);
    }

    let mut scratch = EdtScratch::new([height, width], track_indices);
    for plane in values.chunks_exact(height * width) {
        edt_plane(|i| plane[i] < 0.5, [height, width], &mut scratch, &mut out);
    }
    Ok(out)
}

/// Exact Euclidean distance transform of one host-side plane.
#[derive(Debug, Clone, PartialEq)]
pub struct NearestSites {
    /// Distance from every pixel to its nearest site, row-major.
    pub distances: Vec<f32>,
    /// Flat `row * width + col` index of every pixel's nearest site.
    pub nearest: Vec<usize>,
}

/// Computes, on the host, the distance from every pixel to the nearest `true` site and the
/// site's flat index (SciPy `distance_transform_edt(~is_site, return_indices=True)`).
///
/// A plane without any site gets distance `height + width` and index `0` everywhere.
///
/// # Panics
///
/// Panics if `is_site.len() != height * width`.
#[must_use]
pub fn nearest_sites(is_site: &[bool], [height, width]: [usize; 2]) -> NearestSites {
    assert_eq!(
        is_site.len(),
        height * width,
        "site mask length {} does not match {height}x{width}",
        is_site.len()
    );
    let mut out = EdtOutput {
        dist: Vec::with_capacity(is_site.len()),
        nearest: Vec::with_capacity(is_site.len()),
    };
    if height > 0 && width > 0 {
        let mut scratch = EdtScratch::new([height, width], true);
        edt_plane(|i| is_site[i], [height, width], &mut scratch, &mut out);
    }
    NearestSites {
        distances: out.dist,
        nearest: out.nearest,
    }
}

/// Computes the exact Euclidean distance transform and the coordinates of the nearest site.
///
/// Returns `(distances, (rows, cols))`, where `rows`/`cols` hold the `(y, x)` index of the
/// closest pixel `< 0.5` (SciPy `distance_transform_edt(..., return_indices=True)`).
///
/// # Errors
///
/// Returns [`UtilError`] if the tensor cannot be read back from its device.
pub fn euclidean_distance_transform_with_indices(
    binary_image: &Tensor<4>,
) -> Result<(Tensor<4>, (Tensor<4>, Tensor<4>)), UtilError> {
    let dims @ [.., width] = binary_image.dims();
    let out = run_edt(binary_image, true)?;
    let (rows, cols) = out
        .nearest
        .iter()
        .map(|&idx| (usize_to_f32(idx / width), usize_to_f32(idx % width)))
        .unzip();
    Ok((
        upload_like(out.dist, dims, binary_image),
        (
            upload_like(rows, dims, binary_image),
            upload_like(cols, dims, binary_image),
        ),
    ))
}

/// Computes the exact Euclidean distance from every pixel to the nearest pixel `< 0.5`.
///
/// # Errors
///
/// Returns [`UtilError`] if the tensor cannot be read back from its device.
pub fn euclidean_distance_transform(binary_image: &Tensor<4>) -> Result<Tensor<4>, UtilError> {
    let dims = binary_image.dims();
    let out = run_edt(binary_image, false)?;
    Ok(upload_like(out.dist, dims, binary_image))
}

/// Two-pass chamfer Manhattan (L1) distance of one plane.
fn manhattan_plane(plane: &[f32], [height, width]: [usize; 2], dist: &mut Vec<f32>) {
    let base = dist.len();
    if !plane.iter().any(|&val| val < 0.5) {
        dist.extend(std::iter::repeat_n(
            usize_to_f32(height + width),
            plane.len(),
        ));
        return;
    }

    let far = usize_to_f32(height + width) * 2.0;
    dist.extend(plane.iter().map(|&val| if val < 0.5 { 0.0 } else { far }));
    let d = &mut dist[base..];

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if y > 0 {
                d[idx] = d[idx].min(d[idx - width] + 1.0);
            }
            if x > 0 {
                d[idx] = d[idx].min(d[idx - 1] + 1.0);
            }
        }
    }
    for y in (0..height).rev() {
        for x in (0..width).rev() {
            let idx = y * width + x;
            if y + 1 < height {
                d[idx] = d[idx].min(d[idx + width] + 1.0);
            }
            if x + 1 < width {
                d[idx] = d[idx].min(d[idx + 1] + 1.0);
            }
        }
    }
}

/// Computes the Manhattan (L1) distance from every pixel to the nearest pixel `< 0.5`.
///
/// # Errors
///
/// Returns [`UtilError`] if the tensor cannot be read back from its device.
pub fn manhattan_distance_transform(binary_image: &Tensor<4>) -> Result<Tensor<4>, UtilError> {
    let dims @ [_, _, height, width] = binary_image.dims();
    let values = read_planes(binary_image)?;
    let mut dist = Vec::with_capacity(values.len());
    if height > 0 && width > 0 {
        for plane in values.chunks_exact(height * width) {
            manhattan_plane(plane, [height, width], &mut dist);
        }
    }
    Ok(upload_like(dist, dims, binary_image))
}
