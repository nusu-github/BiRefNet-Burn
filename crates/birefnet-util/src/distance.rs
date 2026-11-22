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

/// Per-plane outputs of the Euclidean transform; index planes are filled only when requested.
struct EdtOutput {
    dist: Vec<f32>,
    rows: Vec<f32>,
    cols: Vec<f32>,
}

/// Computes the 2D EDT of one `height x width` plane, appending results to `out`.
fn edt_plane(
    plane: &[f32],
    [height, width]: [usize; 2],
    scratch: &mut EdtScratch,
    out: &mut EdtOutput,
) {
    let track_indices = scratch.track_indices;
    if !plane.iter().any(|&val| val < 0.5) {
        let fallback = usize_to_f32(height + width);
        out.dist.extend(std::iter::repeat_n(fallback, plane.len()));
        if track_indices {
            out.rows.extend(std::iter::repeat_n(0.0, plane.len()));
            out.cols.extend(std::iter::repeat_n(0.0, plane.len()));
        }
        return;
    }

    // Pass 1: rows.
    for (row, src) in plane.chunks_exact(width).enumerate() {
        for (dst, &val) in scratch.line.iter_mut().zip(src) {
            *dst = if val < 0.5 { 0.0 } else { INF_SQ };
        }
        scratch.dt.run(&scratch.line[..width]);
        let offset = row * width;
        scratch.row_sq[offset..offset + width].copy_from_slice(&scratch.dt.sq[..width]);
        if track_indices {
            scratch.row_arg[offset..offset + width].copy_from_slice(&scratch.dt.arg[..width]);
        }
    }

    // Pass 2: columns over the row-wise squared distances.
    let base = out.dist.len();
    out.dist.resize(base + plane.len(), 0.0);
    if track_indices {
        out.rows.resize(base + plane.len(), 0.0);
        out.cols.resize(base + plane.len(), 0.0);
    }
    for col in 0..width {
        for (row, dst) in scratch.line[..height].iter_mut().enumerate() {
            *dst = scratch.row_sq[row * width + col];
        }
        scratch.dt.run(&scratch.line[..height]);
        for row in 0..height {
            let offset = base + row * width + col;
            out.dist[offset] = scratch.dt.sq[row].sqrt();
            if track_indices {
                let best_row = scratch.dt.arg[row];
                out.rows[offset] = usize_to_f32(best_row);
                out.cols[offset] = usize_to_f32(scratch.row_arg[best_row * width + col]);
            }
        }
    }
}

/// Buffers shared by every plane of one transform call.
struct EdtScratch {
    track_indices: bool,
    dt: Dt1d,
    line: Vec<f32>,
    row_sq: Vec<f32>,
    row_arg: Vec<usize>,
}

impl EdtScratch {
    fn new([height, width]: [usize; 2], track_indices: bool) -> Self {
        let longest = height.max(width);
        Self {
            track_indices,
            dt: Dt1d::with_capacity(longest),
            line: vec![0.0; longest],
            row_sq: vec![0.0; height * width],
            row_arg: if track_indices {
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
        rows: Vec::with_capacity(if track_indices { values.len() } else { 0 }),
        cols: Vec::with_capacity(if track_indices { values.len() } else { 0 }),
    };
    if height == 0 || width == 0 {
        return Ok(out);
    }

    let mut scratch = EdtScratch::new([height, width], track_indices);
    for plane in values.chunks_exact(height * width) {
        edt_plane(plane, [height, width], &mut scratch, &mut out);
    }
    Ok(out)
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
    let dims = binary_image.dims();
    let out = run_edt(binary_image, true)?;
    Ok((
        upload_like(out.dist, dims, binary_image),
        (
            upload_like(out.rows, dims, binary_image),
            upload_like(out.cols, dims, binary_image),
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
