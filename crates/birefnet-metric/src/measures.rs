//! F-measure, E-measure, S-measure, weighted F-measure, and boundary IoU on a [`Sample`].
//!
//! Each measure follows the `py_sod_metrics` implementation used by the BiRefNet evaluation
//! scripts. Curves are computed from one 256-bin histogram (`O(H·W + 256)`) instead of one
//! pass per threshold.

use birefnet_util::distance::nearest_sites;

use crate::{
    curve::{ThresholdCurve, ThresholdHistogram, count_f64, quantize},
    sample::{Sample, usize_f64},
};

/// `np.spacing(1)`, the stabilizer used throughout `py_sod_metrics`.
const EPS: f64 = f64::EPSILON;

/// Adaptive and per-threshold F-measure of one sample (or the average over many).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct FMeasure {
    /// F-measure at the adaptive threshold `min(2 * mean(pred), 1)` (`adpFm`).
    pub adaptive: f64,
    /// F-measure at each 8-bit threshold (`meanFm` / `maxFm` are its mean / max).
    pub curve: ThresholdCurve,
    /// Precision at each threshold.
    pub precision: ThresholdCurve,
    /// Recall at each threshold.
    pub recall: ThresholdCurve,
}

/// Adaptive and per-threshold enhanced-alignment measure (E-measure).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EMeasure {
    /// E-measure at the adaptive threshold (`adpEm`).
    pub adaptive: f64,
    /// E-measure at each 8-bit threshold (`meanEm` / `maxEm` are its mean / max).
    pub curve: ThresholdCurve,
}

fn f_beta(precision: f64, recall: f64, beta_sq: f64) -> f64 {
    let numerator = (1.0 + beta_sq) * precision * recall;
    if numerator == 0.0 {
        0.0
    } else {
        numerator / beta_sq.mul_add(precision, recall)
    }
}

impl Sample {
    /// Computes the adaptive F-measure and precision / recall / F curves.
    ///
    /// `beta_sq` is `β²` (`0.3` in salient object detection).
    #[must_use]
    pub fn f_measure(&self, beta_sq: f64) -> FMeasure {
        let gt_count = count_f64(self.foreground_count() as u64).max(1.0);

        let (tp, fp) = self.confusion_at(self.adaptive_threshold());
        let adaptive = if tp == 0 {
            0.0
        } else {
            let (tp, positives) = (count_f64(tp), count_f64(tp + fp));
            f_beta(tp / positives, tp / gt_count, beta_sq)
        };

        let counts = self.histogram().cumulative();
        let precision = ThresholdCurve::from_thresholds(|threshold| {
            let idx = 255 - usize::from(threshold);
            let positives = count_f64(counts.fg[idx] + counts.bg[idx]).max(1.0);
            count_f64(counts.fg[idx]) / positives
        });
        let recall = ThresholdCurve::from_thresholds(|threshold| {
            count_f64(counts.fg[255 - usize::from(threshold)]) / gt_count
        });
        let curve = ThresholdCurve::from_thresholds(|threshold| {
            f_beta(precision.at(threshold), recall.at(threshold), beta_sq)
        });

        FMeasure {
            adaptive,
            curve,
            precision,
            recall,
        }
    }

    /// Computes the adaptive E-measure and its per-threshold curve.
    #[must_use]
    pub fn e_measure(&self) -> EMeasure {
        let (tp, fp) = self.confusion_at(self.adaptive_threshold());
        let adaptive = self.enhanced_alignment(tp, fp);
        let counts = self.histogram().cumulative();
        let curve = ThresholdCurve::from_thresholds(|threshold| {
            let idx = 255 - usize::from(threshold);
            self.enhanced_alignment(counts.fg[idx], counts.bg[idx])
        });
        EMeasure { adaptive, curve }
    }

    /// Enhanced-alignment score for `true_pos` true and `false_pos` false positives.
    ///
    /// The `size - 1` normalizer is kept for parity with the reference implementation (a
    /// perfect `N`-pixel match scores `N / (N - 1)`); it is clamped to `1` for 1-pixel inputs.
    fn enhanced_alignment(&self, true_pos: u64, false_pos: u64) -> f64 {
        let size = usize_f64(self.len());
        let gt_fg = usize_f64(self.foreground_count());
        let positives = count_f64(true_pos + false_pos);
        let negatives = size - positives;

        let enhanced_sum = if self.foreground_count() == 0 {
            negatives
        } else if self.foreground_count() == self.len() {
            positives
        } else {
            let false_neg = gt_fg - count_f64(true_pos);
            let true_neg = negatives - false_neg;
            let (mean_pred, mean_gt) = (positives / size, gt_fg / size);
            [
                (count_f64(true_pos), 1.0 - mean_pred, 1.0 - mean_gt),
                (count_f64(false_pos), 1.0 - mean_pred, -mean_gt),
                (false_neg, -mean_pred, 1.0 - mean_gt),
                (true_neg, -mean_pred, -mean_gt),
            ]
            .into_iter()
            .map(|(numel, p, g)| {
                let align = 2.0 * p * g / (p.mul_add(p, g * g) + EPS);
                let enhanced = (align + 1.0) * (align + 1.0) / 4.0;
                enhanced * numel
            })
            .sum()
        };

        enhanced_sum / ((size - 1.0).max(1.0) + EPS)
    }

    /// Computes the structure measure `alpha * S_object + (1 - alpha) * S_region`.
    #[must_use]
    pub fn s_measure(&self, alpha: f64) -> f64 {
        let size = usize_f64(self.len());
        let y = usize_f64(self.foreground_count()) / size;
        let mean_pred = || {
            self.predictions()
                .iter()
                .copied()
                .fold(0.0, f64::algebraic_add)
                / size
        };
        if self.foreground_count() == 0 {
            1.0 - mean_pred()
        } else if self.foreground_count() == self.len() {
            mean_pred()
        } else {
            alpha
                .mul_add(
                    self.s_object_score(y),
                    (1.0 - alpha) * self.s_region_score(),
                )
                .max(0.0)
        }
    }

    fn s_object_score(&self, fg_ratio: f64) -> f64 {
        let pred = self.predictions();
        let gt = self.targets();
        // Foreground scores `pred` on `gt`; background scores `1 - pred` on `!gt`.
        let fg = s_object(pred.iter().zip(gt).filter(|&(_, &g)| g).map(|(&p, _)| p));
        let bg = s_object(
            pred.iter()
                .zip(gt)
                .filter(|&(_, &g)| !g)
                .map(|(&p, _)| 1.0 - p),
        );
        fg_ratio.mul_add(fg, (1.0 - fg_ratio) * bg)
    }

    fn s_region_score(&self) -> f64 {
        let [height, width] = self.dims();
        let (cx, cy) = self.centroid();
        let area = usize_f64(height * width);
        [
            (0..cy, 0..cx),
            (0..cy, cx..width),
            (cy..height, 0..cx),
            (cy..height, cx..width),
        ]
        .into_iter()
        .filter(|(rows, cols)| !rows.is_empty() && !cols.is_empty())
        .map(|(rows, cols)| {
            let weight = usize_f64(rows.len() * cols.len()) / area;
            weight * self.region_ssim(rows, cols)
        })
        .sum()
    }

    /// Foreground centroid as 1-based `(x, y)` split coordinates (MATLAB heritage).
    fn centroid(&self) -> (usize, usize) {
        let [height, width] = self.dims();
        let (mut sum_y, mut sum_x) = (0.0, 0.0);
        for (idx, _) in self.targets().iter().enumerate().filter(|&(_, &g)| g) {
            sum_y += usize_f64(idx / width);
            sum_x += usize_f64(idx % width);
        }
        let count = usize_f64(self.foreground_count());
        let round = |value: f64, limit: usize| {
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "a centroid lies inside the image, so the rounded value is in 0..limit"
            )]
            let idx = value.round_ties_even().max(0.0) as usize;
            (idx + 1).min(limit)
        };
        if self.foreground_count() == 0 {
            (
                round(usize_f64(width) / 2.0, width),
                round(usize_f64(height) / 2.0, height),
            )
        } else {
            (round(sum_x / count, width), round(sum_y / count, height))
        }
    }

    fn region_ssim(&self, rows: std::ops::Range<usize>, cols: std::ops::Range<usize>) -> f64 {
        let width = self.dims()[1];
        let n = usize_f64(rows.len() * cols.len());
        let (row_range, col_range) = ((rows.start, rows.end), (cols.start, cols.end));
        let pixels = || {
            (row_range.0..row_range.1).flat_map(move |row| {
                (col_range.0..col_range.1).map(move |col| {
                    let idx = row * width + col;
                    (
                        self.predictions()[idx],
                        f64::from(u8::from(self.targets()[idx])),
                    )
                })
            })
        };
        let (sum_p, sum_g) = pixels().fold((0.0, 0.0), |(sp, sg), (p, g)| (sp + p, sg + g));
        let (x, y) = (sum_p / n, sum_g / n);
        let (mut var_x, mut var_y, mut cov) = (0.0, 0.0, 0.0);
        for (p, g) in pixels() {
            var_x += (p - x) * (p - x);
            var_y += (g - y) * (g - y);
            cov += (p - x) * (g - y);
        }
        let denom = (n - 1.0).max(1.0);
        let (var_x, var_y, cov) = (var_x / denom, var_y / denom, cov / denom);

        let alpha = 4.0 * x * y * cov;
        let beta = x.mul_add(x, y * y) * (var_x + var_y);
        if alpha != 0.0 {
            alpha / (beta + EPS)
        } else if beta == 0.0 {
            1.0
        } else {
            0.0
        }
    }

    /// Computes the weighted F-measure (Margolin et al., 2014); `0` without foreground.
    ///
    /// `beta_sq` is `β²` (`1.0` in the reference evaluation).
    #[must_use]
    pub fn weighted_f_measure(&self, beta_sq: f64) -> f64 {
        if self.foreground_count() == 0 {
            return 0.0;
        }
        let dims @ [height, width] = self.dims();
        let pred = self.predictions();
        let gt = self.targets();
        let sites = nearest_sites(gt, dims);

        // Pixel dependency: background errors borrow the error of their nearest foreground pixel.
        let error: Vec<f64> = pred
            .iter()
            .zip(gt)
            .map(|(&p, &g)| (p - f64::from(u8::from(g))).abs())
            .collect();
        let dependent: Vec<f64> = (0..error.len())
            .map(|i| {
                if gt[i] {
                    error[i]
                } else {
                    error[sites.nearest[i]]
                }
            })
            .collect();
        let smoothed = gaussian_7x7_sigma5(&dependent, [height, width]);

        // Pixel importance decays with distance from the foreground.
        let decay = 0.5_f64.ln() / 5.0;
        let (mut fg_sum, mut bg_sum) = (0.0, 0.0);
        for i in 0..error.len() {
            if gt[i] {
                fg_sum += if smoothed[i] < error[i] {
                    smoothed[i]
                } else {
                    error[i]
                };
            } else {
                let importance = 2.0 - (decay * f64::from(sites.distances[i])).exp();
                bg_sum += error[i] * importance;
            }
        }

        let gt_count = usize_f64(self.foreground_count());
        let tpw = gt_count - fg_sum;
        let recall = 1.0 - fg_sum / gt_count;
        let precision = tpw / (tpw + bg_sum + EPS);
        (1.0 + beta_sq) * recall * precision / (recall + beta_sq * precision + EPS)
    }

    /// Computes the boundary IoU curve with boundaries of width
    /// `max(1, round(dilation_ratio * diagonal))`.
    #[must_use]
    pub fn biou(&self, dilation_ratio: f64) -> ThresholdCurve {
        let dims @ [height, width] = self.dims();
        let diagonal = usize_f64(height).hypot(usize_f64(width));
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "the radius is clamped to [1, diagonal], a small positive integer"
        )]
        let radius = (dilation_ratio * diagonal)
            .round_ties_even()
            .clamp(1.0, diagonal.max(1.0)) as usize;

        let pred: Vec<u8> = self.predictions().iter().map(|&p| quantize(p)).collect();
        let gt: Vec<u8> = self
            .targets()
            .iter()
            .map(|&g| if g { 255 } else { 0 })
            .collect();
        let pred_boundary = boundary(&pred, dims, radius);
        let gt_boundary: Vec<bool> = boundary(&gt, dims, radius)
            .into_iter()
            .map(|v| v > 128)
            .collect();

        let gt_total = count_f64(gt_boundary.iter().filter(|&&g| g).count() as u64).max(1.0);
        let counts =
            ThresholdHistogram::new(pred_boundary.into_iter().zip(gt_boundary)).cumulative();
        ThresholdCurve::from_thresholds(|threshold| {
            let idx = 255 - usize::from(threshold);
            count_f64(counts.fg[idx]) / (gt_total + count_f64(counts.bg[idx]))
        })
    }
}

/// `2 * mean / (mean² + 1 + std + eps)` of the given scores (sample std, `ddof = 1`).
fn s_object(values: impl Iterator<Item = f64> + Clone) -> f64 {
    let (count, sum) = values
        .clone()
        .fold((0_usize, 0.0), |(n, s), v| (n + 1, s + v));
    if count == 0 {
        return 0.0;
    }
    let n = usize_f64(count);
    let mean = sum / n;
    let var = values.map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0).max(1.0);
    2.0 * mean / (mean.mul_add(mean, 1.0) + var.sqrt() + EPS)
}

/// `value - erode(value)` where erosion is a `(2r + 1)²` minimum filter over a zero border,
/// i.e. `r` iterations of OpenCV's `3x3` erosion on an image padded by one zero pixel.
fn boundary(values: &[u8], [height, width]: [usize; 2], radius: usize) -> Vec<u8> {
    let rows = min_filter_lines(values, height, width, radius, |r, c| r * width + c);
    let eroded = min_filter_lines(&rows, width, height, radius, |c, r| r * width + c);
    values.iter().zip(&eroded).map(|(&v, &e)| v - e).collect()
}

/// Sliding minimum of radius `radius` along `lines` lines of length `len`; windows that leave
/// the image see the zero border. `index(line, pos)` maps to the flat buffer index.
fn min_filter_lines(
    values: &[u8],
    lines: usize,
    len: usize,
    radius: usize,
    index: impl Fn(usize, usize) -> usize,
) -> Vec<u8> {
    let mut out = vec![0_u8; values.len()];
    let mut window = std::collections::VecDeque::with_capacity(2 * radius + 1);
    for line in 0..lines {
        if len <= 2 * radius {
            continue; // Every window touches the zero border.
        }
        window.clear();
        for pos in 0..len {
            let value = values[index(line, pos)];
            while window.back().is_some_and(|&(_, v)| v >= value) {
                window.pop_back();
            }
            window.push_back((pos, value));
            if window
                .front()
                .is_some_and(|&(start, _)| start + 2 * radius < pos)
            {
                window.pop_front();
            }
            // Positions closer than `radius` to either edge stay 0.
            if pos >= 2 * radius {
                let center = pos - radius;
                out[index(line, center)] = window.front().map_or(0, |&(_, v)| v);
            }
        }
    }
    out
}

/// `fspecial('gaussian', 7, 5)` correlation with a zero border (`scipy.ndimage.convolve`,
/// `mode="constant"`), applied as two separable passes.
fn gaussian_7x7_sigma5(values: &[f64], [height, width]: [usize; 2]) -> Vec<f64> {
    const RADIUS: usize = 3;
    const SIGMA: f64 = 5.0;
    let taps: [f64; 2 * RADIUS + 1] = std::array::from_fn(|i| {
        let offset = usize_f64(i) - usize_f64(RADIUS);
        (-(offset * offset) / (2.0 * SIGMA * SIGMA)).exp()
    });
    let norm: f64 = taps.iter().sum();
    let taps = taps.map(|t| t / norm);

    let pass = |src: &[f64], lines: usize, len: usize, index: &dyn Fn(usize, usize) -> usize| {
        let mut out = vec![0.0; src.len()];
        for line in 0..lines {
            for pos in 0..len {
                let mut acc = 0.0;
                for (k, &tap) in taps.iter().enumerate() {
                    if let Some(src_pos) = (pos + k).checked_sub(RADIUS).filter(|&p| p < len) {
                        acc += tap * src[index(line, src_pos)];
                    }
                }
                out[index(line, pos)] = acc;
            }
        }
        out
    };
    let rows = pass(values, height, width, &|r, c| r * width + c);
    pass(&rows, width, height, &|c, r| r * width + c)
}
