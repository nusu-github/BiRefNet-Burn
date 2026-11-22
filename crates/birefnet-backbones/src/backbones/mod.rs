pub mod pvt_v2;
pub mod resnet;
pub mod swin;
pub mod vgg;

use crate::error::BackboneError;

pub(crate) fn usize_to_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).unwrap_or_else(|_| {
        panic!("dimension {value} exceeds u32::MAX and cannot be represented exactly in f64")
    }))
}

pub(crate) fn usize_to_i64(value: usize) -> i64 {
    i64::try_from(value).unwrap_or_else(|_| panic!("index {value} exceeds i64::MAX"))
}

/// `steps` evenly spaced values from `start` to `end` inclusive (`torch.linspace` semantics).
pub(crate) fn linspace(start: f64, end: f64, steps: usize) -> Vec<f64> {
    if steps <= 1 {
        return vec![start; steps];
    }
    let step_size = (end - start) / usize_to_f64(steps - 1);
    Vec::from_fn(steps, |i| usize_to_f64(i).mul_add(step_size, start))
}

/// Hidden width of a transformer MLP, `int(dim * mlp_ratio)` as in the upstream models.
pub(crate) fn scaled_hidden_dim(dim: usize, mlp_ratio: f64) -> usize {
    let hidden = (usize_to_f64(dim) * mlp_ratio).trunc();
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "validate_mlp_ratio guarantees a finite positive ratio, so `hidden` is a small \
                  non-negative integer"
    )]
    let hidden = hidden as usize;
    hidden
}

/// Rejects dropout-style probabilities outside `[0, 1)`.
pub(crate) fn validate_rate(name: &str, value: f64) -> Result<(), BackboneError> {
    if (0.0..1.0).contains(&value) {
        Ok(())
    } else {
        Err(BackboneError::invalid_config(format!(
            "{name} must be in [0, 1), got {value}"
        )))
    }
}

/// Rejects MLP expansion ratios that are not finite and strictly positive.
pub(crate) fn validate_mlp_ratio(value: f64) -> Result<(), BackboneError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(BackboneError::invalid_config(format!(
            "mlp_ratio must be finite and positive, got {value}"
        )))
    }
}
