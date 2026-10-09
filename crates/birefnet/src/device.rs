//! Runtime device selection for `BiRefNet`.
//!
//! Burn chooses the backend at run time from a [`Device`] value. Cargo features only decide
//! which backends are compiled into this binary; [`select_device`] maps a user-facing name to
//! the matching constructor.

use anyhow::{Context, Result, bail};
use burn::tensor::{Device, FloatDType};

/// Names accepted by [`select_device`] in this build, in the order shown to users.
pub const AVAILABLE_DEVICES: &[&str] = &[
    "default",
    #[cfg(feature = "flex")]
    "flex",
    #[cfg(feature = "cpu")]
    "cpu",
    #[cfg(feature = "wgpu")]
    "wgpu",
    #[cfg(feature = "wgpu")]
    "wgpu-cpu",
    #[cfg(feature = "vulkan")]
    "vulkan",
    #[cfg(feature = "metal")]
    "metal",
    #[cfg(feature = "cuda")]
    "cuda[:N]",
    #[cfg(feature = "rocm")]
    "rocm[:N]",
];

/// Maps a device name (for example a `--device` flag) to an explicit device.
///
/// `default` defers to [`Device::default`], which picks the first compiled-in backend in
/// Burn's fixed order (CUDA, Metal, ROCm, Vulkan, wgpu, CPU, Flex) unless the `BURN_DEVICE`
/// environment variable overrides it. GPU indices are given as `cuda:1` or `rocm:1`.
///
/// Constructing a device does not touch the hardware; an invalid index fails at the first
/// tensor operation.
///
/// # Errors
///
/// Returns an error if the name is unknown or its backend is not compiled into this binary.
pub fn select_device(name: &str) -> Result<Device> {
    let (backend, index) = match name.split_once(':') {
        Some((backend, index)) => {
            let index: usize = index
                .parse()
                .with_context(|| format!("invalid device index in '{name}'"))?;
            (backend, Some(index))
        }
        None => (name, None),
    };

    let device = match (backend, index) {
        ("default", None) => Device::default(),
        #[cfg(feature = "flex")]
        ("flex", None) => Device::flex(),
        #[cfg(feature = "cpu")]
        ("cpu", None) => Device::cpu(),
        #[cfg(feature = "wgpu")]
        ("wgpu", None) => Device::wgpu(burn::tensor::DeviceKind::DefaultDevice),
        #[cfg(feature = "wgpu")]
        ("wgpu-cpu", None) => Device::wgpu(burn::tensor::DeviceKind::Cpu),
        #[cfg(feature = "vulkan")]
        ("vulkan", None) => Device::vulkan(burn::tensor::DeviceKind::DefaultDevice),
        #[cfg(feature = "metal")]
        ("metal", None) => Device::metal(burn::tensor::DeviceKind::DefaultDevice),
        #[cfg(feature = "cuda")]
        ("cuda", index) => Device::cuda(index.unwrap_or(0)),
        #[cfg(feature = "rocm")]
        ("rocm", index) => Device::rocm(index.unwrap_or(0)),
        _ => bail!(
            "unknown or disabled device '{name}'; available in this build: {}",
            AVAILABLE_DEVICES.join(", ")
        ),
    };

    Ok(device)
}

/// Floating-point precision used for every tensor on the selected device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Precision {
    /// 32-bit floats (the default of every backend).
    F32,
    /// IEEE half precision.
    F16,
    /// bfloat16.
    Bf16,
}

impl From<Precision> for FloatDType {
    fn from(precision: Precision) -> Self {
        match precision {
            Precision::F32 => Self::F32,
            Precision::F16 => Self::F16,
            Precision::Bf16 => Self::BF16,
        }
    }
}

/// Sets the default float dtype of `device`.
///
/// Must run before the first tensor is created on the device (the settings lock on first use).
/// Half precision is not available on every backend or adapter.
///
/// # Errors
///
/// Returns an error if the backend does not support the dtype or the device is already in use.
pub fn configure_precision(device: &mut Device, precision: Precision) -> Result<()> {
    device
        .configure(FloatDType::from(precision))
        .with_context(|| format!("cannot use {precision:?} on {device:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_device_lists_available_ones() {
        let err = select_device("tpu").unwrap_err().to_string();

        assert!(err.contains("unknown or disabled device 'tpu'"), "{err}");
        assert!(err.contains("default"), "{err}");
    }

    #[test]
    fn invalid_index_is_rejected() {
        assert!(select_device("cuda:x").is_err());
        assert!(select_device("default:0").is_err());
    }

    #[cfg(feature = "flex")]
    #[test]
    fn flex_is_selectable() {
        assert_eq!(select_device("flex").unwrap(), Device::flex());
    }
}
