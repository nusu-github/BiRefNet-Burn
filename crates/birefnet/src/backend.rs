//! Backend selection utilities for `BiRefNet`.
//!
//! Burn 0.22 chooses the backend at run time through a [`Device`] value. The Cargo features of this
//! crate decide which device constructor is compiled in; the top-level binary picks one here and
//! hands the device to everything else.

use burn::tensor::{Device, DeviceError, FloatDType};

/// Device type used by inference and training.
pub type InferenceDevice = Device;

/// Name of the DType the device is configured with.
#[cfg(feature = "f16")]
pub const DTYPE_NAME: &str = "f16";
/// Name of the DType the device is configured with.
#[cfg(all(feature = "bf16", not(feature = "f16")))]
pub const DTYPE_NAME: &str = "bf16";
/// Name of the DType the device is configured with.
#[cfg(not(any(feature = "f16", feature = "bf16")))]
pub const DTYPE_NAME: &str = "f32";

/// Backend selected at compile time (highest priority feature wins).
pub mod burn_backend_types {
    use super::{Device, DeviceError, FloatDType};

    #[cfg(feature = "cuda")]
    pub const NAME: &str = "cuda";
    #[cfg(all(feature = "rocm", not(feature = "cuda")))]
    pub const NAME: &str = "rocm";
    #[cfg(all(feature = "metal", not(any(feature = "cuda", feature = "rocm"))))]
    pub const NAME: &str = "metal";
    #[cfg(all(
        feature = "vulkan",
        not(any(feature = "cuda", feature = "rocm", feature = "metal"))
    ))]
    pub const NAME: &str = "vulkan";
    #[cfg(all(
        feature = "wgpu",
        not(any(
            feature = "cuda",
            feature = "rocm",
            feature = "metal",
            feature = "vulkan"
        ))
    ))]
    pub const NAME: &str = "wgpu";
    #[cfg(all(
        feature = "flex",
        not(any(
            feature = "cuda",
            feature = "rocm",
            feature = "metal",
            feature = "vulkan",
            feature = "wgpu"
        ))
    ))]
    pub const NAME: &str = "flex";
    #[cfg(all(
        feature = "cpu",
        not(any(
            feature = "cuda",
            feature = "rocm",
            feature = "metal",
            feature = "vulkan",
            feature = "wgpu",
            feature = "flex"
        ))
    ))]
    pub const NAME: &str = "cpu";

    pub use super::InferenceDevice;

    /// Creates the device of the compiled-in backend and applies the configured float precision.
    ///
    /// Burn's `Device::default()` is avoided on purpose: it depends on feature unification
    /// across the whole dependency graph.
    ///
    /// # Panics
    ///
    /// Panics if no backend feature is enabled.
    #[must_use]
    pub fn default_device() -> InferenceDevice {
        let mut device = raw_device();
        // Defaults lock on the first tensor created on the device, so configure first.
        if let Err(error) = configure_dtype(&mut device) {
            tracing::warn!(%error, "could not apply the requested float dtype; using device defaults");
        }
        device
    }

    #[cfg(feature = "cuda")]
    fn raw_device() -> Device {
        Device::cuda(0)
    }
    #[cfg(all(feature = "rocm", not(feature = "cuda")))]
    fn raw_device() -> Device {
        Device::rocm(0)
    }
    #[cfg(all(feature = "metal", not(any(feature = "cuda", feature = "rocm"))))]
    fn raw_device() -> Device {
        Device::metal(burn::tensor::DeviceKind::DefaultDevice)
    }
    #[cfg(all(
        feature = "vulkan",
        not(any(feature = "cuda", feature = "rocm", feature = "metal"))
    ))]
    fn raw_device() -> Device {
        Device::vulkan(burn::tensor::DeviceKind::DefaultDevice)
    }
    #[cfg(all(
        feature = "wgpu",
        not(any(
            feature = "cuda",
            feature = "rocm",
            feature = "metal",
            feature = "vulkan"
        ))
    ))]
    fn raw_device() -> Device {
        Device::wgpu(burn::tensor::DeviceKind::DefaultDevice)
    }
    #[cfg(all(
        feature = "flex",
        not(any(
            feature = "cuda",
            feature = "rocm",
            feature = "metal",
            feature = "vulkan",
            feature = "wgpu"
        ))
    ))]
    fn raw_device() -> Device {
        Device::flex()
    }
    #[cfg(all(
        feature = "cpu",
        not(any(
            feature = "cuda",
            feature = "rocm",
            feature = "metal",
            feature = "vulkan",
            feature = "wgpu",
            feature = "flex"
        ))
    ))]
    fn raw_device() -> Device {
        Device::cpu()
    }
    #[cfg(not(any(
        feature = "cuda",
        feature = "rocm",
        feature = "metal",
        feature = "vulkan",
        feature = "wgpu",
        feature = "flex",
        feature = "cpu"
    )))]
    fn raw_device() -> Device {
        panic!(
            "no Burn backend feature enabled; enable one of: flex, cpu, wgpu, vulkan, metal, cuda, rocm"
        )
    }

    #[cfg(feature = "f16")]
    fn configure_dtype(device: &mut Device) -> Result<(), DeviceError> {
        // NOTE: f16 is not supported on every wgpu adapter.
        device.configure(FloatDType::F16)
    }
    #[cfg(all(feature = "bf16", not(feature = "f16")))]
    fn configure_dtype(device: &mut Device) -> Result<(), DeviceError> {
        device.configure(FloatDType::BF16)
    }
    #[cfg(not(any(feature = "f16", feature = "bf16")))]
    fn configure_dtype(_device: &mut Device) -> Result<(), DeviceError> {
        let _ = FloatDType::F32;
        Ok(())
    }
}

pub use burn_backend_types::default_device;
