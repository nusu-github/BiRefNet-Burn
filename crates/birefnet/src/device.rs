//! Runtime device and floating-point precision selection for `BiRefNet`.
//!
//! Burn selects the compute backend at run time from a [`Device`] value. Cargo features
//! decide which backends are compiled into a binary; [`select_device`] maps a user-facing
//! name such as `"default"`, `"flex"`, `"wgpu"`, or `"cuda:0"` to the corresponding device.

use std::{backtrace::Backtrace, fmt, str::FromStr};

use burn::tensor::{Device, FloatDType};

type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ErrorKind {
    InvalidIndex,
    UnsupportedDevice,
    PrecisionConfiguration,
}

#[derive(Debug)]
struct ErrorInner {
    kind: ErrorKind,
    message: String,
    source: Option<BoxError>,
    backtrace: Backtrace,
}

/// Error returned when parsing or configuring a runtime [`Device`] or [`Precision`].
pub struct DeviceError {
    inner: Box<ErrorInner>,
}

const _: () = {
    const fn assert_error_bounds<E: std::error::Error + Send + Sync + 'static>() {}
    assert_error_bounds::<DeviceError>();
};

impl DeviceError {
    fn new(kind: ErrorKind, message: impl Into<String>, source: Option<BoxError>) -> Self {
        Self {
            inner: Box::new(ErrorInner {
                kind,
                message: message.into(),
                source,
                backtrace: Backtrace::capture(),
            }),
        }
    }

    fn invalid_index(name: &str, source: std::num::ParseIntError) -> Self {
        Self::new(
            ErrorKind::InvalidIndex,
            format!("invalid device index in `{name}`: {source}"),
            Some(Box::new(source)),
        )
    }

    fn unsupported_device(name: &str) -> Self {
        Self::new(
            ErrorKind::UnsupportedDevice,
            format!(
                "unknown or disabled device `{name}`; available in this build: {}",
                AVAILABLE_DEVICES.join(", ")
            ),
            None,
        )
    }

    fn precision_configuration(message: impl Into<String>, source: Option<BoxError>) -> Self {
        Self::new(ErrorKind::PrecisionConfiguration, message, source)
    }

    /// Returns `true` if a device index suffix (such as `cuda:abc`) failed to parse.
    #[must_use]
    pub fn is_invalid_index(&self) -> bool {
        self.inner.kind == ErrorKind::InvalidIndex
    }

    /// Returns `true` if the named backend is unknown or disabled in this build.
    #[must_use]
    pub fn is_unsupported_device(&self) -> bool {
        self.inner.kind == ErrorKind::UnsupportedDevice
    }

    /// Returns `true` if configuring or parsing floating-point precision failed.
    #[must_use]
    pub fn is_precision_configuration(&self) -> bool {
        self.inner.kind == ErrorKind::PrecisionConfiguration
    }

    /// Returns the backtrace recorded when this error was constructed.
    #[must_use]
    pub fn backtrace(&self) -> &Backtrace {
        &self.inner.backtrace
    }
}

impl fmt::Debug for DeviceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceError")
            .field("kind", &self.inner.kind)
            .field("message", &self.inner.message)
            .field("source", &self.inner.source)
            .finish_non_exhaustive()
    }
}

impl fmt::Display for DeviceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.inner.message)
    }
}

impl std::error::Error for DeviceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        let err = self.inner.source.as_deref()?;
        Some(err)
    }
}

/// Device names accepted by [`select_device`] in the current build, in display order.
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

/// Maps a device name (for example a CLI `--device` flag) to a Burn [`Device`].
///
/// `default` defers to [`Device::default`], picking the first compiled-in backend in
/// Burn's priority order (`CUDA`, `Metal`, `ROCm`, `Vulkan`, `wgpu`, `CPU`, `Flex`) unless
/// `BURN_DEVICE` overrides it. GPU indices are given as `cuda:1` or `rocm:1`.
///
/// # Errors
///
/// Returns [`DeviceError`] if the backend name is unknown or not enabled in this build,
/// or if an explicit `:N` index is invalid.
pub fn select_device(name: &str) -> Result<Device, DeviceError> {
    let (backend, index) = match name.split_once(':') {
        Some((backend, index_str)) => {
            let index: usize = index_str
                .parse()
                .map_err(|err| DeviceError::invalid_index(name, err))?;
            (backend, Some(index))
        }
        None => (name, None),
    };

    match (backend, index) {
        ("default", None) => Ok(Device::default()),
        #[cfg(feature = "flex")]
        ("flex", None) => Ok(Device::flex()),
        #[cfg(feature = "cpu")]
        ("cpu", None) => Ok(Device::cpu()),
        #[cfg(feature = "wgpu")]
        ("wgpu", None) => Ok(Device::wgpu(burn::tensor::DeviceKind::DefaultDevice)),
        #[cfg(feature = "wgpu")]
        ("wgpu-cpu", None) => Ok(Device::wgpu(burn::tensor::DeviceKind::Cpu)),
        #[cfg(feature = "vulkan")]
        ("vulkan", None) => Ok(Device::vulkan(burn::tensor::DeviceKind::DefaultDevice)),
        #[cfg(feature = "metal")]
        ("metal", None) => Ok(Device::metal(burn::tensor::DeviceKind::DefaultDevice)),
        #[cfg(feature = "cuda")]
        ("cuda", index) => Ok(Device::cuda(index.unwrap_or(0))),
        #[cfg(feature = "rocm")]
        ("rocm", index) => Ok(Device::rocm(index.unwrap_or(0))),
        _ => Err(DeviceError::unsupported_device(name)),
    }
}

/// Floating-point precision used for tensors created on a selected [`Device`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Precision {
    /// IEEE 32-bit floating point (default across all backends).
    #[default]
    F32,
    /// IEEE 16-bit half precision.
    F16,
    /// Brain floating point (`bfloat16`).
    Bf16,
}

impl fmt::Display for Precision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
        })
    }
}

impl FromStr for Precision {
    type Err = DeviceError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "f32" => Ok(Self::F32),
            "f16" => Ok(Self::F16),
            "bf16" => Ok(Self::Bf16),
            other => Err(DeviceError::precision_configuration(
                format!("unsupported precision `{other}`; expected `f32`, `f16`, or `bf16`"),
                None,
            )),
        }
    }
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

/// Sets the default floating-point dtype on `device`.
///
/// Must be called before the first tensor allocation on `device`, as settings lock on first use.
///
/// # Errors
///
/// Returns [`DeviceError`] if the backend does not support `precision` or is already initialized.
pub fn configure_precision(device: &mut Device, precision: Precision) -> Result<(), DeviceError> {
    device
        .configure(FloatDType::from(precision))
        .map_err(|err| {
            DeviceError::precision_configuration(
                format!("cannot configure precision `{precision}` on {device:?}: {err}"),
                None,
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_device_lists_available_ones() {
        let err = select_device("tpu").unwrap_err();
        assert!(err.is_unsupported_device());
        let msg = err.to_string();
        assert!(msg.contains("unknown or disabled device `tpu`"), "{msg}");
        assert!(msg.contains("default"), "{msg}");
    }

    #[test]
    fn invalid_index_is_rejected() {
        let bad_idx = select_device("cuda:x").unwrap_err();
        assert!(bad_idx.is_invalid_index());

        let bad_default = select_device("default:0").unwrap_err();
        assert!(bad_default.is_unsupported_device());
    }

    #[test]
    fn precision_round_trips_display_and_from_str() {
        for precision in [Precision::F32, Precision::F16, Precision::Bf16] {
            let parsed: Precision = precision.to_string().to_uppercase().parse().unwrap();
            assert_eq!(parsed, precision);
        }
        let err = "fp64".parse::<Precision>().unwrap_err();
        assert!(err.is_precision_configuration());
    }

    #[cfg(feature = "flex")]
    #[test]
    fn flex_is_selectable() {
        assert_eq!(select_device("flex").unwrap(), Device::flex());
    }
}
