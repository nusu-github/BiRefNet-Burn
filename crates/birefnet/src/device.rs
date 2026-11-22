//! Runtime device and floating-point precision selection.
//!
//! Burn selects the compute backend at run time from a [`Device`] value. Cargo features decide
//! which backends are compiled in; [`select_device`] maps a name such as `"default"`, `"flex"`,
//! `"wgpu"`, or `"cuda:1"` to a device of this build.

use std::{
    backtrace::{Backtrace, BacktraceStatus},
    error::Error,
    fmt,
    str::FromStr,
};

use burn::tensor::{Device, FloatDType};

/// Error returned when parsing a device or precision name, or configuring a device.
#[derive(Debug)]
pub struct DeviceError {
    inner: Box<ErrorInner>,
}

#[derive(Debug)]
struct ErrorInner {
    kind: ErrorKind,
    backtrace: Backtrace,
}

#[derive(Debug)]
enum ErrorKind {
    UnsupportedDevice {
        name: String,
    },
    InvalidIndex {
        name: String,
        source: std::num::ParseIntError,
    },
    #[cfg_attr(
        not(any(feature = "cuda", feature = "rocm")),
        expect(dead_code, reason = "only indexed GPU backends construct it")
    )]
    IndexOutOfRange {
        name: String,
        available: usize,
    },
    #[cfg_attr(
        any(
            feature = "flex",
            feature = "cpu",
            feature = "wgpu",
            feature = "cuda",
            feature = "rocm"
        ),
        expect(dead_code, reason = "only builds without any backend construct it")
    )]
    NoBackend,
    InvalidPrecision {
        value: String,
    },
    Configuration(burn::tensor::DeviceError),
}

impl DeviceError {
    fn from_kind(kind: ErrorKind) -> Self {
        Self {
            inner: Box::new(ErrorInner {
                kind,
                backtrace: Backtrace::capture(),
            }),
        }
    }

    /// Returns `true` if the backend name is unknown or not compiled into this build.
    #[must_use]
    pub fn is_unsupported_device(&self) -> bool {
        matches!(self.inner.kind, ErrorKind::UnsupportedDevice { .. })
    }

    /// Returns `true` if a device index suffix (such as `cuda:abc`) failed to parse.
    #[must_use]
    pub fn is_invalid_index(&self) -> bool {
        matches!(self.inner.kind, ErrorKind::InvalidIndex { .. })
    }

    /// Returns `true` if a GPU index exceeds the number of devices present.
    #[must_use]
    pub fn is_index_out_of_range(&self) -> bool {
        matches!(self.inner.kind, ErrorKind::IndexOutOfRange { .. })
    }

    /// Returns `true` if `default` was requested in a build without any backend.
    #[must_use]
    pub fn is_no_backend(&self) -> bool {
        matches!(self.inner.kind, ErrorKind::NoBackend)
    }

    /// Returns `true` if a precision name (such as `fp64`) is not recognized.
    #[must_use]
    pub fn is_invalid_precision(&self) -> bool {
        matches!(self.inner.kind, ErrorKind::InvalidPrecision { .. })
    }

    /// Returns `true` if the device rejected a configuration (unsupported dtype, or settings
    /// already locked by an earlier allocation).
    #[must_use]
    pub fn is_configuration(&self) -> bool {
        matches!(self.inner.kind, ErrorKind::Configuration(_))
    }

    /// Returns the captured [`Backtrace`] for this error.
    #[must_use]
    pub fn backtrace(&self) -> &Backtrace {
        &self.inner.backtrace
    }
}

impl From<burn::tensor::DeviceError> for DeviceError {
    fn from(source: burn::tensor::DeviceError) -> Self {
        Self::from_kind(ErrorKind::Configuration(source))
    }
}

impl fmt::Display for DeviceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.inner.kind {
            ErrorKind::UnsupportedDevice { name } => write!(
                f,
                "unknown or disabled device `{name}`; available in this build: {}",
                AVAILABLE_DEVICES.join(", ")
            ),
            ErrorKind::InvalidIndex { name, .. } => write!(f, "invalid device index in `{name}`"),
            ErrorKind::IndexOutOfRange { name, available } => write!(
                f,
                "device `{name}` does not exist; {available} such device(s) are present"
            ),
            ErrorKind::NoBackend => f.write_str(
                "no backend is compiled into this build; enable a backend feature such as `flex`",
            ),
            ErrorKind::InvalidPrecision { value } => write!(
                f,
                "unsupported precision `{value}`; expected `f32`, `f16`, or `bf16`"
            ),
            ErrorKind::Configuration(_) => f.write_str("failed to configure the device"),
        }?;
        if self.inner.backtrace.status() == BacktraceStatus::Captured {
            write!(f, "\n\nstack backtrace:\n{}", self.inner.backtrace)?;
        }
        Ok(())
    }
}

impl Error for DeviceError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match &self.inner.kind {
            ErrorKind::InvalidIndex { source, .. } => Some(source),
            ErrorKind::Configuration(source) => Some(source),
            _ => None,
        }
    }
}

const _: fn() = || {
    fn assert_send_sync_static<T: Send + Sync + 'static>() {}
    assert_send_sync_static::<DeviceError>();
};

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

/// Returns `Device::default()`, or an error when no backend is compiled in.
fn default_device() -> Result<Device, DeviceError> {
    cfg_select! {
        any(
            feature = "flex",
            feature = "cpu",
            feature = "wgpu",
            feature = "cuda",
            feature = "rocm"
        ) => Ok(Device::default()),
        _ => Err(DeviceError::from_kind(ErrorKind::NoBackend)),
    }
}

/// Validates `index` against the enumerated devices before creating the device, so a
/// missing GPU is an error here rather than a panic on a backend worker thread.
#[cfg(any(feature = "cuda", feature = "rocm"))]
fn indexed_device(
    name: &str,
    index: usize,
    kind: burn::tensor::DeviceType,
    make: fn(usize) -> Device,
) -> Result<Device, DeviceError> {
    let available = Device::enumerate(kind).into_vec().len();
    if index < available {
        Ok(make(index))
    } else {
        Err(DeviceError::from_kind(ErrorKind::IndexOutOfRange {
            name: name.to_owned(),
            available,
        }))
    }
}

/// Maps a device name (for example a CLI `--device` flag) to a Burn [`Device`].
///
/// `default` defers to [`Device::default`], which honors `BURN_DEVICE` and otherwise picks the
/// first compiled-in backend in Burn's priority order (CUDA, Metal, ROCm, Vulkan, wgpu, CPU,
/// Flex). GPU indices are given as `cuda:1` or `rocm:1` and are checked against the devices
/// present.
///
/// # Errors
///
/// Returns [`DeviceError`] if the backend name is unknown or not enabled in this build, an
/// index suffix is malformed or out of range, or no backend is compiled in.
pub fn select_device(name: impl AsRef<str>) -> Result<Device, DeviceError> {
    let name = name.as_ref();
    let (backend, index) = match name.split_once(':') {
        Some((backend, index)) => {
            let index = index.parse::<usize>().map_err(|source| {
                DeviceError::from_kind(ErrorKind::InvalidIndex {
                    name: name.to_owned(),
                    source,
                })
            })?;
            (backend, Some(index))
        }
        None => (name, None),
    };

    match (backend, index) {
        ("default", None) => default_device(),
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
        ("cuda", index) => indexed_device(
            name,
            index.unwrap_or(0),
            burn::tensor::DeviceType::Cuda,
            Device::cuda,
        ),
        #[cfg(feature = "rocm")]
        ("rocm", index) => indexed_device(
            name,
            index.unwrap_or(0),
            burn::tensor::DeviceType::Rocm,
            Device::rocm,
        ),
        _ => Err(DeviceError::from_kind(ErrorKind::UnsupportedDevice {
            name: name.to_owned(),
        })),
    }
}

/// Returns a device on the first physical GPU this build can reach, if any (CubeCL GPU
/// backends only).
///
/// Unlike `default`, this skips software rasterizers (such as `llvmpipe` under Vulkan) and
/// CPU backends, which report no physical card.
#[cfg(any(feature = "wgpu", feature = "cuda", feature = "rocm"))]
#[must_use]
pub fn first_hardware_gpu() -> Option<Device> {
    Device::enumerate_physical()
        .into_iter()
        .find_map(|gpu| gpu.devices.into_vec().into_iter().next())
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
            _ => Err(DeviceError::from_kind(ErrorKind::InvalidPrecision {
                value: s.to_owned(),
            })),
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
/// Returns [`DeviceError`] (`is_configuration`) if the backend does not support `precision` or
/// its settings are already locked.
pub fn configure_precision(device: &mut Device, precision: Precision) -> Result<(), DeviceError> {
    Ok(device.configure(FloatDType::from(precision))?)
}
