//! Backend selection utilities for BiRefNet
//!
//! This module provides a centralized way to handle backend selection
//! based on feature flags for the main BiRefNet crate.

use cfg_if::cfg_if;

cfg_if! {
    if #[cfg(feature = "cuda")] {
        use burn::backend::cuda::{Cuda, CudaDevice};

        /// Selected backend type
        pub type SelectedBackend = Cuda;
        /// Selected device type
        pub type SelectedDevice = CudaDevice;

        /// Creates the appropriate device for the selected backend
        pub fn create_device() -> SelectedDevice {
            CudaDevice::default()
        }

        /// Gets the backend name for logging purposes
        pub const fn get_backend_name() -> &'static str {
            "CUDA (NVIDIA GPU)"
        }
    } else if #[cfg(feature = "wgpu")] {
        use burn::backend::wgpu::{Wgpu, WgpuDevice};

        /// Selected backend type
        pub type SelectedBackend = Wgpu;
        /// Selected device type
        pub type SelectedDevice = WgpuDevice;

        /// Creates the appropriate device for the selected backend
        pub fn create_device() -> SelectedDevice {
            WgpuDevice::default()
        }

        /// Gets the backend name for logging purposes
        pub const fn get_backend_name() -> &'static str {
            "WGPU (GPU)"
        }
    } else {
        // Default to ndarray backend
        use burn::backend::ndarray::{NdArray, NdArrayDevice};

        /// Selected backend type
        pub type SelectedBackend = NdArray;
        /// Selected device type
        pub type SelectedDevice = NdArrayDevice;

        /// Creates the appropriate device for the selected backend
         pub fn create_device() -> SelectedDevice {
            NdArrayDevice::default()
        }

        /// Gets the backend name for logging purposes
         pub const fn get_backend_name() -> &'static str {
            "NdArray (CPU)"
        }
    }
}
