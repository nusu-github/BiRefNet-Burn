//! Backend selection utilities for BiRefNet
//!
//! This module provides a centralized way to handle backend selection
//! based on feature flags for the main BiRefNet crate.
//! https://github.com/tracel-ai/burn-lm/blob/main/crates/burn-lm-inference/src/backends.rs

mod elems {
    cfg_if::cfg_if! {
        // NOTE: f16/bf16 is not always supported on wgpu depending on the hardware
        // https://github.com/gfx-rs/wgpu/issues/7468
        if #[cfg(all(feature = "f16", any(feature = "cuda", feature = "wgpu", feature = "vulkan", feature = "metal", feature = "rocm")))]{
            pub type ElemType = burn::tensor::f16;
            pub const DTYPE_NAME: &str = "f16";
        }
        else if #[cfg(all(feature = "f16", any(feature = "cuda", feature = "wgpu", feature = "vulkan", feature = "metal", feature = "rocm")))]{
            pub type ElemType = burn::tensor::bf16;
            pub const DTYPE_NAME: &str = "bf16";
        } else {
            pub type ElemType = f32;
            pub const DTYPE_NAME: &str = "f32";
        }
    }
}

pub use elems::*;

// Cuda ----------------------------------------------------------------------

#[cfg(feature = "cuda")]
pub mod burn_backend_types {
    use burn::backend::cuda::{Cuda, CudaDevice};

    use super::*;
    pub type InferenceBackend = Cuda<ElemType>;
    pub type InferenceDevice = CudaDevice;
    pub const NAME: &str = "cuda";
}

// ROCm ----------------------------------------------------------------------

#[cfg(feature = "rocm")]
pub mod burn_backend_types {
    use burn::backend::rocm::{Rocm, RocmDevice};

    use super::*;
    pub type InferenceBackend = Rocm<ElemType>;
    pub type InferenceDevice = RocmDevice;
    pub const NAME: &str = "rocm";
}

// ndarray -------------------------------------------------------------------
// This backend is used for testing and by default when no backend is selected.

#[cfg(feature = "ndarray")]
pub mod burn_backend_types {
    use burn::backend::ndarray::{NdArray, NdArrayDevice};

    use super::*;

    pub type InferenceBackend = NdArray<ElemType>;
    pub type InferenceDevice = NdArrayDevice;
    pub const NAME: &str = "ndarray";
}

// WebGPU --------------------------------------------------------------------

#[cfg(any(feature = "wgpu", feature = "vulkan", feature = "metal"))]
pub mod burn_backend_types {
    use burn::backend::wgpu::{Wgpu, WgpuDevice};

    use super::*;
    pub type InferenceBackend = Wgpu<ElemType>;
    pub type InferenceDevice = WgpuDevice;
    #[cfg(all(feature = "wgpu", not(feature = "vulkan"), not(feature = "metal")))]
    pub const NAME: &str = "wgpu";
    #[cfg(feature = "vulkan")]
    pub const NAME: &str = "vulkan";
    #[cfg(feature = "metal")]
    pub const NAME: &str = "metal";
}

#[cfg(feature = "wgpu-cpu")]
pub mod burn_backend_types {
    use burn::backend::wgpu::{Wgpu, WgpuDevice};

    use super::*;
    pub type InferenceBackend = Wgpu<ElemType>;
    pub type InferenceDevice = WgpuDevice;
    pub const NAME: &str = "wgpu-cpu";
}
