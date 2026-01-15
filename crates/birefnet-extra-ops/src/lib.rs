//! Additional operations for the Burn deep learning framework
//!
//! This crate provides operations that are commonly used in deep learning but are not
//! yet available in the core Burn framework.

mod drop_path;
mod erfinv;
mod identity;
mod trunc_normal;

// Convenient re-exports
pub use drop_path::{DropPath, DropPathConfig};
pub use erfinv::{Erfinv, erfinv};
pub use identity::Identity;
pub use trunc_normal::{trunc_normal, trunc_normal_};

#[cfg(test)]
mod tests {
    use burn::backend::Cpu;

    pub type TestBackend = Cpu;
}
