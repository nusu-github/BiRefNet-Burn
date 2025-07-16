//! # Special-Purpose Modules and Functions
//!
//! This module provides a collection of specialized, often low-level, modules and functions
//! that are used throughout the BiRefNet implementation. These include custom tensor
//! operations and regularization techniques that are not part of the standard Burn library.
//!
//! ## Modules
//!
//! - `drop_path`: Implements DropPath, a form of stochastic depth regularization.
//! - `erfinv`: Provides an implementation of the inverse error function.
//! - `identity`: A simple identity module.
//! - `roll`: A function to cyclically shift a tensor along given dimensions.
//! - `slice`: A helper for creating tensor slices with negative indexing support.
//! - `trunc_normal`: A function for initializing tensors with values from a truncated
//!   normal distribution.

mod drop_path;
mod erfinv;
mod identity;
mod roll;
mod slice;
mod trunc_normal;

pub use drop_path::*;
pub use identity::*;
pub use roll::*;
pub use slice::*;
pub use trunc_normal::*;
