//! Additional tensor operations and layers for the Burn deep learning framework.

mod drop_path;
mod erfinv;
mod trunc_normal;

#[doc(inline)]
pub use drop_path::{DropPath, DropPathConfig};
#[doc(inline)]
pub use erfinv::{Erfinv, erfinv};
#[doc(inline)]
pub use trunc_normal::trunc_normal;
