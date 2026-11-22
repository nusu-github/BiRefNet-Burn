//! Host-side per-component loss values for logging.

use std::fmt;

use burn::tensor::{Tensor, TensorReadError, Transaction};

use crate::error::LossError;

/// Weighted loss contributions read back to the host, each summed across prediction scales.
///
/// Every component already includes its configured weight and global scale, so the components
/// add up to [`total`](Self::total). Disabled components are `0.0`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LossBreakdown {
    /// Binary cross-entropy with logits.
    pub bce: f64,
    /// Soft IoU loss.
    pub iou: f64,
    /// Patch-wise IoU loss.
    pub iou_patch: f64,
    /// Mean absolute error.
    pub mae: f64,
    /// Mean squared error.
    pub mse: f64,
    /// Threshold regularization.
    pub reg: f64,
    /// Structural similarity loss.
    pub ssim: f64,
    /// Boundary contour loss.
    pub cnt: f64,
    /// Structure (weighted BCE + weighted IoU) loss.
    pub structure: f64,
    /// Auxiliary classification loss.
    pub classification: f64,
    /// Sum of all components.
    pub total: f64,
}

impl LossBreakdown {
    /// Returns `(name, value)` pairs for every component, excluding [`total`](Self::total).
    #[must_use]
    pub const fn components(&self) -> [(&'static str, f64); 10] {
        [
            ("bce", self.bce),
            ("iou", self.iou),
            ("iou_patch", self.iou_patch),
            ("mae", self.mae),
            ("mse", self.mse),
            ("reg", self.reg),
            ("ssim", self.ssim),
            ("cnt", self.cnt),
            ("structure", self.structure),
            ("classification", self.classification),
        ]
    }

    fn slot_mut(&mut self, component: Component) -> &mut f64 {
        match component {
            Component::Bce => &mut self.bce,
            Component::Iou => &mut self.iou,
            Component::IouPatch => &mut self.iou_patch,
            Component::Mae => &mut self.mae,
            Component::Mse => &mut self.mse,
            Component::Reg => &mut self.reg,
            Component::Ssim => &mut self.ssim,
            Component::Cnt => &mut self.cnt,
            Component::Structure => &mut self.structure,
            Component::Classification => &mut self.classification,
        }
    }
}

impl fmt::Display for LossBreakdown {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "total={:.6}", self.total)?;
        for (name, value) in self.components() {
            if value != 0.0 {
                write!(f, " {name}={value:.6}")?;
            }
        }
        Ok(())
    }
}

/// Loss component identifiers used while accumulating device tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Component {
    Bce,
    Iou,
    IouPatch,
    Mae,
    Mse,
    Reg,
    Ssim,
    Cnt,
    Structure,
    Classification,
}

/// Reads `total` and every `(component, value)` tensor in a single device transaction.
pub(crate) fn read_breakdown(
    total: Tensor<1>,
    components: Vec<(Component, Tensor<1>)>,
) -> Result<LossBreakdown, LossError> {
    try_read_breakdown(total, components).map_err(LossError::tensor_read)
}

fn try_read_breakdown(
    total: Tensor<1>,
    components: Vec<(Component, Tensor<1>)>,
) -> Result<LossBreakdown, TensorReadError> {
    let mut transaction = Transaction::default().register(total);
    let mut slots = Vec::with_capacity(components.len());
    for (component, tensor) in components {
        transaction = transaction.register(tensor);
        slots.push(component);
    }

    let mut values = Vec::with_capacity(slots.len() + 1);
    for data in transaction.try_execute()? {
        values.push(
            data.try_into_vec_as::<f64>()?
                .first()
                .copied()
                .unwrap_or(0.0),
        );
    }

    let mut values = values.into_iter();
    let mut breakdown = LossBreakdown {
        total: values.next().unwrap_or(0.0),
        ..LossBreakdown::default()
    };
    for (component, value) in slots.into_iter().zip(values) {
        *breakdown.slot_mut(component) += value;
    }
    Ok(breakdown)
}
