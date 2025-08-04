//! Training functionality for BiRefNet.
//!
//! This module implements the TrainStep and ValidStep traits for the BiRefNet model,
//! enabling integration with the Burn training framework.

use crate::metrics::{BiRefNetLossInput, FMeasureInput, IoUInput, MAEInput};
use burn::{
    prelude::*,
    tensor::backend::AutodiffBackend,
    tensor::{backend::Backend, Transaction},
    train::metric::{Adaptor, ItemLazy},
};

/// Output structure for BiRefNet training and validation steps.
#[derive(Debug, Clone)]
pub struct BiRefNetOutput<B: Backend> {
    pub logits: Tensor<B, 4>,
    pub target: Tensor<B, 4>,
    pub loss: Tensor<B, 1>,
}

/// Extended output structure for BiRefNet full forward pass containing all prediction scales and components.
#[derive(Debug, Clone)]
pub struct BiRefNetFullOutput<B: Backend> {
    /// Main scaled predictions for different scales
    pub scaled_preds: Vec<Tensor<B, 4>>,
    /// Classification predictions (if applicable)
    pub class_preds: Option<Vec<Tensor<B, 4>>>,
    /// Gradient Direction Tensor outputs
    pub gdt_outputs: Option<Vec<Tensor<B, 4>>>,
    /// Primary prediction (usually the highest resolution)
    pub primary_pred: Tensor<B, 4>,
}

impl<B: Backend> Adaptor<FMeasureInput<B>> for BiRefNetFullOutput<B> {
    fn adapt(&self) -> FMeasureInput<B> {
        FMeasureInput::new(
            self.primary_pred.clone(),
            self.primary_pred.clone(), // Using primary_pred as both pred and target for now
        )
    }
}

impl<B: Backend> Adaptor<MAEInput<B>> for BiRefNetFullOutput<B> {
    fn adapt(&self) -> MAEInput<B> {
        MAEInput::new(
            self.primary_pred.clone(),
            self.primary_pred.clone(), // Using primary_pred as both pred and target for now
        )
    }
}

impl<B: Backend> Adaptor<IoUInput<B>> for BiRefNetFullOutput<B> {
    fn adapt(&self) -> IoUInput<B> {
        IoUInput::new(
            self.primary_pred.clone(),
            self.primary_pred.clone(), // Using primary_pred as both pred and target for now
        )
    }
}

impl<B: Backend> Adaptor<BiRefNetLossInput<B>> for BiRefNetFullOutput<B> {
    fn adapt(&self) -> BiRefNetLossInput<B> {
        // Calculate a dummy loss for now - this should be replaced with actual loss computation
        let dummy_loss = self.primary_pred.clone().mean().unsqueeze();
        let batch_size = self.primary_pred.dims()[0];
        BiRefNetLossInput::new(dummy_loss, batch_size)
    }
}

impl<B: Backend> BiRefNetFullOutput<B> {
    /// Create a new BiRefNetFullOutput
    pub fn new(
        scaled_preds: Vec<Tensor<B, 4>>,
        class_preds: Option<Vec<Tensor<B, 4>>>,
        gdt_outputs: Option<Vec<Tensor<B, 4>>>,
    ) -> Self {
        let primary_pred = scaled_preds[0].clone();
        Self {
            scaled_preds,
            class_preds,
            gdt_outputs,
            primary_pred,
        }
    }
}

impl<B: AutodiffBackend> ItemLazy for BiRefNetFullOutput<B> {
    type ItemSync = BiRefNetFullOutput<B::InnerBackend>;
    fn sync(self) -> Self::ItemSync {
        let scaled_preds = self.scaled_preds.into_iter().map(|t| t.inner()).collect();
        let class_preds = self
            .class_preds
            .map(|preds| preds.into_iter().map(|t| t.inner()).collect());
        let gdt_outputs = self
            .gdt_outputs
            .map(|outputs| outputs.into_iter().map(|t| t.inner()).collect());

        BiRefNetFullOutput {
            scaled_preds,
            class_preds,
            gdt_outputs,
            primary_pred: self.primary_pred.inner(),
        }
    }
}

impl<B: Backend> ItemLazy for BiRefNetOutput<B> {
    type ItemSync = Self;

    fn sync(self) -> Self::ItemSync {
        let transaction_result = Transaction::default()
            .register(self.logits)
            .register(self.target)
            .register(self.loss)
            .execute();

        let [logits, target, loss] = transaction_result.try_into().unwrap_or_else(|_| {
            panic!(
                "Failed to extract exactly 3 tensors from transaction. \
                     Expected: [logits, target, loss]. This indicates a programming error \
                     in BiRefNetOutput::sync implementation."
            )
        });

        let device = &Default::default();

        Self {
            logits: Tensor::from_data(logits, device),
            target: Tensor::from_data(target, device),
            loss: Tensor::from_data(loss, device),
        }
    }
}

impl<B: Backend> Adaptor<FMeasureInput<B>> for BiRefNetOutput<B> {
    fn adapt(&self) -> FMeasureInput<B> {
        FMeasureInput {
            predictions: self.logits.clone(),
            targets: self.target.clone(),
        }
    }
}

impl<B: Backend> Adaptor<MAEInput<B>> for BiRefNetOutput<B> {
    fn adapt(&self) -> MAEInput<B> {
        MAEInput {
            predictions: self.logits.clone(),
            targets: self.target.clone(),
        }
    }
}

impl<B: Backend> Adaptor<IoUInput<B>> for BiRefNetOutput<B> {
    fn adapt(&self) -> IoUInput<B> {
        IoUInput {
            predictions: self.logits.clone(),
            targets: self.target.clone(),
        }
    }
}

impl<B: Backend> Adaptor<BiRefNetLossInput<B>> for BiRefNetOutput<B> {
    fn adapt(&self) -> BiRefNetLossInput<B> {
        BiRefNetLossInput {
            loss: self.loss.clone(),
            batch_size: self.logits.dims()[0],
        }
    }
}
