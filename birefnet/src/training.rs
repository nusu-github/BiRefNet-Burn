//! Training functionality for BiRefNet.
//!
//! This module implements the TrainStep and ValidStep traits for the BiRefNet model,
//! enabling integration with the Burn training framework.

use crate::metrics::{BiRefNetLossInput, FMeasureInput, IoUInput, MAEInput};
use burn::{
    prelude::*,
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

impl<B: Backend> ItemLazy for BiRefNetOutput<B> {
    type ItemSync = BiRefNetOutput<burn::backend::ndarray::NdArray>;

    fn sync(self) -> Self::ItemSync {
        let [logits, target, loss] = Transaction::default()
            .register(self.logits)
            .register(self.target)
            .register(self.loss)
            .execute()
            .try_into()
            .expect("Correct amount of tensor data");

        let device = &Default::default();

        BiRefNetOutput {
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
