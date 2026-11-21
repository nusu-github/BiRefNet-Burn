//! U-Net based refinement module (`RefUNet`) for `BiRefNet`.

use burn::{
    nn::{
        BatchNorm, BatchNormConfig, PaddingConfig2d, Relu,
        conv::{Conv2d, Conv2dConfig},
        pool::{MaxPool2d, MaxPool2dConfig},
    },
    prelude::*,
};

use crate::{
    config::InterpolationStrategy, error::ModelError,
    models::modules::utils::intelligent_interpolate,
};

/// Configuration for [`RefUNet`].
#[derive(Config, Debug)]
pub struct RefUNetConfig {
    /// Number of input channels (default: `4` for RGB + mask).
    #[config(default = 4)]
    in_channels: usize,
    /// Interpolation strategy for decoder upsampling.
    interpolation_strategy: InterpolationStrategy,
}

/// U-Net refinement module operating on concatenated image and coarse mask predictions.
#[derive(Module, Debug)]
pub struct RefUNet {
    encoder_1: (Conv2d, Conv2d, BatchNorm, Relu),
    encoder_2: (MaxPool2d, Conv2d, BatchNorm, Relu),
    encoder_3: (MaxPool2d, Conv2d, BatchNorm, Relu),
    encoder_4: (MaxPool2d, Conv2d, BatchNorm, Relu),
    pool4: MaxPool2d,

    decoder_5: (Conv2d, BatchNorm, Relu),
    decoder_4: (Conv2d, BatchNorm, Relu),
    decoder_3: (Conv2d, BatchNorm, Relu),
    decoder_2: (Conv2d, BatchNorm, Relu),
    decoder_1: (Conv2d, BatchNorm, Relu),

    conv_d0: Conv2d,
    #[module(skip)]
    interpolation_strategy: InterpolationStrategy,
}

fn encoder_stage(in_channels: usize, device: &Device) -> (MaxPool2d, Conv2d, BatchNorm, Relu) {
    (
        MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init(),
        Conv2dConfig::new([in_channels, 64], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .init(device),
        BatchNormConfig::new(64).init(device),
        Relu::new(),
    )
}

fn decoder_stage(in_channels: usize, device: &Device) -> (Conv2d, BatchNorm, Relu) {
    (
        Conv2dConfig::new([in_channels, 64], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .init(device),
        BatchNormConfig::new(64).init(device),
        Relu::new(),
    )
}

impl RefUNetConfig {
    /// Initializes a [`RefUNet`] module.
    #[must_use]
    pub fn init(&self, device: &Device) -> RefUNet {
        let encoder_1 = (
            Conv2dConfig::new([self.in_channels, 64], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
            Conv2dConfig::new([64, 64], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
            BatchNormConfig::new(64).init(device),
            Relu::new(),
        );
        let encoder_2 = encoder_stage(64, device);
        let encoder_3 = encoder_stage(64, device);
        let encoder_4 = encoder_stage(64, device);
        let pool4 = MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init();

        let decoder_5 = decoder_stage(64, device);
        let decoder_4 = decoder_stage(128, device);
        let decoder_3 = decoder_stage(128, device);
        let decoder_2 = decoder_stage(128, device);
        let decoder_1 = decoder_stage(128, device);

        let conv_d0 = Conv2dConfig::new([64, 1], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .init(device);

        RefUNet {
            encoder_1,
            encoder_2,
            encoder_3,
            encoder_4,
            pool4,
            decoder_5,
            decoder_4,
            decoder_3,
            decoder_2,
            decoder_1,
            conv_d0,
            interpolation_strategy: self.interpolation_strategy,
        }
    }
}

impl RefUNet {
    fn run_encoder_stage(stage: &(MaxPool2d, Conv2d, BatchNorm, Relu), x: Tensor<4>) -> Tensor<4> {
        let x = stage.0.forward(x);
        let x = stage.1.forward(x);
        let x = stage.2.forward(x);
        stage.3.forward(x)
    }

    fn run_decoder_stage(
        &self,
        stage: &(Conv2d, BatchNorm, Relu),
        prev: Tensor<4>,
        skip: Tensor<4>,
    ) -> Tensor<4> {
        let [_, _, h, w] = skip.dims();
        let up = intelligent_interpolate(prev, [h, w], self.interpolation_strategy);
        let cat = Tensor::cat(vec![up, skip], 1);
        let x = stage.0.forward(cat);
        let x = stage.1.forward(x);
        stage.2.forward(x)
    }

    /// Runs the refinement U-Net over concatenated `inputs`.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if `inputs` is empty.
    pub fn forward(&self, inputs: &[Tensor<4>]) -> Result<Vec<Tensor<4>>, ModelError> {
        let hx = match inputs {
            [] => {
                return Err(ModelError::invalid_input(
                    "RefUNet requires at least one input tensor",
                ));
            }
            [single] => single.clone(),
            many => Tensor::cat(many.to_vec(), 1),
        };

        let hx1 = {
            let x = self.encoder_1.0.forward(hx);
            let x = self.encoder_1.1.forward(x);
            let x = self.encoder_1.2.forward(x);
            self.encoder_1.3.forward(x)
        };
        let hx2 = Self::run_encoder_stage(&self.encoder_2, hx1.clone());
        let hx3 = Self::run_encoder_stage(&self.encoder_3, hx2.clone());
        let hx4 = Self::run_encoder_stage(&self.encoder_4, hx3.clone());

        let bottleneck = {
            let x = self.pool4.forward(hx4.clone());
            let x = self.decoder_5.0.forward(x);
            let x = self.decoder_5.1.forward(x);
            self.decoder_5.2.forward(x)
        };

        let d4 = self.run_decoder_stage(&self.decoder_4, bottleneck, hx4);
        let d3 = self.run_decoder_stage(&self.decoder_3, d4, hx3);
        let d2 = self.run_decoder_stage(&self.decoder_2, d3, hx2);
        let d1 = self.run_decoder_stage(&self.decoder_1, d2, hx1);

        Ok(vec![self.conv_d0.forward(d1)])
    }
}
