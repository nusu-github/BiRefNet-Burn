use burn::prelude::*;

use super::{swin_v1_b, swin_v1_l, swin_v1_s, swin_v1_t, SwinTransformer};
use crate::config::{Backbone, ModelConfig};

#[derive(Module, Debug)]
pub enum BackboneEnum<B: Backend> {
    SwinTransformer(SwinTransformer<B>),
}

pub fn build_backbone<B: Backend>(config: &ModelConfig, device: &Device<B>) -> BackboneEnum<B> {
    match config.backbone {
        Backbone::Vgg16 => unimplemented!(),
        Backbone::Vgg16bn => unimplemented!(),
        Backbone::Resnet50 => unimplemented!(),
        Backbone::SwinV1T => BackboneEnum::SwinTransformer(swin_v1_t(device)),
        Backbone::SwinV1S => BackboneEnum::SwinTransformer(swin_v1_s(device)),
        Backbone::SwinV1B => BackboneEnum::SwinTransformer(swin_v1_b(device)),
        Backbone::SwinV1L => BackboneEnum::SwinTransformer(swin_v1_l(device)),
        Backbone::PvtV2B0 => unimplemented!(),
        Backbone::PvtV2B1 => unimplemented!(),
        Backbone::PvtV2B2 => unimplemented!(),
        Backbone::PvtV2B5 => unimplemented!(),
    }
}
