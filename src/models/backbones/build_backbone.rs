use burn::prelude::*;

use super::{swin_v1_b, swin_v1_l, swin_v1_s, swin_v1_t, SwinTransformer};

#[derive(Module, Debug)]
pub enum BackboneEnum<B: Backend> {
    SwinTransformer(SwinTransformer<B>),
}

#[derive(Config, Debug)]
pub enum Backbone {
    VGG16,
    VGG16BN,
    ResNet50,
    SwinV1T,
    SwinV1S,
    SwinV1B,
    SwinV1L,
    PVTv2B0,
    PVTv2B1,
    PVTv2B2,
    PVTv2B5,
}

pub fn build_backbone<B: Backend>(
    backbone: &Backbone,
    pretrained: bool,
    device: &Device<B>,
) -> BackboneEnum<B> {
    match backbone {
        Backbone::VGG16 => unimplemented!(),
        Backbone::VGG16BN => unimplemented!(),
        Backbone::ResNet50 => unimplemented!(),
        Backbone::SwinV1T => BackboneEnum::SwinTransformer(swin_v1_t(device)),
        Backbone::SwinV1S => BackboneEnum::SwinTransformer(swin_v1_s(device)),
        Backbone::SwinV1B => BackboneEnum::SwinTransformer(swin_v1_b(device)),
        Backbone::SwinV1L => BackboneEnum::SwinTransformer(swin_v1_l(device)),
        Backbone::PVTv2B0 => unimplemented!(),
        Backbone::PVTv2B1 => unimplemented!(),
        Backbone::PVTv2B2 => unimplemented!(),
        Backbone::PVTv2B5 => unimplemented!(),
    }
}
