//! Integration tests for `birefnet-backbones`.

use birefnet_backbones::{
    Backbone, BackboneType, PvtV2Config, PvtV2Variant, ResNetConfig, ResNetVariant,
    SwinTransformerConfig, SwinVariant, VggVariant,
};
use burn::prelude::*;

#[test]
fn tiny_swin_forward_handles_even_and_odd_patch_merging_dimensions() {
    let device = Device::flex();
    let swin = SwinTransformerConfig::new()
        .with_pretrain_img_size(32)
        .with_patch_size(4)
        .with_embed_dim(8)
        .with_depths([1, 1, 1, 1])
        .with_num_heads([1, 2, 4, 8])
        .with_window_size(4)
        .init(&device)
        .unwrap();

    assert_eq!(swin.output_channels(), [8, 16, 32, 64]);

    // 32x32 -> 8x8 -> 4x4 -> 2x2 -> 1x1
    let [s1, s2, s3, s4] = swin
        .forward(Tensor::zeros([1, 3, 32, 32], &device))
        .unwrap();
    assert_eq!(s1.dims(), [1, 8, 8, 8]);
    assert_eq!(s2.dims(), [1, 16, 4, 4]);
    assert_eq!(s3.dims(), [1, 32, 2, 2]);
    assert_eq!(s4.dims(), [1, 64, 1, 1]);

    // 20x20 -> 5x5 (odd) -> PatchMerging ceil(5/2)=3 -> 3x3 (odd) -> 2x2 -> 1x1
    let [o1, o2, o3, o4] = swin
        .forward(Tensor::zeros([1, 3, 20, 20], &device))
        .unwrap();
    assert_eq!(o1.dims(), [1, 8, 5, 5]);
    assert_eq!(o2.dims(), [1, 16, 3, 3]);
    assert_eq!(o3.dims(), [1, 32, 2, 2]);
    assert_eq!(o4.dims(), [1, 64, 1, 1]);
}

#[test]
fn tiny_pvt_v2_outputs_expected_shapes_and_channels() {
    let device = Device::flex();
    let pvt = PvtV2Config::new(
        3,
        [8, 16, 32, 64],
        [1, 2, 4, 8],
        [2.0; 4],
        true,
        [1, 1, 1, 1],
        [4, 2, 1, 1],
        0.0,
        0.1,
    )
    .init(&device)
    .unwrap();

    assert_eq!(pvt.output_channels(), [8, 16, 32, 64]);
    let [p1, p2, p3, p4] = pvt.forward(Tensor::zeros([1, 3, 32, 32], &device));
    assert_eq!(p1.dims(), [1, 8, 8, 8]);
    assert_eq!(p2.dims(), [1, 16, 4, 4]);
    assert_eq!(p3.dims(), [1, 32, 2, 2]);
    assert_eq!(p4.dims(), [1, 64, 1, 1]);
}

#[test]
fn resnet18_and_resnet50_report_accurate_output_channels_and_shapes() {
    let device = Device::flex();
    let r18 = BackboneType::ResNet(ResNetVariant::ResNet18)
        .init(&device)
        .unwrap();
    assert_eq!(r18.output_channels(), [64, 128, 256, 512]);
    let [x1, x2, x3, x4] = r18.forward(Tensor::zeros([1, 3, 64, 64], &device));
    assert_eq!(x1.dims(), [1, 64, 16, 16]);
    assert_eq!(x2.dims(), [1, 128, 8, 8]);
    assert_eq!(x3.dims(), [1, 256, 4, 4]);
    assert_eq!(x4.dims(), [1, 512, 2, 2]);

    let r50 = BackboneType::ResNet(ResNetVariant::ResNet50)
        .init(&device)
        .unwrap();
    assert_eq!(r50.output_channels(), [256, 512, 1024, 2048]);
}

#[test]
fn vgg16_variants_extract_expected_feature_pyramid() {
    let device = Device::flex();
    for variant in [VggVariant::Vgg16, VggVariant::Vgg16Bn] {
        let model = BackboneType::Vgg(variant).init(&device).unwrap();
        assert_eq!(model.output_channels(), [128, 256, 512, 512]);
        let [v1, v2, v3, v4] = model.forward(Tensor::zeros([1, 3, 64, 64], &device));
        assert_eq!(v1.dims(), [1, 128, 16, 16]);
        assert_eq!(v2.dims(), [1, 256, 8, 8]);
        assert_eq!(v3.dims(), [1, 512, 4, 4]);
        assert_eq!(v4.dims(), [1, 512, 2, 2]);
    }
}

#[test]
fn output_channels_match_variant_expectations_across_families() {
    let device = Device::flex();
    assert_eq!(
        BackboneType::SwinTransformer(SwinVariant::SwinT)
            .init(&device)
            .unwrap()
            .output_channels(),
        [96, 192, 384, 768]
    );
    assert_eq!(
        BackboneType::PvtV2(PvtV2Variant::B0)
            .init(&device)
            .unwrap()
            .output_channels(),
        [32, 64, 160, 256]
    );
    assert_eq!(
        BackboneType::ResNet(ResNetVariant::ResNet34)
            .init(&device)
            .unwrap()
            .output_channels(),
        [64, 128, 256, 512]
    );
}

#[test]
fn invalid_configs_return_typed_errors() {
    let device = Device::flex();
    let bad_swin = SwinTransformerConfig::new()
        .with_embed_dim(10)
        .with_num_heads([3, 6, 12, 24])
        .init(&device)
        .unwrap_err();
    assert!(bad_swin.is_invalid_config());

    let bad_resnet = ResNetConfig::new([3, 4, 6, 3])
        .with_expansion(2)
        .init(&device)
        .unwrap_err();
    assert!(bad_resnet.is_invalid_config());

    let bad_pvt = PvtV2Config::b0(0).init(&device).unwrap_err();
    assert!(bad_pvt.is_invalid_config());
}

#[test]
#[ignore = "slow full-scale Swin-L 224x224 forward pass"]
fn full_scale_swin_l_forward_224() {
    let device = Device::flex();
    let model = BackboneType::SwinTransformer(SwinVariant::SwinL)
        .init(&device)
        .unwrap();
    assert_eq!(model.output_channels(), [192, 384, 768, 1536]);
    let [s1, s2, s3, s4] = model.forward(Tensor::zeros([1, 3, 224, 224], &device));
    assert_eq!(s1.dims(), [1, 192, 56, 56]);
    assert_eq!(s2.dims(), [1, 384, 28, 28]);
    assert_eq!(s3.dims(), [1, 768, 14, 14]);
    assert_eq!(s4.dims(), [1, 1536, 7, 7]);
}
