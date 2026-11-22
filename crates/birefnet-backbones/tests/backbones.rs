//! Integration tests for `birefnet-backbones`.

use birefnet_backbones::{
    Backbone, BackboneType, PvtV2Config, PvtV2Variant, ResNetConfig, ResNetVariant,
    SwinTransformerConfig, SwinVariant, VggVariant,
};
use burn::{prelude::*, tensor::Distribution};

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

    let bad_pvt = PvtV2Config::from_variant(PvtV2Variant::B0, 0)
        .init(&device)
        .unwrap_err();
    assert!(bad_pvt.is_invalid_config());

    let duplicate_outputs = SwinTransformerConfig::new()
        .with_out_indices([0, 1, 1, 3])
        .init(&device)
        .unwrap_err();
    assert!(duplicate_outputs.is_invalid_config());
    assert!(duplicate_outputs.to_string().contains("distinct"));

    let bad_ratio = SwinTransformerConfig::new()
        .with_mlp_ratio(0.0)
        .init(&device)
        .unwrap_err();
    assert!(bad_ratio.is_invalid_config());

    let bad_drop = SwinTransformerConfig::new()
        .with_drop_path_rate(1.0)
        .init(&device)
        .unwrap_err();
    assert!(bad_drop.is_invalid_config());
}

fn tiny_swin(depths: [usize; 4]) -> SwinTransformerConfig {
    SwinTransformerConfig::new()
        .with_pretrain_img_size(32)
        .with_embed_dim(8)
        .with_depths(depths)
        .with_num_heads([1, 2, 4, 8])
        .with_window_size(4)
}

#[test]
fn shifted_window_stages_mix_information_across_window_borders() {
    let device = Device::flex();
    device.seed(3);
    // Depth 2 adds a shifted block to every stage; 64x64 -> 16x16 tokens -> 4x4 windows.
    let swin = tiny_swin([2, 2, 2, 2]).init(&device).unwrap();

    let base = Tensor::<4>::random([1, 3, 64, 64], Distribution::Uniform(0.0, 1.0), &device);
    // Perturb a single pixel in the top-left 4x4 window of the stage-1 token grid.
    let perturbed = base.clone().slice_fill(s![.., .., 0..4, 0..4], 5.0);

    let [a, ..] = swin.forward(base).unwrap();
    let [b, ..] = swin.forward(perturbed).unwrap();
    assert_eq!(a.dims(), [1, 8, 16, 16]);

    // Token (5, 5) lies outside the perturbed regular window (rows/cols 0..4) but inside the
    // shifted window that straddles it, so it only changes if shifted attention ran.
    let diff = (a - b)
        .slice(s![.., .., 4..8, 4..8])
        .abs()
        .max()
        .into_scalar::<f32>();
    assert!(
        diff > 1e-6,
        "shifted windows did not propagate the change: {diff}"
    );
}

#[test]
fn absolute_position_embedding_is_interpolated_to_the_input_grid() {
    let device = Device::flex();
    let swin = tiny_swin([1, 1, 1, 1])
        .with_ape(true)
        .init(&device)
        .unwrap();
    // Pretrain grid is 8x8; a 48x48 input yields a 12x12 grid, exercising interpolation.
    let [s1, _, _, s4] = swin
        .forward(Tensor::zeros([1, 3, 48, 48], &device))
        .unwrap();
    assert_eq!(s1.dims(), [1, 8, 12, 12]);
    assert_eq!(s4.dims(), [1, 64, 2, 2]);
}

#[test]
fn pvt_v2_with_four_input_channels_uses_the_fourth_channel() {
    let device = Device::flex();
    device.seed(11);
    let pvt = PvtV2Config::new(
        4,
        [8, 16, 32, 64],
        [1, 2, 4, 8],
        [2.0; 4],
        true,
        [1, 1, 1, 1],
        [4, 2, 1, 1],
        0.0,
        0.0,
    )
    .init(&device)
    .unwrap();

    let rgb = Tensor::<4>::random([1, 3, 32, 32], Distribution::Uniform(0.0, 1.0), &device);
    let with_mask = |value: f32| {
        let mask = Tensor::<4>::full([1, 1, 32, 32], value, &device);
        Tensor::cat(vec![rgb.clone(), mask], 1)
    };
    let [a, ..] = pvt.forward(with_mask(0.0));
    let [b, ..] = pvt.forward(with_mask(1.0));
    let diff = (a - b).abs().max().into_scalar::<f32>();
    assert!(diff > 1e-4, "fourth channel had no effect: {diff}");
}

#[test]
fn backbone_types_display_upstream_names() {
    assert_eq!(
        BackboneType::SwinTransformer(SwinVariant::SwinL).to_string(),
        "swin_v1_l"
    );
    assert_eq!(
        BackboneType::PvtV2(PvtV2Variant::B2).to_string(),
        "pvt_v2_b2"
    );
    assert_eq!(
        BackboneType::ResNet(ResNetVariant::ResNet50).to_string(),
        "resnet50"
    );
    assert_eq!(
        BackboneType::Vgg(VggVariant::Vgg16Bn).to_string(),
        "vgg16bn"
    );
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
