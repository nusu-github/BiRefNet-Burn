//! Integration tests for `birefnet-model` configuration, forward passes, and training steps.

use birefnet_model::{
    Backbone, BackboneConfig, BiRefNetConfig, DecoderAttention, DecoderConfig,
    InterpolationStrategy, ModelConfig, MultiScaleInput, Refine, RefineConfig, SqueezeBlock,
    TaskConfig,
};
use burn::{config::Config as _, prelude::*, tensor::Distribution};

fn tiny_config(cxt_num: usize, refine: Refine) -> ModelConfig {
    ModelConfig::new(InterpolationStrategy::Nearest)
        .with_task(TaskConfig::new().with_batch_size(1))
        .with_backbone(BackboneConfig::new().with_backbone(Backbone::PvtV2B0))
        .with_decoder(
            DecoderConfig::new()
                .with_ms_supervision(false)
                .with_out_ref(false)
                .with_dec_ipt(true)
                .with_dec_ipt_split(false)
                .with_cxt_num(cxt_num)
                .with_mul_scl_ipt(MultiScaleInput::None)
                .with_dec_att(DecoderAttention::None)
                .with_squeeze_block(SqueezeBlock::BasicDecBlk(1)),
        )
        .with_refine(RefineConfig::new().with_refine(refine))
}

#[test]
fn config_validate_rejects_inconsistent_combinations() {
    let invalid_cxt = ModelConfig::new(InterpolationStrategy::Nearest)
        .with_decoder(DecoderConfig::new().with_cxt_num(4));
    assert!(invalid_cxt.validate().unwrap_err().is_invalid_config());

    let out_ref_without_ms = ModelConfig::new(InterpolationStrategy::Nearest).with_decoder(
        DecoderConfig::new()
            .with_ms_supervision(false)
            .with_out_ref(true),
    );
    assert!(
        out_ref_without_ms
            .validate()
            .unwrap_err()
            .is_invalid_config()
    );

    let split_without_ipt = ModelConfig::new(InterpolationStrategy::Nearest).with_decoder(
        DecoderConfig::new()
            .with_dec_ipt(false)
            .with_dec_ipt_split(true),
    );
    assert!(
        split_without_ipt
            .validate()
            .unwrap_err()
            .is_invalid_config()
    );

    let incompatible_att = ModelConfig::new(InterpolationStrategy::Nearest).with_decoder(
        DecoderConfig::new()
            .with_dec_att(DecoderAttention::ASPP)
            .with_squeeze_block(SqueezeBlock::BasicDecBlk(1)),
    );
    assert!(incompatible_att.validate().unwrap_err().is_invalid_config());
    assert!(
        BiRefNetConfig::new(incompatible_att)
            .init(&Device::flex())
            .unwrap_err()
            .is_invalid_config()
    );

    let itself = ModelConfig::new(InterpolationStrategy::Nearest)
        .with_refine(RefineConfig::new().with_refine(Refine::Itself));
    let err = itself.validate().unwrap_err();
    assert!(err.is_invalid_config());
    assert!(err.to_string().contains("itself"));
}

#[test]
fn multi_scale_input_add_and_cat_forward() {
    let device = Device::flex();
    let input = Tensor::<4>::random([1, 3, 64, 64], Distribution::Normal(0.0, 1.0), &device);

    for mode in [MultiScaleInput::Add, MultiScaleInput::Cat] {
        let mut cfg = tiny_config(1, Refine::None);
        cfg.decoder.mul_scl_ipt = mode;
        let model = BiRefNetConfig::new(cfg).init(&device).unwrap();

        let [x1, .., x4] = model.forward_enc(input.clone()).unwrap();
        // PVTv2-B0 emits 32 channels at stride 4; `Cat` doubles every stage.
        let expected = if mode == MultiScaleInput::Cat { 64 } else { 32 };
        assert_eq!(x1.dims(), [1, expected, 16, 16], "{mode}");
        assert_eq!(x4.dims()[2..], [2, 2], "{mode}");

        let output = model.forward(input.clone()).unwrap();
        assert_eq!(output.dims(), [1, 1, 64, 64], "{mode}");
    }
}

#[test]
fn split_image_pyramid_and_multi_scale_supervision_forward() {
    let device = Device::flex();
    let mut cfg = tiny_config(0, Refine::None);
    cfg.decoder.dec_ipt_split = true;
    cfg.decoder.ms_supervision = true;
    cfg.decoder.out_ref = true;
    let model = BiRefNetConfig::new(cfg).init(&device).unwrap();
    // 64x64 keeps every pyramid level (down to 2x2 features, i.e. 32x32 tiles of 2x2 pixels).
    let output = model
        .forward(Tensor::<4>::random(
            [1, 3, 64, 64],
            Distribution::Uniform(0.0, 1.0),
            &device,
        ))
        .unwrap();
    assert_eq!(output.dims(), [1, 1, 64, 64]);
}

#[test]
fn forward_enc_rejects_non_rgb_inputs() {
    let device = Device::flex();
    let model = BiRefNetConfig::new(tiny_config(0, Refine::None))
        .init(&device)
        .unwrap();
    let err = model
        .forward_enc(Tensor::zeros([1, 4, 64, 64], &device))
        .unwrap_err();
    assert!(err.is_invalid_input());
}

#[test]
fn config_enums_display_upstream_option_strings() {
    assert_eq!(Backbone::SwinV1L.to_string(), "swin_v1_l");
    assert_eq!(Backbone::Vgg16Bn.to_string(), "vgg16bn");
    assert_eq!(SqueezeBlock::BasicDecBlk(1).to_string(), "BasicDecBlk_x1");
    assert_eq!(MultiScaleInput::Cat.to_string(), "cat");
    assert_eq!(Refine::RefUNet.to_string(), "RefUNet");
}

#[test]
fn legacy_backbone_names_still_deserialize() {
    let legacy = BackboneConfig::load_binary(br#"{"backbone":"Resnet50"}"#).unwrap();
    assert_eq!(legacy.backbone, Backbone::ResNet50);
    let legacy = BackboneConfig::load_binary(br#"{"backbone":"Vgg16bn"}"#).unwrap();
    assert_eq!(legacy.backbone, Backbone::Vgg16Bn);
}

#[test]
fn forward_supports_all_context_counts_0_to_3() {
    let device = Device::flex();
    let input = Tensor::<4>::random([1, 3, 64, 64], Distribution::Normal(0.0, 1.0), &device);

    for cxt_num in 0..=3 {
        let model = BiRefNetConfig::new(tiny_config(cxt_num, Refine::None))
            .init(&device)
            .expect("model initialization must succeed");
        let output = model
            .forward(input.clone())
            .expect("forward pass must succeed");
        assert_eq!(output.dims(), [1, 1, 64, 64]);
        let vals = output.into_data().try_to_vec::<f32>().unwrap();
        assert!(vals.iter().all(|v| v.is_finite()), "cxt_num={cxt_num}");
    }
}

#[test]
fn forward_works_with_refunet_and_stem_refiner() {
    let device = Device::flex();
    let input = Tensor::<4>::random([1, 3, 64, 64], Distribution::Normal(0.0, 1.0), &device);

    for refine in [
        Refine::RefUNet,
        Refine::Refiner,
        Refine::RefinerPVTInChannels4,
    ] {
        let model = BiRefNetConfig::new(tiny_config(1, refine))
            .init(&device)
            .expect("refinement model init must succeed");
        let output = model
            .forward(input.clone())
            .expect("refinement forward must succeed");
        assert_eq!(output.dims(), [1, 1, 64, 64]);
    }
}

#[test]
fn channel_tables_match_backbones_and_cat_doubling() {
    let cfg = ModelConfig::new(InterpolationStrategy::Nearest)
        .with_backbone(BackboneConfig::new().with_backbone(Backbone::ResNet50))
        .with_decoder(DecoderConfig::new().with_mul_scl_ipt(MultiScaleInput::None));
    assert_eq!(cfg.lateral_channels_in_collection(), [2048, 1024, 512, 256]);

    let cat_cfg = cfg.with_decoder(DecoderConfig::new().with_mul_scl_ipt(MultiScaleInput::Cat));
    assert_eq!(
        cat_cfg.lateral_channels_in_collection(),
        [4096, 2048, 1024, 512]
    );
}

#[cfg(feature = "train")]
#[test]
fn train_and_inference_steps_produce_finite_loss_and_sync_detaches_autodiff() {
    use birefnet_loss::BiRefNetLossConfig;
    use birefnet_model::{BiRefNetBatch, BiRefNetOutput};
    use burn::train::{InferenceStep, TrainStep, metric::ItemLazy};

    let device = Device::flex().autodiff();
    let mut model_cfg = tiny_config(1, Refine::None);
    model_cfg.task.batch_size = 2;
    model_cfg.decoder.ms_supervision = true;
    let config = BiRefNetConfig::new(model_cfg).with_loss_config(Some(BiRefNetLossConfig::new()));
    let model = config.init(&device).unwrap();

    let batch = BiRefNetBatch::new(
        Tensor::<4>::random([2, 3, 64, 64], Distribution::Normal(0.0, 1.0), &device),
        Tensor::<4>::random([2, 1, 64, 64], Distribution::Uniform(0.0, 1.0), &device),
    );
    assert_eq!(batch.batch_size(), 2);

    let train_out = TrainStep::step(&model, batch);
    assert!(train_out.item.loss.clone().into_scalar::<f32>().is_finite());
    assert_eq!(train_out.item.output.dims(), [2, 1, 64, 64]);

    let synced: BiRefNetOutput = train_out.item.sync().unwrap();
    assert!(!synced.loss.is_autodiff());
    assert!(!synced.output.is_autodiff());
    assert!(!synced.targets.is_autodiff());

    let eval_device = Device::flex();
    let mut eval_cfg = tiny_config(1, Refine::None);
    eval_cfg.task.batch_size = 2;
    let eval_model = BiRefNetConfig::new(eval_cfg)
        .with_loss_config(Some(BiRefNetLossConfig::new()))
        .init(&eval_device)
        .unwrap();
    let eval_batch = BiRefNetBatch::new(
        Tensor::<4>::random([2, 3, 64, 64], Distribution::Normal(0.0, 1.0), &eval_device),
        Tensor::<4>::random(
            [2, 1, 64, 64],
            Distribution::Uniform(0.0, 1.0),
            &eval_device,
        ),
    );
    let valid_out = InferenceStep::step(&eval_model, eval_batch);
    assert!(valid_out.loss.into_scalar::<f32>().is_finite());
}

#[test]
#[ignore = "slow SwinV1L allocation; run explicitly to smoke-test default config"]
fn default_swin_large_config_initializes_and_forwards() {
    let device = Device::flex();
    let model = BiRefNetConfig::new(ModelConfig::new(InterpolationStrategy::Nearest))
        .init(&device)
        .unwrap();
    let input = Tensor::<4>::random([1, 3, 128, 128], Distribution::Normal(0.0, 1.0), &device);
    let output = model.forward(input).unwrap();
    assert_eq!(output.dims(), [1, 1, 128, 128]);
}
