//! Integration tests for mask postprocessing and end-to-end `Segmenter` inference.

use birefnet_inference::{
    PostprocessConfig, Segmenter, SegmenterConfig, apply_threshold, fill_holes, gaussian_blur,
    mask_to_u8, morphological_closing, morphological_opening, postprocess_mask,
    remove_small_components, resize_tensor,
};
use birefnet_model::{
    Backbone, BackboneConfig, BiRefNetConfig, DecoderAttention, DecoderConfig,
    InterpolationStrategy, ModelConfig, MultiScaleInput, Refine, RefineConfig, SqueezeBlock,
    TaskConfig,
};
use burn::{prelude::*, tensor::ops::InterpolateMode};
use image::{DynamicImage, GenericImageView, Rgb, RgbImage};
use tempfile::TempDir;

fn tiny_model(device: &Device) -> birefnet_model::BiRefNet {
    let model_cfg = ModelConfig::new(InterpolationStrategy::Nearest)
        .with_task(TaskConfig::new().with_batch_size(1))
        .with_backbone(BackboneConfig::new().with_backbone(Backbone::PvtV2B0))
        .with_decoder(
            DecoderConfig::new()
                .with_ms_supervision(false)
                .with_out_ref(false)
                .with_dec_ipt(true)
                .with_dec_ipt_split(false)
                .with_cxt_num(0)
                .with_mul_scl_ipt(MultiScaleInput::None)
                .with_dec_att(DecoderAttention::None)
                .with_squeeze_block(SqueezeBlock::BasicDecBlk(1)),
        )
        .with_refine(RefineConfig::new().with_refine(Refine::None));
    BiRefNetConfig::new(model_cfg).init(device).unwrap()
}

#[test]
fn remove_small_components_filters_isolated_speckle_and_keeps_block() {
    let device = Device::flex();
    let mut grid = [0.0_f32; 7 * 7];
    // Isolated 1-pixel noise at (0, 0)
    grid[0] = 1.0;
    // 2x2 connected component at rows 3..5, cols 3..5 (size = 4)
    for y in 3..5 {
        for x in 3..5 {
            grid[y * 7 + x] = 1.0;
        }
    }

    let mask = Tensor::<4>::from_data(TensorData::new(grid.to_vec(), [1, 1, 7, 7]), &device);
    let cleaned = remove_small_components(mask, 2);
    let out: Vec<f32> = cleaned.to_data().iter::<f32>().collect();

    assert!(out[0].abs() < 1e-6, "isolated pixel should be removed");
    for y in 3..5 {
        for x in 3..5 {
            assert!((out[y * 7 + x] - 1.0).abs() < 1e-6);
        }
    }
}

#[test]
fn fill_holes_closes_interior_hole_while_preserving_exterior_background() {
    let device = Device::flex();
    let mut grid = [0.0_f32; 7 * 7];
    // Ring from rows 1..6, cols 1..6 with an enclosed hole at (3, 3)
    for y in 1..6 {
        for x in 1..6 {
            if !(y == 3 && x == 3) {
                grid[y * 7 + x] = 1.0;
            }
        }
    }

    let mask = Tensor::<4>::from_data(TensorData::new(grid.to_vec(), [1, 1, 7, 7]), &device);
    let filled = fill_holes(mask);
    let out: Vec<f32> = filled.to_data().iter::<f32>().collect();

    assert!(
        (out[3 * 7 + 3] - 1.0).abs() < 1e-6,
        "interior hole should be filled"
    );
    assert!(out[0].abs() < 1e-6, "border background should remain zero");
    assert!(
        out[6 * 7 + 6].abs() < 1e-6,
        "corner background should remain zero"
    );
}

#[test]
fn blur_morphology_resize_and_mask_export_operate_correctly() {
    let device = Device::flex();
    let mut grid = [0.0_f32; 7 * 7];
    grid[3 * 7 + 3] = 1.0;
    let impulse = Tensor::<4>::from_data(TensorData::new(grid.to_vec(), [1, 1, 7, 7]), &device);

    // Opening with radius 1 removes an isolated 1-pixel impulse.
    let opened = morphological_opening(impulse.clone(), 1);
    let opened_sum = opened.sum().into_scalar::<f32>();
    assert!(opened_sum.abs() < 1e-6);

    // Gaussian blur spreads impulse mass onto neighbors while remaining non-negative.
    let blurred = gaussian_blur(impulse.clone(), 4, 1.0);
    let center = blurred.to_data().iter::<f32>().nth(3 * 7 + 3).unwrap();
    assert!(center > 0.0 && center < 1.0);

    // Closing fills a 1-pixel gap between adjacent 3x3 blocks.
    let mut bridge = [0.0_f32; 7 * 7];
    for y in 2..5 {
        for x in 1..6 {
            if x != 3 {
                bridge[y * 7 + x] = 1.0;
            }
        }
    }
    let gap = Tensor::<4>::from_data(TensorData::new(bridge.to_vec(), [1, 1, 7, 7]), &device);
    let closed = morphological_closing(gap, 1);
    let center_closed = closed.to_data().iter::<f32>().nth(3 * 7 + 3).unwrap();
    assert!((center_closed - 1.0).abs() < 1e-6);

    // Pipeline helper & resize_tensor & mask_to_u8.
    let cfg = PostprocessConfig {
        threshold: Some(0.4),
        blur_kernel: Some((3, 0.8)),
        opening_radius: None,
        closing_radius: Some(1),
        min_component_size: Some(2),
        fill_holes: true,
    };
    let piped = postprocess_mask(apply_threshold(impulse, 0.5), &cfg);
    assert_eq!(piped.dims(), [1, 1, 7, 7]);

    let resized = resize_tensor(piped, 14, 10, InterpolateMode::Bilinear);
    assert_eq!(resized.dims(), [1, 1, 14, 10]);

    let gray = mask_to_u8(&resized).unwrap();
    assert_eq!(gray.dimensions(), (10, 14));
}

#[test]
fn segmenter_end_to_end_preserves_original_dimensions_and_outputs() {
    let device = Device::flex();
    let model = tiny_model(&device);
    let config = SegmenterConfig {
        input_size: (32, 32),
        refine_foreground: true,
        foreground_refine_radius: 2,
        postprocess: PostprocessConfig::default(),
    };
    let segmenter = Segmenter::new(model, config);

    let img = DynamicImage::ImageRgb8(RgbImage::from_fn(20, 28, |x, y| {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "small image indices stay within u8"
        )]
        Rgb([(x * 10) as u8, (y * 8) as u8, 128])
    }));

    let output = segmenter.segment_image(&img, &device).unwrap();
    assert_eq!(output.original_size(), (28, 20));
    assert_eq!(output.mask().dims(), [1, 1, 28, 20]);
    assert!(output.foreground().is_some());

    let rgba = output.to_rgba_image().unwrap();
    assert_eq!(rgba.dimensions(), (20, 28));
    let gray = output.to_mask_image().unwrap();
    assert_eq!(gray.dimensions(), (20, 28));

    let dir = TempDir::new().unwrap();
    let file_path = dir.path().join("sample.png");
    img.save(&file_path).unwrap();
    let file_output = segmenter.segment_file(&file_path, &device).unwrap();
    assert_eq!(file_output.original_size(), (28, 20));
    assert_eq!(file_output.mask().dims(), [1, 1, 28, 20]);
}
