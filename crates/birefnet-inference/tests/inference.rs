//! Integration tests for mask postprocessing and end-to-end `Segmenter` inference.

use std::{io::Cursor, num::NonZeroUsize};

use birefnet_inference::{
    Segmenter, SegmenterConfig,
    postprocess::{
        GaussianBlur, PostprocessConfig, apply_threshold, fill_holes, gaussian_blur, mask_to_u8,
        morphological_closing, morphological_opening, postprocess_mask, remove_small_components,
        resize_tensor,
    },
};
use birefnet_model::{
    Backbone, BackboneConfig, BiRefNetConfig, DecoderAttention, DecoderConfig,
    InterpolationStrategy, ModelConfig, MultiScaleInput, Refine, RefineConfig, SqueezeBlock,
    TaskConfig,
};
use burn::{prelude::*, tensor::ops::InterpolateMode};
use image::{DynamicImage, GenericImageView, ImageFormat, Rgb, RgbImage};
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

fn mask_7x7(grid: &[f32; 49], device: &Device) -> Tensor<4> {
    Tensor::<4>::from_data(TensorData::new(grid.to_vec(), [1, 1, 7, 7]), device)
}

fn values(tensor: Tensor<4>) -> Vec<f32> {
    tensor.try_into_vec_as::<f32>().unwrap()
}

#[test]
fn remove_small_components_filters_isolated_speckle_and_keeps_block() {
    let device = Device::flex();
    let mut grid = [0.0_f32; 49];
    grid[0] = 1.0; // isolated pixel
    for y in 3..5 {
        for x in 3..5 {
            grid[y * 7 + x] = 1.0; // 4-pixel block
        }
    }

    let out = values(remove_small_components(mask_7x7(&grid, &device), 2).unwrap());
    assert_eq!(out[0], 0.0, "isolated pixel should be removed");
    for y in 3..5 {
        for x in 3..5 {
            assert_eq!(out[y * 7 + x], 1.0);
        }
    }
    // Diagonal neighbors are not 4-connected: two diagonal pixels are two components.
    let mut diagonal = [0.0_f32; 49];
    diagonal[8] = 1.0;
    diagonal[16] = 1.0;
    let out = values(remove_small_components(mask_7x7(&diagonal, &device), 2).unwrap());
    assert!(out.iter().all(|&v| v == 0.0));
}

#[test]
fn fill_holes_closes_interior_hole_while_preserving_exterior_background() {
    let device = Device::flex();
    let mut grid = [0.0_f32; 49];
    for y in 1..6 {
        for x in 1..6 {
            if !(y == 3 && x == 3) {
                grid[y * 7 + x] = 1.0;
            }
        }
    }
    let out = values(fill_holes(mask_7x7(&grid, &device)).unwrap());
    assert_eq!(out[3 * 7 + 3], 1.0, "interior hole should be filled");
    assert_eq!(out[0], 0.0, "border background should remain zero");
    assert_eq!(out[48], 0.0, "corner background should remain zero");
}

#[test]
fn blur_morphology_resize_and_mask_export_operate_correctly() {
    let device = Device::flex();
    let mut grid = [0.0_f32; 49];
    grid[3 * 7 + 3] = 1.0;
    let impulse = mask_7x7(&grid, &device);

    // Opening with radius 1 removes an isolated 1-pixel impulse.
    assert_eq!(
        morphological_opening(impulse.clone(), 1)
            .sum()
            .into_scalar::<f32>(),
        0.0
    );

    // Gaussian blur spreads the impulse symmetrically and preserves its mass.
    let blurred = values(gaussian_blur(
        impulse.clone(),
        GaussianBlur::new(3, 1.0).unwrap(),
    ));
    assert!(blurred[24] > 0.0 && blurred[24] < 1.0);
    assert!((blurred[23] - blurred[25]).abs() < 1e-7);
    assert!((blurred.iter().sum::<f32>() - 1.0).abs() < 1e-5);

    // Closing fills a 1-pixel gap between two blocks.
    let mut bridge = [0.0_f32; 49];
    for y in 2..5 {
        for x in 1..6 {
            if x != 3 {
                bridge[y * 7 + x] = 1.0;
            }
        }
    }
    let closed = values(morphological_closing(mask_7x7(&bridge, &device), 1));
    assert_eq!(closed[3 * 7 + 3], 1.0);

    let config = PostprocessConfig::new()
        .with_threshold(0.4)
        .unwrap()
        .with_blur(GaussianBlur::new(3, 0.8).unwrap())
        .with_closing(1)
        .with_min_component_size(2)
        .with_fill_holes(true);
    let piped = postprocess_mask(apply_threshold(impulse, 0.5), &config).unwrap();
    assert_eq!(piped.dims(), [1, 1, 7, 7]);

    let resized = resize_tensor(piped, 14, 10, InterpolateMode::Bilinear);
    assert_eq!(resized.dims(), [1, 1, 14, 10]);
    assert_eq!(mask_to_u8(&resized).unwrap().dimensions(), (10, 14));
}

#[test]
fn mask_export_rounds_and_clamps() {
    let device = Device::flex();
    let mask = Tensor::<4>::from_floats([[[[-0.2, 0.0, 0.5, 1.0, 1.7]]]], &device);
    let gray = mask_to_u8(&mask).unwrap();
    assert_eq!(gray.into_raw(), vec![0, 0, 128, 255, 255]);
    assert!(
        mask_to_u8(&Tensor::zeros([1, 2, 2, 2], &device))
            .unwrap_err()
            .is_invalid_input()
    );
}

#[test]
fn configs_validate_their_parameters() {
    assert!(GaussianBlur::new(4, 1.0).unwrap_err().is_invalid_config());
    assert!(GaussianBlur::new(3, 0.0).unwrap_err().is_invalid_config());
    assert!(
        PostprocessConfig::new()
            .with_threshold(1.5)
            .unwrap_err()
            .is_invalid_config()
    );
    assert!(
        SegmenterConfig::new()
            .with_input_size([0, 32])
            .unwrap_err()
            .is_invalid_config()
    );
    // Zero radii and trivial sizes disable a step instead of failing.
    let config = PostprocessConfig::new()
        .with_opening(0)
        .with_min_component_size(1);
    assert_eq!(config, PostprocessConfig::new());
}

#[test]
fn segmenter_end_to_end_preserves_original_dimensions_and_outputs() {
    let device = Device::flex();
    let model = tiny_model(&device);
    let config = SegmenterConfig::new()
        .with_input_size([32, 32])
        .unwrap()
        .with_foreground_refinement(NonZeroUsize::new(2));
    let segmenter = Segmenter::new(&model, config);

    let img = DynamicImage::ImageRgb8(RgbImage::from_fn(20, 28, |x, y| {
        let channel = |v: u32| u8::try_from(v).unwrap();
        Rgb([channel(x * 10), channel(y * 8), 128])
    }));

    let output = segmenter.segment_image(img.clone(), &device).unwrap();
    assert_eq!(output.original_size(), [28, 20]);
    assert_eq!(output.mask().dims(), [1, 1, 28, 20]);
    assert!(output.foreground().is_some());
    assert_eq!(output.to_rgba_image().unwrap().dimensions(), (20, 28));
    assert_eq!(output.to_mask_image().unwrap().dimensions(), (20, 28));

    let dir = TempDir::new().unwrap();
    let file_path = dir.path().join("sample.png");
    img.save(&file_path).unwrap();
    let from_file = segmenter.segment_file(&file_path, &device).unwrap();
    let bytes = std::fs::read(&file_path).unwrap();
    let from_bytes = segmenter.segment_bytes(&bytes, &device).unwrap();
    let from_reader = segmenter
        .segment_reader(Cursor::new(&bytes), Some(ImageFormat::Png), &device)
        .unwrap();
    for other in [&from_bytes, &from_reader] {
        assert_eq!(
            values(other.mask().clone()),
            values(from_file.mask().clone())
        );
    }

    // A clone can serve requests from another thread and produces the same mask.
    let worker = segmenter.clone();
    let (worker_device, worker_path) = (device.clone(), file_path);
    let cloned = std::thread::spawn(move || worker.segment_file(&worker_path, &worker_device))
        .join()
        .unwrap()
        .unwrap();
    assert_eq!(
        values(cloned.mask().clone()),
        values(from_file.mask().clone())
    );

    let err = segmenter
        .segment_tensor(Tensor::zeros([1, 1, 8, 8], &device))
        .unwrap_err();
    assert!(err.is_invalid_input());
}

#[test]
fn segmenter_runs_autodiff_models_in_validation_mode() {
    let device = Device::flex().autodiff();
    let segmenter = Segmenter::new(
        &tiny_model(&device),
        SegmenterConfig::new()
            .with_input_size([32, 32])
            .unwrap()
            .with_foreground_refinement(None),
    );
    let image = Tensor::<4>::full([1, 3, 16, 16], 0.5, &device);
    let output = segmenter.segment_tensor(image).unwrap();
    assert!(!output.mask().is_autodiff());
}
