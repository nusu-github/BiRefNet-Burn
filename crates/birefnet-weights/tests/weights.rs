//! Offline integration tests for model registry, fetchers, and checkpoint loading.

use std::path::{Path, PathBuf};

use birefnet_model::{
    Backbone, BackboneConfig, BiRefNet, BiRefNetConfig, DecoderAttention, DecoderConfig,
    InterpolationStrategy, ModelConfig, MultiScaleInput, Refine, RefineConfig, SqueezeBlock,
    TaskConfig,
};
use birefnet_weights::{
    BiRefNetWeightLoading, Fetcher, MODEL_SPECS, ManagedModel, ModelName, WeightError,
    WeightFormat, WeightSource, get_model_spec, list_available_models, load_weights_from_path,
};
use burn::{
    module::{Module, Param, ParamId},
    nn::{Linear, LinearConfig},
    prelude::*,
    store::{BurnToPyTorchAdapter, ModuleSnapshot, SafetensorsStore},
    tensor::{DType, Distribution, FloatDType, IntDType},
};
use tempfile::TempDir;

#[derive(Debug)]
struct StubFetcher {
    resolved: Option<PathBuf>,
}

impl Fetcher for StubFetcher {
    fn fetch(&self, _repo_id: &str, _filename: &str) -> Result<PathBuf, WeightError> {
        self.resolved
            .clone()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "missing").into())
    }

    fn is_cached(&self, _repo_id: &str, _filename: &str) -> bool {
        self.resolved.as_ref().is_some_and(|path| path.exists())
    }
}

fn tiny_birefnet_config() -> BiRefNetConfig {
    let model = ModelConfig::new(InterpolationStrategy::Nearest)
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
    BiRefNetConfig::new(model)
}

#[test]
fn catalog_contains_all_15_valid_presets() {
    let names = list_available_models();
    assert_eq!(names.len(), 15);
    assert_eq!(MODEL_SPECS.len(), 15);

    for name in names {
        let spec = get_model_spec(name).expect("listed model must resolve");
        assert_eq!(spec.name, name);
        assert!(spec.repo_id().starts_with("ZhengPeng7/"));

        let managed = ManagedModel::from_pretrained(name).expect("preset must construct");
        assert_eq!(managed.name().as_str(), name);
        assert_eq!(managed.default_resolution(), Some(spec.default_resolution));
        assert_eq!(
            managed.supports_dynamic_resolution(),
            spec.supports_dynamic_resolution
        );
        assert!(matches!(
            managed.weight_source(),
            WeightSource::Remote { repo_id, filename }
                if repo_id == &spec.repo_id() && filename == "model.safetensors"
        ));
    }

    assert!(
        get_model_spec("General-dynamic")
            .unwrap()
            .supports_dynamic_resolution
    );
    assert!(
        !get_model_spec("General")
            .unwrap()
            .supports_dynamic_resolution
    );
}

#[test]
fn unknown_model_and_missing_local_file_report_expected_kinds() {
    let unknown = ManagedModel::from_pretrained("NonExistentModel").unwrap_err();
    assert!(unknown.is_unknown_model());

    let dir = TempDir::new().unwrap();
    let missing_path = dir.path().join("does_not_exist.bpk");
    let local = ManagedModel::from_local("custom", &missing_path, Some(tiny_birefnet_config()));
    let fetcher = StubFetcher { resolved: None };
    assert!(!local.is_available_with(&fetcher));
    assert!(
        local
            .resolve_path_with(&fetcher)
            .unwrap_err()
            .is_not_found()
    );

    let device = Device::flex();
    let load_err = BiRefNet::from_managed_model_with(&local, &fetcher, &device).unwrap_err();
    assert!(load_err.is_not_found());
}

#[test]
fn weight_format_detects_extensions_case_insensitively() {
    assert_eq!(
        WeightFormat::from_path(Path::new("weights.safetensors")),
        WeightFormat::SafeTensors
    );
    assert_eq!(
        WeightFormat::from_path(Path::new("weights.SAFETENSORS")),
        WeightFormat::SafeTensors
    );
    assert_eq!(
        WeightFormat::from_path(Path::new("weights.pth")),
        WeightFormat::PyTorch
    );
    assert_eq!(
        WeightFormat::from_path(Path::new("weights.PT")),
        WeightFormat::PyTorch
    );
    assert_eq!(
        WeightFormat::from_path(Path::new("weights.bpk")),
        WeightFormat::Burnpack
    );
    assert_eq!(
        WeightFormat::from_path(Path::new("weights.bin")),
        WeightFormat::Auto
    );
}

#[test]
fn burnpack_round_trip_preserves_model_outputs() {
    // Burn `Module` visitor/serialization traverses deep nested structs; match Linux's 8 MiB stack
    // on Windows unoptimized test threads.
    std::thread::Builder::new()
        .name(String::from("burnpack-round-trip"))
        .stack_size(8 * 1024 * 1024)
        .spawn(|| {
            let device = Device::flex();
            let cfg = tiny_birefnet_config();
            let original = cfg.init(&device).unwrap();

            let dir = TempDir::new().unwrap();
            let save_stem = dir.path().join("tiny_model");
            original.clone().save_file(&save_stem).unwrap();
            let bpk_path = dir.path().join("tiny_model.bpk");
            assert!(bpk_path.exists());

            let input =
                Tensor::<4>::random([1, 3, 32, 32], Distribution::Uniform(0.0, 1.0), &device);
            let expected_out = original.forward(input.clone()).unwrap();

            // Load via explicit local ManagedModel with Auto format (no extension -> exercises fallback).
            let no_ext_path = dir.path().join("tiny_model_no_ext");
            std::fs::copy(&bpk_path, &no_ext_path).unwrap();
            let local =
                ManagedModel::from_local(ModelName::from("tiny"), &no_ext_path, Some(cfg.clone()));
            let fetcher = StubFetcher { resolved: None };
            assert!(local.is_available_with(&fetcher));

            let reloaded = BiRefNet::from_managed_model_with(&local, &fetcher, &device).unwrap();
            let reloaded_out = reloaded.forward(input.clone()).unwrap();
            let max_diff = (expected_out.clone() - reloaded_out)
                .abs()
                .max()
                .into_scalar::<f64>();
            assert!(max_diff < 1e-6, "max_diff={max_diff}");

            // Also load via remote source resolved through StubFetcher to exact `.bpk` file.
            let remote = ManagedModel::new(
                ModelName::from("tiny-remote"),
                Some(cfg),
                WeightSource::Remote {
                    repo_id: String::from("ZhengPeng7/BiRefNet"),
                    filename: String::from("model.bpk"),
                },
            );
            let remote_fetcher = StubFetcher {
                resolved: Some(bpk_path),
            };
            assert!(remote.is_available_with(&remote_fetcher));
            let from_remote =
                BiRefNet::from_managed_model_with(&remote, &remote_fetcher, &device).unwrap();
            let remote_diff = (expected_out - from_remote.forward(input).unwrap())
                .abs()
                .max()
                .into_scalar::<f64>();
            assert!(remote_diff < 1e-6, "remote_diff={remote_diff}");
        })
        .unwrap()
        .join()
        .unwrap();
}

#[cfg(feature = "test-util")]
#[test]
fn mock_fetcher_resolves_registered_artifacts() {
    use birefnet_weights::MockFetcher;

    let dir = TempDir::new().unwrap();
    let device = Device::flex();
    let cfg = tiny_birefnet_config();
    let model = cfg.init(&device).unwrap();
    let bpk_stem = dir.path().join("cached");
    model.save_file(&bpk_stem).unwrap();
    let bpk_file = dir.path().join("cached.bpk");

    let fetcher =
        MockFetcher::new().with_file("ZhengPeng7/BiRefNet", "model.safetensors", &bpk_file);
    let mut managed = ManagedModel::from_pretrained("General").unwrap();
    // Override config for fast test execution while keeping remote WeightSource untouched.
    managed = ManagedModel::new(
        managed.name().clone(),
        Some(cfg),
        managed.weight_source().clone(),
    );

    assert!(managed.is_available_with(&fetcher));
    assert_eq!(managed.resolve_path_with(&fetcher).unwrap(), bpk_file);

    let missing_managed = ManagedModel::from_pretrained("DIS").unwrap();
    assert!(!missing_managed.is_available_with(&fetcher));
    assert!(
        missing_managed
            .resolve_path_with(&fetcher)
            .unwrap_err()
            .is_not_found()
    );
}

#[derive(Module, Debug)]
struct SafetensorsMixedDTypeModule {
    linear: Linear,
    index: Param<Tensor<2, Int>>,
}

#[test]
fn safetensors_loads_i64_int_buffer_and_casts_f16_weights() {
    let device = Device::flex();
    let dir = TempDir::new().unwrap();
    let st_path = dir.path().join("mixed.safetensors");

    let mut linear = LinearConfig::new(3, 2).with_bias(true).init(&device);
    linear.weight = linear.weight.map(|w| w.cast(FloatDType::F16));
    linear.bias = linear.bias.map(|b| b.map(|t| t.cast(FloatDType::F16)));

    let expected_indices = vec![4_i64, 3, 1, 0];
    let index_tensor =
        Tensor::<2, Int>::from_data(TensorData::new(expected_indices.clone(), [2, 2]), &device)
            .cast(IntDType::I64);
    assert_eq!(index_tensor.dtype(), DType::I64);

    let saved = SafetensorsMixedDTypeModule {
        linear: linear.clone(),
        index: Param::initialized(ParamId::new(), index_tensor),
    };
    let mut save_store =
        SafetensorsStore::from_file(&st_path).with_to_adapter(BurnToPyTorchAdapter);
    saved.save_into(&mut save_store).unwrap();

    let fresh = SafetensorsMixedDTypeModule {
        linear: LinearConfig::new(3, 2).with_bias(true).init(&device),
        index: Param::initialized(ParamId::new(), Tensor::<2, Int>::zeros([2, 2], &device)),
    };
    let loaded = load_weights_from_path(fresh, &st_path, &device).unwrap();

    assert_eq!(loaded.linear.weight.val().dtype(), DType::F32);
    assert!(loaded.index.val().dtype().is_int());

    let loaded_indices = loaded
        .index
        .val()
        .into_data()
        .try_to_vec_as::<i64>()
        .unwrap();
    assert_eq!(loaded_indices, expected_indices);

    let weight_diff = (linear.weight.val().cast(FloatDType::F32) - loaded.linear.weight.val())
        .abs()
        .max()
        .into_scalar::<f64>();
    assert!(weight_diff < 1e-3, "weight_diff={weight_diff}");
}
