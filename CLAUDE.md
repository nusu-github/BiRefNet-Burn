# BiRefNet-Burn

Rust workspace (`crates/*`) implementing BiRefNet on the [Burn](https://burn.dev) deep learning framework (Burn 0.22, Rust >= 1.95).

## Commands

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --features birefnet/train,birefnet-model/train -- -D warnings
cargo test -p <crate>                          # library tests run on Burn's `flex` (pure-Rust CPU) backend
cargo test -p birefnet-model --features train  # TrainStep/InferenceStep tests
cargo build --release -p birefnet --features wgpu
```

- Burn 0.22 picks the backend at run time from a `Device` value. The top-level `birefnet` binary owns every backend/performance feature (`flex` default, `cpu`, `wgpu`, `vulkan`, `metal`, `cuda`, `rocm`, plus `fusion` = fusion + autotune); `--device` selects one at run time (`crates/birefnet/src/device.rs`).
- Library crates depend on `burn` with `default-features = false, features = ["std"]` (workspace dependency) plus only the API features they use (`train`, `pytorch`, `safetensors`). They must never enable a backend; their tests get `flex` (+ `autodiff`) from `[dev-dependencies]`.
- Prefer `-p <crate>` in cloud sessions: GPU features (`cuda`, `rocm`, `metal`) need vendor toolchains that are not available there, and `cpu`/`cuda` download an LLVM bundle at build time.
- Dependencies build at `opt-level = 2` in the dev profile (tests run real kernels on Flex). Model-sized tests (e.g. `birefnet-backbones`) still take minutes.

## Burn 0.22 conventions

- No backend generics: `Tensor<D, K>`, non-generic `#[derive(Module, Debug)]` structs (no `#[derive(Clone)]`), `&Device` arguments. Never call `Device::default()` inside library code; take the device from the caller or from an input tensor.
- Training: build the model on `device.autodiff()`; `TrainStep`/`InferenceStep`; `SupervisedTraining` + `Learner::new`; check `LearningResult::error`. Stochastic layers use a `Param<Flag>` training flag and only act on autodiff inputs (see `DropPath`).
- Weights: `Module::save_file` / `ModuleRecord` (Burnpack `.bpk`); PyTorch/SafeTensors through `burn::store`. The old recorder formats (`.mpk`, `.bin`) cannot be read.
- Reading values back: `into_scalar::<T>()`, converting readers (`iter::<T>()`, `try_to_vec_as::<T>()`). Int tensors are I32 on Flex/CubeCL, so never assume `i64` storage.
- Use `squeeze_dim`/`squeeze_dims` instead of bare `squeeze()`, which drops every size-1 axis (e.g. a batch of 1).
- Do not mix crates.io `burn` with git-pinned Burn dependencies (two copies of Burn link silently; check with `cargo tree -d`).

## Cloud environment

- `.claude/hooks/session-start.sh` (SessionStart, remote only) checks rustc >= 1.95 (required by Burn 0.22), installs `rustfmt`/`clippy`, and runs `cargo fetch`.
- Network access: the default "Trusted" level covers crates.io and github.com (needed for Burn's LLVM bundle download). Pretrained-weight downloads via `hf-hub` need `huggingface.co` added under Allowed domains in the environment settings; tests should not depend on it.
- No GPU in cloud sessions: validate with `flex`/`cpu`/`wgpu` (software Vulkan) only.
