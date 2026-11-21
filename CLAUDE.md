# BiRefNet-Burn

Rust workspace (`crates/*`, 11 crates, `resolver = "3"`, `edition = "2024"`, Rust >= 1.95) implementing BiRefNet on the [Burn](https://burn.dev) deep learning framework (`burn = "=0.22.1"`).

## Commands

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --features birefnet/train,birefnet-model/train,birefnet-cli/train,birefnet-util/image-io,birefnet-weights/hub,birefnet-weights/test-util -- -D warnings
cargo test -p <crate>                          # library tests run on Burn's `flex` (pure-Rust CPU) backend
cargo test -p birefnet-model --features train  # TrainStep/InferenceStep tests
cargo test --workspace --features birefnet/train,birefnet-model/train,birefnet-cli/train,birefnet-util/image-io,birefnet-weights/hub,birefnet-weights/test-util
cargo build --release -p birefnet-cli --features wgpu
cargo run --release -p birefnet-cli -- info
```

- Burn 0.22 picks the backend at run time from a `Device` value. `birefnet` is a pure library umbrella crate without CLI dependencies; `crates/birefnet-cli` owns the `birefnet` binary and exposes runtime backend features (`flex` default, `cpu`, `wgpu`, `vulkan`, `metal`, `cuda`, `rocm`, plus opt-in `fusion` = fusion + autotune for CubeCL backends). `--device` and `--precision` configure the runtime device (`crates/birefnet/src/device.rs`).
- Library crates depend on `burn` with `default-features = false, features = ["std"]` (workspace dependency) plus only the API features they need (`train`, `pytorch`, `safetensors`). They never enable a backend; their tests get `flex` (+ `autodiff` where required) from `[dev-dependencies]`.
- Heavy optional subsystems stay behind additive features: `birefnet-util` gates `image` decoding/encoding behind `image-io`; `birefnet-weights` gates Hugging Face Hub downloads behind `hub` (and `MockFetcher` behind `test-util`); `birefnet-model` and `birefnet` gate training support behind `train`.
- Prefer `-p <crate>` in cloud sessions: GPU features (`cuda`, `rocm`, `metal`) need vendor toolchains that are not available there, and `cpu`/`cuda` download an LLVM bundle at build time.
- Dependencies build at `opt-level = 2` (`image` at `opt-level = 3`, build-overrides at `opt-level = 1`) in the dev profile so tests run real kernels on Flex while keeping incremental rebuilds fast.

## Burn 0.22 conventions

- No backend generics: `Tensor<D, K>`, non-generic `#[derive(Module, Debug)]` structs (no `#[derive(Clone)]`), `&Device` arguments. Never call `Device::default()` inside library code; take the device from the caller or from an input tensor.
- Training: build the model on `device.autodiff()`; `TrainStep`/`InferenceStep`; `SupervisedTraining` + `Learner::new`; check `LearningResult::error`. Stochastic layers use a `Param<Flag>` training flag and only act on autodiff inputs (see `DropPath`).
- Weights: `birefnet-weights` loads Burnpack (`.bpk` via `Module::save_file` / `ModuleRecord`), SafeTensors (with upstream PyTorch key remapping), and PyTorch checkpoints via `burn::store`. The legacy recorder formats (`.mpk`, `.bin`) cannot be read.
- Never rename `#[derive(Module)]` struct fields in `birefnet-backbones` or `birefnet-model`: pretrained checkpoint keys and SafeTensors remappings bind directly to Burn module field names.
- Reading values back: `into_scalar::<T>()`, converting readers (`iter::<T>()`, `try_to_vec_as::<T>()`). Int tensors are `I32` on Flex/CubeCL, so never assume `i64` storage.
- Use `squeeze_dim`/`squeeze_dims` instead of bare `squeeze()`, which drops every size-1 axis (e.g. a batch of 1).
- Do not mix crates.io `burn` with git-pinned Burn dependencies (two copies of Burn link silently; check with `cargo tree -d`).

## Cloud environment

- `.claude/hooks/session-start.sh` (SessionStart, remote only) checks `rustc >= 1.95` (required by Burn 0.22), installs `rustfmt`/`clippy`, and runs `cargo fetch`.
- Network access: the default "Trusted" level covers `crates.io` and `github.com` (needed for Burn's LLVM bundle download). Pretrained-weight downloads via `hf-hub` need `huggingface.co` added under Allowed domains in the environment settings; offline tests use `MockFetcher`.
- No GPU in cloud sessions: validate with `flex`/`cpu`/`wgpu` (software Vulkan) only.
