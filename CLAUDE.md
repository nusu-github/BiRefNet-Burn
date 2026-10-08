# BiRefNet-Burn

Rust workspace (`crates/*`) implementing BiRefNet on the [Burn](https://burn.dev) deep learning framework.

## Commands

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test -p <crate>            # per-crate tests use Burn's `cpu` backend (LLVM JIT)
cargo build --release -p birefnet --features wgpu --no-default-features
```

- The top-level `birefnet` crate selects the backend via features (`ndarray` default, `wgpu`, `vulkan`, `cuda`, `rocm`, `metal`). Library crates must not enable a backend.
- Prefer `-p <crate>` over workspace-wide `--all-features` in cloud sessions: GPU features (`cuda`, `rocm`, `metal`) need vendor toolchains that are not available there.

## Cloud environment

- `.claude/hooks/session-start.sh` (SessionStart, remote only) checks rustc >= 1.95 (required by Burn 0.22), installs `rustfmt`/`clippy`, and runs `cargo fetch`.
- Network access: the default "Trusted" level covers crates.io and github.com (needed for Burn's LLVM bundle download). Pretrained-weight downloads via `hf-hub` need `huggingface.co` added under Allowed domains in the environment settings; tests should not depend on it.
- No GPU in cloud sessions: validate with `cpu`/`ndarray`/`wgpu` (software Vulkan) only.

## Burn 0.22 migration notes

The workspace currently pins `burn = 0.21`. Burn 0.22 removes the backend type parameter (`Tensor<B, D>` -> `Tensor<D>`, runtime `Device`), `Autodiff<B>` (-> `device.autodiff()`), recorders (-> burnpack `.bpk`), and `LearnerBuilder` (-> `SupervisedTraining`). Do not mix crates.io `burn` with git-pinned Burn dependencies (two copies of Burn link silently).

## Known baseline (before the 0.22 upgrade)

`cargo test -p birefnet-loss` on Burn 0.21 has pre-existing failures (e.g. `Squeeze` shape errors in `contour`/`iou`/`ssim` tests); they are not caused by the environment.
