#!/bin/bash
# SessionStart hook for Claude Code cloud sessions.
# Prepares the Rust toolchain and prefetches crates so build/test/lint work right away.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-$(pwd)}"

# Burn 0.22 requires Rust >= 1.95 (edition 2024, resolver 3).
MIN_RUST="1.95.0"
current="$(rustc --version | awk '{print $2}')"
if [ "$(printf '%s\n%s\n' "$MIN_RUST" "$current" | sort -V | head -n1)" != "$MIN_RUST" ]; then
  echo "rustc $current < $MIN_RUST; updating stable toolchain" >&2
  rustup update stable --no-self-update
fi

# Needed for `cargo fmt` / `cargo clippy` (no-op if already installed).
rustup component add rustfmt clippy >/dev/null 2>&1 || true

# Prefetch dependencies (idempotent; the container state is cached afterwards).
cargo fetch --locked 2>/dev/null || cargo fetch

# Burn's `cpu`/`cuda` features (tracel-llvm-bundler) download an LLVM bundle at build time.
# Its build script uses reqwest without the sandbox proxy CA, so the download fails in cloud
# sessions. Pre-seed its cache with curl (which honors the proxy CA) instead.
LLVM_VERSION="$(ls ~/.cargo/registry/src/*/ 2>/dev/null | sed -n 's/^tracel-llvm-bundler-//p' | sort -V | tail -n1)"
if [ -n "$LLVM_VERSION" ] && [ "$(uname -m)" = "x86_64" ]; then
  cache="$HOME/.cache/tracel"
  base="https://github.com/tracel-ai/tracel-llvm/releases/download/v${LLVM_VERSION}"
  mkdir -p "$cache"
  for f in linux-x64.checksums.json linux-x64.tar.xz; do
    [ -s "$cache/tracel-llvm-${LLVM_VERSION}-$f" ] || \
      curl -fsSL --retry 3 -o "$cache/tracel-llvm-${LLVM_VERSION}-$f" "$base/$f" || \
      echo "warning: could not prefetch $f; `cpu` backend builds may fail" >&2
  done
fi

if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  {
    echo 'export CARGO_TERM_COLOR=never'
    echo 'export CARGO_INCREMENTAL=0'
    echo 'export RUST_MIN_STACK=67108864'  # deep Burn graphs overflow the 2 MiB test thread stack
    echo 'export TRACEL_LLVM_BUNDLER_SKIP_CHECKSUM_DOWNLOAD=1'
  } >> "$CLAUDE_ENV_FILE"
fi
