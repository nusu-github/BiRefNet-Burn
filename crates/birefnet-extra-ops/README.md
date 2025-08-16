# birefnet-extra-ops

[![Crates.io](https://img.shields.io/crates/v/birefnet-extra-ops.svg)](https://crates.io/crates/birefnet-extra-ops)
[![Documentation](https://docs.rs/birefnet-extra-ops/badge.svg)](https://docs.rs/birefnet-extra-ops)

**Additional operations for the Burn deep learning framework**

## Implementation

This crate implements specialized operations extending Burn's functionality:

### Regularization Operations

- `DropPath`: Stochastic depth for training regularization
- Layer scaling for transformer training stability

### Mathematical Functions

- `erfinv`: Inverse error function implementation
- Advanced statistical operations

### Weight Initialization

- `trunc_normal`: Truncated normal distribution initialization
- Custom initialization schemes for modern architectures

### Utility Operations

- `Identity`: Pass-through layer for conditional architectures
- Custom activation functions

### Core Components

- DropPath with configurable schedules
- Mathematical function implementations
- Weight initialization utilities
- Identity and utility layers
- Efficient backend-agnostic implementations

## License

MIT OR Apache-2.0