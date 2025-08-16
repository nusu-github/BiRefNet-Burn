# birefnet-util

[![Crates.io](https://img.shields.io/crates/v/birefnet-util.svg)](https://crates.io/crates/birefnet-util)
[![Documentation](https://docs.rs/birefnet-util/badge.svg)](https://docs.rs/birefnet-util)

**Utility functions and tools for BiRefNet models**

## Implementation

This crate implements essential utilities for BiRefNet:

### Image Processing

- `ImageUtils`: Image loading, resizing, format conversion
- Tensor conversion with normalization
- Preprocessing pipelines

### Weight Management

- PyTorch weight loading and conversion
- Burn model serialization and deserialization
- Checkpoint management

### Dataset Utilities

- Dataset scanning and validation
- File path management
- Data organization tools

### Planned Features (Not Yet Implemented)

- Result overlay and comparison
- Debug visualization tools
- Grid layout generation

### Configuration

- JSON/YAML configuration loading
- Configuration validation
- Parameter management

### Core Components

- Cross-format weight conversion
- Image processing pipelines
- Dataset preparation tools
- Debugging and profiling utilities
- Configuration management

## License

MIT OR Apache-2.0