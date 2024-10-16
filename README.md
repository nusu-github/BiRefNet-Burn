# BiRefNet-Burn

![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)

The objective of the BiRefNet-Burn project is to adapt the cutting-edge BiRefNet model, which has been developed within the PyTorch framework, for utilisation within the Burn framework.

> [!NOTE]
> At the moment, it works with the combination of the BiRefNet Swin v1 backbone and the Burn WebGPU backend.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

BiRefNet is a cutting-edge model designed for high-resolution dichotomous image segmentation, as detailed in the paper "Bilateral Reference for High-Resolution Dichotomous Image Segmentation" [1]. This project employs the Burn framework, a unified deep learning framework written in Rust, with the objective of reimplementing BiRefNet in order to achieve enhanced performance and efficiency.

- **BiRefNet**: [ZhengPeng7/BiRefNet](https://github.com/ZhengPeng7/BiRefNet)
- **Burn**: [tracel-ai/burn](https://github.com/tracel-ai/burn)

## Installation

### Prerequisites

- Rust (latest stable version)

### Steps

1. Install Rust from [https://www.rust-lang.org/tools/install](https://www.rust-lang.org/tools/install)
2. Clone the repository:
   ```bash
   git clone https://github.com/nusu-github/BiRefNet-Burn.git
   cd BiRefNet-Burn
   ```

3. Build the project:
   ```bash
   cargo build --release
   ```

## Usage

To run inference using the Burn-ported BiRefNet model:

```bash
cargo run --release -- --help
```

This will display available options for running inference on your images.

## License

This project is dual-licensed under both the MIT and Apache-2.0 licenses. You may choose either license when using this project.

- [MIT License](LICENSE-MIT)
- [Apache-2.0 License](LICENSE-APACHE)

## Acknowledgements

- [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) by ZhengPeng7, licensed under MIT
- [Burn](https://github.com/tracel-ai/burn) by Tracel.ai

## References

[1] Zheng, P., “Bilateral Reference for High-Resolution Dichotomous Image Segmentation”, <i>arXiv e-prints</i>, Art. no. arXiv:2401.03407, 2024. doi:10.48550/arXiv.2401.03407.

[2] Liu, Z., “Swin Transformer: Hierarchical Vision Transformer using Shifted Windows”, <i>arXiv e-prints</i>, Art. no. arXiv:2103.14030, 2021. doi:10.48550/arXiv.2103.14030.

---

For any questions or support, please [open an issue](https://github.com/nusu-github/BiRefNet-Burn/issues).