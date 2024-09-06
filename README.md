# BiRefNet-Burn

![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)

BiRefNet-Burn is an ambitious project that ports the cutting-edge BiRefNet model from PyTorch to the Burn framework, enabling efficient inference for high-resolution dichotomous image segmentation tasks.

> [!NOTE]
> We'll start with the inference implementation of the Swin v1 backbone, which is a key component of the original BiRefNet architecture.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

BiRefNet is a state-of-the-art model designed for high-resolution dichotomous image segmentation, introduced in the paper "Bilateral Reference for High-Resolution Dichotomous Image Segmentation" [1]. This project leverages the Burn framework, a unified deep learning framework in Rust, to reimplement BiRefNet, aiming for improved performance and efficiency.

- **BiRefNet**: [ZhengPeng7/BiRefNet](https://github.com/ZhengPeng7/BiRefNet)
- **Burn**: [tracel-ai/burn](https://github.com/tracel-ai/burn)

## Features

- Rust implementation of BiRefNet using the Burn framework
- Efficient inference for high-resolution segmentation tasks
- Maintained accuracy and performance consistency with the original PyTorch implementation
- Utilizes Swin Transformer v1 as the backbone, known for its hierarchical structure and shifted windows [2]

## Installation

### Prerequisites

- Rust (latest stable version)
- Burn (follow installation instructions from the [Burn repository](https://github.com/tracel-ai/burn))

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

For detailed Burn setup instructions, refer to the [Burn documentation](https://github.com/tracel-ai/burn#getting-started).

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