#!/usr/bin/env python3
"""
Generate test data for Swin Transformer Rust implementation by running the Python model.
This script should be run from the Python BiRefNet environment.
"""

import sys
from pathlib import Path

import numpy as np
import torch

# Add BiRefNet Python implementation to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "BiRefNet"))

from models.backbones.swin_v1 import swin_v1_t, swin_v1_s, swin_v1_b, swin_v1_l


def save_tensor_as_numpy(tensor: torch.Tensor, path: Path):
    """Save a PyTorch tensor as a numpy array file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np_array = tensor.detach().cpu().numpy()
    np.save(path, np_array)
    print(f"Saved tensor with shape {np_array.shape} to {path}")


def generate_swin_v1_t_test_data():
    """Generate test data for Swin-T model."""
    # Create model
    model = swin_v1_t()
    model.eval()

    # Create test input
    batch_size = 1
    channels = 3
    height = 224
    width = 224

    # Use a fixed seed for reproducibility
    torch.manual_seed(42)
    input_tensor = torch.randn(batch_size, channels, height, width)

    # Save input
    test_data_dir = Path(__file__).parent / "test_data"
    save_tensor_as_numpy(input_tensor, test_data_dir / "inputs" / "swin_v1_t_input.npy")

    # Run forward pass
    with torch.no_grad():
        outputs = model(input_tensor)

    # Save outputs
    for i, output in enumerate(outputs):
        save_tensor_as_numpy(
            output,
            test_data_dir / "outputs" / f"swin_v1_t_output_{i}.npy"
        )

    # Generate metadata
    metadata = {
        "model": "swin_v1_t",
        "input_shape": list(input_tensor.shape),
        "output_shapes": [list(out.shape) for out in outputs],
        "torch_version": torch.__version__,
        "seed": 42
    }

    import json
    with open(test_data_dir / "metadata" / "swin_v1_t_test.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {test_data_dir / 'metadata' / 'swin_v1_t_test.json'}")


def generate_swin_v1_s_test_data():
    """Generate test data for Swin-S model."""
    # Create model
    model = swin_v1_s()
    model.eval()

    # Create test input
    batch_size = 1
    channels = 3
    height = 224
    width = 224

    # Use a fixed seed for reproducibility
    torch.manual_seed(42)
    input_tensor = torch.randn(batch_size, channels, height, width)

    # Save input
    test_data_dir = Path(__file__).parent / "test_data"
    save_tensor_as_numpy(input_tensor, test_data_dir / "inputs" / "swin_v1_s_input.npy")

    # Run forward pass
    with torch.no_grad():
        outputs = model(input_tensor)

    # Save outputs
    for i, output in enumerate(outputs):
        save_tensor_as_numpy(
            output,
            test_data_dir / "outputs" / f"swin_v1_s_output_{i}.npy"
        )

    # Generate metadata
    metadata = {
        "model": "swin_v1_s",
        "input_shape": list(input_tensor.shape),
        "output_shapes": [list(out.shape) for out in outputs],
        "torch_version": torch.__version__,
        "seed": 42
    }

    import json
    with open(test_data_dir / "metadata" / "swin_v1_s_test.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {test_data_dir / 'metadata' / 'swin_v1_s_test.json'}")


def generate_patch_embed_test_data():
    """Generate test data for PatchEmbed module."""
    from models.backbones.swin_v1 import PatchEmbed

    # PatchEmbed parameters (matching Swin-T first stage)
    img_size = 224
    patch_size = 4
    in_chans = 3
    embed_dim = 96

    # Create patch embedding module
    patch_embed = PatchEmbed(
        patch_size=patch_size,
        in_channels=in_chans,
        embed_dim=embed_dim
    )
    patch_embed.eval()

    # Create test input
    batch_size = 1

    torch.manual_seed(42)
    input_tensor = torch.randn(batch_size, in_chans, img_size, img_size)

    # Save input
    test_data_dir = Path(__file__).parent / "test_data"
    save_tensor_as_numpy(input_tensor, test_data_dir / "inputs" / "patch_embed_input.npy")

    # Run forward pass
    with torch.no_grad():
        output = patch_embed(input_tensor)

    # Save output
    save_tensor_as_numpy(output, test_data_dir / "outputs" / "patch_embed_output.npy")

    # Save metadata
    metadata = {
        "module": "PatchEmbed",
        "img_size": img_size,
        "patch_size": patch_size,
        "in_chans": in_chans,
        "embed_dim": embed_dim,
        "input_shape": list(input_tensor.shape),
        "output_shape": list(output.shape),
        "seed": 42
    }

    import json
    metadata_path = test_data_dir / "metadata" / "patch_embed_test.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")


def generate_window_attention_test_data():
    """Generate test data for WindowAttention module."""
    from models.backbones.swin_v1 import WindowAttention

    # WindowAttention parameters (matching Swin-T)
    dim = 96
    window_size = (7, 7)  # Python implementation expects tuple
    num_heads = 3
    qkv_bias = True
    qk_scale = None
    attn_drop = 0.0
    proj_drop = 0.0

    # Create window attention module
    window_attn = WindowAttention(
        dim=dim,
        window_size=window_size,
        num_heads=num_heads,
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        attn_drop=attn_drop,
        proj_drop=proj_drop
    )
    window_attn.eval()

    # Create test input (B*num_windows, window_size*window_size, C)
    batch_size = 1
    num_windows = 64  # (224/7)^2 for 7x7 windows
    window_tokens = 49  # 7*7

    torch.manual_seed(42)
    input_tensor = torch.randn(batch_size * num_windows, window_tokens, dim)

    # Save input
    test_data_dir = Path(__file__).parent / "test_data"
    save_tensor_as_numpy(input_tensor, test_data_dir / "inputs" / "window_attention_input.npy")

    # Run forward pass
    with torch.no_grad():
        output = window_attn(input_tensor)

    # Save output
    save_tensor_as_numpy(output, test_data_dir / "outputs" / "window_attention_output.npy")

    # Save metadata
    metadata = {
        "module": "WindowAttention",
        "dim": dim,
        "window_size": list(window_size),
        "num_heads": num_heads,
        "input_shape": list(input_tensor.shape),
        "output_shape": list(output.shape),
        "seed": 42
    }

    import json
    metadata_path = test_data_dir / "metadata" / "window_attention_test.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    print("Generating Swin Transformer test data...")

    # Check if we're in the right environment
    try:
        from models.backbones.swin_v1 import swin_v1_t
    except ImportError:
        print(
            "Error: Cannot import Swin Transformer from BiRefNet. Make sure you run this script from the BiRefNet Python environment.")
        sys.exit(1)

    # Generate test data
    generate_swin_v1_t_test_data()
    generate_swin_v1_s_test_data()
    generate_patch_embed_test_data()
    generate_window_attention_test_data()

    print("Test data generation complete!")
    print("You can now run the Rust tests with: cargo test --features ndarray")
