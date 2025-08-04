#!/usr/bin/env python3
"""
Generate test data for PVT v2 Rust implementation by running the Python model.
This script should be run from the Python BiRefNet environment.
"""

import sys
from pathlib import Path

import numpy as np
import torch

# Add BiRefNet Python implementation to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "BiRefNet"))

from models.backbones.pvt_v2 import pvt_v2_b2


def save_tensor_as_numpy(tensor: torch.Tensor, path: Path):
    """Save a PyTorch tensor as a numpy array file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np_array = tensor.detach().cpu().numpy()
    np.save(path, np_array)
    print(f"Saved tensor with shape {np_array.shape} to {path}")


def generate_pvt_v2_b2_test_data():
    """Generate test data for PVT v2 B2 model."""
    # Create model
    model = pvt_v2_b2(pretrained=False)
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
    save_tensor_as_numpy(input_tensor, test_data_dir / "inputs" / "pvt_v2_b2_input.npy")

    # Run forward pass
    with torch.no_grad():
        outputs = model(input_tensor)

    # Save outputs
    for i, output in enumerate(outputs):
        save_tensor_as_numpy(
            output,
            test_data_dir / "outputs" / f"pvt_v2_b2_output_{i}.npy"
        )

    # Generate metadata
    metadata = {
        "model": "pvt_v2_b2",
        "input_shape": list(input_tensor.shape),
        "output_shapes": [list(out.shape) for out in outputs],
        "torch_version": torch.__version__,
        "seed": 42
    }

    import json
    with open(test_data_dir / "metadata" / "pvt_v2_b2_test.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {test_data_dir / 'metadata' / 'pvt_v2_b2_test.json'}")


def generate_block_test_data():
    """Generate test data for Block module."""
    from models.backbones.pvt_v2 import Block

    # Block module parameters (matching PVT-B2 first stage)
    dim = 64
    num_heads = 1
    mlp_ratio = 8.0
    qkv_bias = True
    qk_scale = None
    drop = 0.0
    attn_drop = 0.0
    drop_path = 0.0
    sr_ratio = 8

    # Create block module
    block = Block(
        dim=dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        drop=drop,
        attn_drop=attn_drop,
        drop_path=drop_path,
        sr_ratio=sr_ratio
    )
    block.eval()

    # Create test input (B, N, C format where N = H*W)
    batch_size = 1
    height = 56
    width = 56
    seq_len = height * width

    torch.manual_seed(42)
    input_tensor = torch.randn(batch_size, seq_len, dim)

    # Reshape to B, C, H, W format for consistency with Rust implementation
    input_4d = input_tensor.transpose(1, 2).reshape(batch_size, dim, height, width)

    # Save input
    test_data_dir = Path(__file__).parent / "test_data"
    save_tensor_as_numpy(input_4d, test_data_dir / "inputs" / "block_input.npy")

    # Run forward pass
    with torch.no_grad():
        output = block(input_tensor, height, width)

    # Reshape output to B, C, H, W format
    output_4d = output.transpose(1, 2).reshape(batch_size, dim, height, width)

    # Save output
    save_tensor_as_numpy(output_4d, test_data_dir / "outputs" / "block_output.npy")

    # Save metadata
    metadata = {
        "module": "Block",
        "dim": dim,
        "num_heads": num_heads,
        "mlp_ratio": mlp_ratio,
        "sr_ratio": sr_ratio,
        "input_shape": list(input_4d.shape),
        "output_shape": list(output_4d.shape),
        "seed": 42
    }

    import json
    metadata_path = test_data_dir / "metadata" / "block_test.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")


def generate_patch_embed_test_data():
    """Generate test data for OverlapPatchEmbed module."""
    from models.backbones.pvt_v2 import OverlapPatchEmbed

    # OverlapPatchEmbed parameters (matching PVT-B2 first stage)
    img_size = 224
    patch_size = 7
    stride = 4
    in_chans = 3
    embed_dim = 64

    # Create patch embedding module
    patch_embed = OverlapPatchEmbed(
        img_size=img_size,
        patch_size=patch_size,
        stride=stride,
        in_chans=in_chans,
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
        output, h, w = patch_embed(input_tensor)

    # Save output
    save_tensor_as_numpy(output, test_data_dir / "outputs" / "patch_embed_output.npy")

    # Save metadata
    metadata = {
        "module": "OverlapPatchEmbed",
        "img_size": img_size,
        "patch_size": patch_size,
        "stride": stride,
        "in_chans": in_chans,
        "embed_dim": embed_dim,
        "input_shape": list(input_tensor.shape),
        "output_shape": list(output.shape),
        "output_h": h,
        "output_w": w,
        "seed": 42
    }

    import json
    metadata_path = test_data_dir / "metadata" / "patch_embed_test.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    print("Generating PVT v2 test data...")

    # Check if we're in the right environment
    try:
        import models.backbones.pvt_v2
    except ImportError:
        print(
            "Error: Cannot import PVT v2 from BiRefNet. Make sure you run this script from the BiRefNet Python environment.")
        sys.exit(1)

    # Generate test data
    generate_pvt_v2_b2_test_data()
    generate_block_test_data()
    generate_patch_embed_test_data()

    print("Test data generation complete!")
    print("You can now run the Rust tests with: cargo test --features ndarray")
