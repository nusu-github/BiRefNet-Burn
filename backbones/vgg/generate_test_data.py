#!/usr/bin/env python3
"""
Generate test data for VGG Rust implementation by running the Python model.
This script should be run from the Python BiRefNet environment.
"""

import sys
from pathlib import Path

import numpy as np
import torch

# Add BiRefNet Python implementation to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "BiRefNet"))

from models.backbones.build_backbone import build_backbone


def save_tensor_as_numpy(tensor: torch.Tensor, path: Path):
    """Save a PyTorch tensor as a numpy array file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np_array = tensor.detach().cpu().numpy()
    np.save(path, np_array)
    print(f"Saved tensor with shape {np_array.shape} to {path}")


def generate_vgg16_test_data():
    """Generate test data for VGG16 model."""
    # Create VGG16 model using build_backbone
    model = build_backbone('vgg16', pretrained=False)
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
    save_tensor_as_numpy(input_tensor, test_data_dir / "inputs" / "vgg16_input.npy")

    # Run forward pass
    with torch.no_grad():
        # VGG model returns OrderedDict with conv1, conv2, conv3, conv4
        x = input_tensor
        outputs = {}
        for name, layer in model.named_children():
            x = layer(x)
            outputs[name] = x.clone()

    # Save outputs
    for stage_name, output in outputs.items():
        save_tensor_as_numpy(
            output,
            test_data_dir / "outputs" / f"vgg16_{stage_name}_output.npy"
        )

    # Generate metadata
    metadata = {
        "model": "vgg16",
        "input_shape": list(input_tensor.shape),
        "output_shapes": {name: list(out.shape) for name, out in outputs.items()},
        "torch_version": torch.__version__,
        "seed": 42,
        "stages": list(outputs.keys())
    }

    import json
    with open(test_data_dir / "metadata" / "vgg16_test.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {test_data_dir / 'metadata' / 'vgg16_test.json'}")


def generate_vgg16bn_test_data():
    """Generate test data for VGG16 with Batch Normalization model."""
    # Create VGG16BN model using build_backbone
    model = build_backbone('vgg16bn', pretrained=False)
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
    save_tensor_as_numpy(input_tensor, test_data_dir / "inputs" / "vgg16bn_input.npy")

    # Run forward pass
    with torch.no_grad():
        # VGG model returns OrderedDict with conv1, conv2, conv3, conv4
        x = input_tensor
        outputs = {}
        for name, layer in model.named_children():
            x = layer(x)
            outputs[name] = x.clone()

    # Save outputs
    for stage_name, output in outputs.items():
        save_tensor_as_numpy(
            output,
            test_data_dir / "outputs" / f"vgg16bn_{stage_name}_output.npy"
        )

    # Generate metadata
    metadata = {
        "model": "vgg16bn",
        "input_shape": list(input_tensor.shape),
        "output_shapes": {name: list(out.shape) for name, out in outputs.items()},
        "torch_version": torch.__version__,
        "seed": 42,
        "stages": list(outputs.keys())
    }

    import json
    with open(test_data_dir / "metadata" / "vgg16bn_test.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {test_data_dir / 'metadata' / 'vgg16bn_test.json'}")


def generate_vgg_feature_block_test_data():
    """Generate test data for individual VGG feature blocks."""
    from torchvision.models import vgg16

    # Get VGG16 features
    vgg_model = vgg16(weights=None)
    features = vgg_model.features

    # Extract conv1 block (first 10 layers)
    conv1_block = features[:10]
    conv1_block.eval()

    # Create test input
    batch_size = 1
    channels = 3
    height = 224
    width = 224

    torch.manual_seed(42)
    input_tensor = torch.randn(batch_size, channels, height, width)

    # Save input
    test_data_dir = Path(__file__).parent / "test_data"
    save_tensor_as_numpy(input_tensor, test_data_dir / "inputs" / "vgg_conv1_input.npy")

    # Run forward pass
    with torch.no_grad():
        output = conv1_block(input_tensor)

    # Save output
    save_tensor_as_numpy(output, test_data_dir / "outputs" / "vgg_conv1_output.npy")

    # Save metadata
    metadata = {
        "module": "VGGConv1Block",
        "input_shape": list(input_tensor.shape),
        "output_shape": list(output.shape),
        "seed": 42,
        "description": "First convolutional block of VGG16 (features[:10])"
    }

    import json
    metadata_path = test_data_dir / "metadata" / "vgg_conv1_test.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")


def test_model_structure():
    """Test and print the structure of VGG models."""
    print("=== VGG16 Structure ===")
    model = build_backbone('vgg16', pretrained=False)
    print(f"Model type: {type(model)}")
    for name, layer in model.named_children():
        print(f"{name}: {type(layer)} -> {len(list(layer.children()))} sublayers")

    print("\n=== VGG16BN Structure ===")
    model_bn = build_backbone('vgg16bn', pretrained=False)
    print(f"Model type: {type(model_bn)}")
    for name, layer in model_bn.named_children():
        print(f"{name}: {type(layer)} -> {len(list(layer.children()))} sublayers")


if __name__ == "__main__":
    print("Generating VGG test data...")

    # Check if we're in the right environment
    try:
        from models.backbones.build_backbone import build_backbone
    except ImportError:
        print(
            "Error: Cannot import build_backbone from BiRefNet. Make sure you run this script from the BiRefNet Python environment.")
        sys.exit(1)

    # Print model structure first
    test_model_structure()
    print()

    # Generate test data
    generate_vgg16_test_data()
    generate_vgg16bn_test_data()
    generate_vgg_feature_block_test_data()

    print("VGG test data generation complete!")
    print("You can now run the Rust tests with: cargo test")
