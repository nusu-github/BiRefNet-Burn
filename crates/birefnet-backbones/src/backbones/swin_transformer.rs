//! # Swin Transformer v1 Implementation
//!
//! This module provides a Rust implementation of the Swin Transformer backbone using the Burn framework.
//! The Swin Transformer is a hierarchical vision transformer that uses shifted windows for efficient
//! self-attention computation.
//!
//! ## Architecture Overview
//!
//! The Swin Transformer consists of several key components:
//! - **PatchEmbed**: Converts input images into patch embeddings
//! - **SwinTransformerBlock**: Core transformer block with window-based multi-head self-attention
//! - **WindowAttention**: Multi-head self-attention with relative position bias
//! - **PatchMerging**: Downsampling layer that merges neighboring patches
//! - **BasicLayer**: A sequence of Swin transformer blocks at a particular resolution
//!
//! ## Key Features
//! - Hierarchical feature representation with 4 stages
//! - Window-based self-attention with shifted windows for cross-window connections
//! - Relative position bias for improved spatial understanding
//! - Configurable window sizes and model depths
//!
//! ## Model Variants
//! - **Swin-T**: Tiny model (96 dim, [2,2,6,2] depths, [3,6,12,24] heads)
//! - **Swin-S**: Small model (96 dim, [2,2,18,2] depths, [3,6,12,24] heads)  
//! - **Swin-B**: Base model (128 dim, [2,2,18,2] depths, [4,8,16,32] heads)
//! - **Swin-L**: Large model (192 dim, [2,2,18,2] depths, [6,12,24,48] heads)
//!
//! ## Reference
//! Based on "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
//! - Paper: https://arxiv.org/pdf/2103.14030
//! - Original PyTorch implementation: Microsoft Research

use birefnet_extra_ops::{trunc_normal, DropPath, DropPathConfig};
use burn::{
    module::Param,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        Dropout, DropoutConfig, Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig,
    },
    prelude::*,
    tensor::{
        activation::softmax,
        module::interpolate,
        ops::{InterpolateMode, InterpolateOptions},
    },
};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SwinTransformerError {
    #[error("Model initialization failed: {reason}")]
    ModelInitializationFailed { reason: String },
    #[error("Tensor operation failed: {operation}")]
    TensorOperationFailed { operation: String },
}

pub type SwinTransformerResult<T> = Result<T, SwinTransformerError>;

/// Configuration for a Multi-Layer Perceptron (MLP) used in Swin Transformer blocks.
///
/// The MLP consists of two linear transformations with a GELU activation function
/// and dropout applied between them. This follows the standard transformer FFN design.
///
/// # Architecture
/// ```text
/// Input -> Linear -> GELU -> Dropout -> Linear -> Dropout -> Output
/// ```
///
/// # Arguments
/// - `in_features`: Number of input features
/// - `hidden_features`: Number of hidden features (defaults to `in_features` if None)
/// - `out_features`: Number of output features (defaults to `in_features` if None)
/// - `drop`: Dropout probability applied after each linear layer
#[derive(Config, Debug)]
pub struct MlpConfig {
    in_features: usize,
    #[config(default = "None")]
    hidden_features: Option<usize>,
    #[config(default = "None")]
    out_features: Option<usize>,
    // act_layer:Module,
    #[config(default = "0.0")]
    drop: f64,
}

impl MlpConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Mlp<B> {
        let out_features = self.out_features.unwrap_or(self.in_features);
        let hidden_features = self.hidden_features.unwrap_or(self.in_features);
        let fc1 = LinearConfig::new(self.in_features, hidden_features).init(device);
        let act = Gelu::new();
        let fc2 = LinearConfig::new(hidden_features, out_features).init(device);
        let drop = DropoutConfig::new(self.drop).init();

        Mlp {
            fc1,
            act,
            fc2,
            drop,
        }
    }
}

/// Multi-Layer Perceptron (MLP) module for Swin Transformer blocks.
///
/// This is the feed-forward network component used in each Swin Transformer block.
/// It applies two linear transformations with GELU activation and dropout.
///
/// # Components
/// - `fc1`: First linear transformation (input -> hidden)
/// - `act`: GELU activation function
/// - `fc2`: Second linear transformation (hidden -> output)  
/// - `drop`: Dropout layer applied after both linear layers
#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    fc1: Linear<B>,
    act: Gelu,
    fc2: Linear<B>,
    drop: Dropout,
}

impl<B: Backend> Mlp<B> {
    /// Forward pass through the MLP.
    ///
    /// Applies the sequence: Linear -> GELU -> Dropout -> Linear -> Dropout
    ///
    /// # Arguments
    /// - `x`: Input tensor of shape `[batch_size, sequence_length, features]`
    ///
    /// # Returns
    /// Output tensor of shape `[batch_size, sequence_length, out_features]`
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.fc1.forward(x);
        let x = self.act.forward(x);
        let x = self.drop.forward(x);
        let x = self.fc2.forward(x);

        self.drop.forward(x)
    }
}

/// Create a 2D coordinate grid matching PyTorch's torch.meshgrid([coords_h, coords_w], indexing='ij').
/// PyTorch produces: coords shape [2, Wh, Ww] where coords[0] is height mesh, coords[1] is width mesh
/// Uses stable tensor operations (arange, reshape, repeat_dim, stack) instead of complex operations.
fn create_coordinate_grid<B: Backend>(
    height: usize,
    width: usize,
    device: &Device<B>,
) -> Tensor<B, 3> {
    // Create height coordinates (first dimension in PyTorch meshgrid)
    let h_coords = Tensor::arange(0..height as i64, device)
        .reshape([height, 1])
        .repeat_dim(1, width)
        .reshape([height, width]);
    // Create width coordinates (second dimension in PyTorch meshgrid)
    let w_coords = Tensor::arange(0..width as i64, device)
        .reshape([1, width])
        .repeat_dim(0, height)
        .reshape([height, width]);

    // Stack in [h, w] order with first dimension as stacking dim to match PyTorch [2, Wh, Ww]
    Tensor::stack(vec![h_coords.float(), w_coords.float()], 0)
}

/// Partitions input feature maps into non-overlapping windows.
///
/// This function divides the input tensor into windows of size `window_size x window_size`
/// for efficient window-based attention computation in Swin Transformer.
///
/// # Arguments
/// - `x`: Input tensor of shape `[batch_size, height, width, channels]`
/// - `window_size`: Size of each window (e.g., 7 for 7x7 windows)
///
/// # Returns
/// Tensor of shape `[num_windows * batch_size, window_size, window_size, channels]`
/// where `num_windows = (height / window_size) * (width / window_size)`
///
/// # Panics
/// The function assumes that both height and width are divisible by `window_size`.
fn window_partition<B: Backend>(x: Tensor<B, 4>, window_size: usize) -> Tensor<B, 4> {
    let [b, h, w, c] = x.dims();
    let x = x.reshape([
        b,
        h / window_size,
        window_size,
        w / window_size,
        window_size,
        c,
    ]);

    x.permute([0, 1, 3, 2, 4, 5]).reshape([
        b * (h / window_size) * (w / window_size),
        window_size,
        window_size,
        c,
    ])
}

/// Reverses the window partitioning operation, merging windows back into feature maps.
///
/// This function is the inverse of `window_partition`, reconstructing the original
/// feature map layout from windowed tensors.
///
/// # Arguments
/// - `windows`: Windowed tensor of shape `[num_windows * batch_size, window_size, window_size, channels]`
/// - `window_size`: Size of each window that was used in partitioning
/// - `h`: Original height of the feature map
/// - `w`: Original width of the feature map
///
/// # Returns
/// Tensor of shape `[batch_size, height, width, channels]` representing the
/// reconstructed feature map
fn window_reverse<B: Backend>(
    windows: Tensor<B, 4>,
    window_size: usize,
    h: usize,
    w: usize,
) -> Tensor<B, 4> {
    let [total_windows, _, _, channels] = windows.dims();
    let b = total_windows / (h * w / window_size / window_size);
    let x = windows.reshape([
        b,
        h / window_size,
        w / window_size,
        window_size,
        window_size,
        channels,
    ]);

    x.permute([0, 1, 3, 2, 4, 5]).reshape([b, h, w, channels])
}

/// Configuration for Window-based Multi-Head Self-Attention.
///
/// This is the core attention mechanism in Swin Transformer that performs
/// self-attention within non-overlapping windows with relative position bias.
///
/// # Key Features
/// - Window-based attention computation for efficiency
/// - Relative position bias for better spatial understanding
/// - Multi-head attention with configurable head dimensions
/// - Support for both regular and shifted window attention
///
/// # Arguments
/// - `dim`: Number of input channels/embedding dimension
/// - `window_size`: Height and width of the attention window (typically [7, 7])
/// - `num_heads`: Number of attention heads
/// - `qkv_bias`: Whether to add learnable bias to query, key, value projections
/// - `qk_scale`: Override default scale factor (head_dim^-0.5) if provided
/// - `attn_drop`: Dropout probability for attention weights
/// - `proj_drop`: Dropout probability for output projection
#[derive(Config, Debug)]
pub struct WindowAttentionConfig {
    dim: usize,
    window_size: [usize; 2],
    num_heads: usize,
    #[config(default = "true")]
    qkv_bias: bool,
    #[config(default = "None")]
    qk_scale: Option<f64>,
    #[config(default = "0.0")]
    attn_drop: f64,
    #[config(default = "0.0")]
    proj_drop: f64,
}

impl WindowAttentionConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> WindowAttention<B> {
        let head_dim = self.dim / self.num_heads;
        let relative_position_bias_table = Tensor::zeros(
            [
                (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1),
                self.num_heads,
            ],
            device,
        );

        // Recreate PyTorch's exact relative position calculation
        // coords: [2, Wh, Ww] -> coords_flatten: [2, Wh*Ww]
        let coords = create_coordinate_grid(self.window_size[0], self.window_size[1], device);
        let coords_flatten: Tensor<B, 2> = coords.flatten(1, 2); // Shape: [2, window_size^2]

        // relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        // Result shape: [2, Wh*Ww, Wh*Ww]
        let relative_coords: Tensor<B, 3> =
            coords_flatten.clone().unsqueeze_dim::<3>(2) - coords_flatten.unsqueeze_dim::<3>(1);

        // relative_coords = relative_coords.permute(1, 2, 0)
        // Result shape: [Wh*Ww, Wh*Ww, 2]
        let relative_coords = relative_coords.permute([1, 2, 0]);

        // Extract h and w coordinates (now in correct order)
        let [num_positions, _, _] = relative_coords.dims();
        let h_coords = relative_coords
            .clone()
            .slice([0..num_positions, 0..num_positions, 0..1]);
        let w_coords = relative_coords.slice([0..num_positions, 0..num_positions, 1..2]);

        // Remove the last dimension using reshape
        let h_coords = h_coords.reshape([num_positions, num_positions]);
        let w_coords = w_coords.reshape([num_positions, num_positions]);

        // Apply shifts exactly as in PyTorch
        // relative_coords[:, :, 0] += self.window_size[0] - 1
        // relative_coords[:, :, 1] += self.window_size[1] - 1
        let h_coords_shifted = h_coords + (self.window_size[0] - 1) as f64;
        let w_coords_shifted = w_coords + (self.window_size[1] - 1) as f64;

        // relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        let h_index = h_coords_shifted * (2 * self.window_size[1] - 1) as f64;
        // relative_position_index = relative_coords.sum(-1)
        let relative_position_index = (h_index + w_coords_shifted).int();

        // Final tensor shape: [window_area, window_area]
        let relative_position_index = Param::from_tensor(relative_position_index.float());

        let qkv = LinearConfig::new(self.dim, self.dim * 3)
            .with_bias(self.qkv_bias)
            .init(device);
        let attn_drop = DropoutConfig::new(self.attn_drop).init();
        let proj = LinearConfig::new(self.dim, self.dim).init(device);
        let proj_drop = DropoutConfig::new(self.proj_drop).init();

        let relative_position_bias_table = Param::from_tensor(trunc_normal(
            relative_position_bias_table,
            0.02,
            1.0,
            -2.0,
            2.0,
        ));

        WindowAttention {
            dim: self.dim,
            window_size: self.window_size,
            num_heads: self.num_heads,
            scale: self
                .qk_scale
                .unwrap_or_else(|| (head_dim as f64).powf(-0.5)),
            relative_position_bias_table,
            relative_position_index,
            qkv,
            attn_drop,
            proj,
            proj_drop,
        }
    }
}

/// Window-based Multi-Head Self-Attention module.
///
/// This module implements the core attention mechanism of Swin Transformer, performing
/// multi-head self-attention within local windows with relative position bias.
///
/// # Architecture
/// The attention computation follows:
/// 1. Linear projection to Query, Key, Value
/// 2. Multi-head attention with relative position bias
/// 3. Optional attention mask for shifted windows
/// 4. Output projection with dropout
///
/// # Components
/// - `dim`: Input embedding dimension
/// - `window_size`: Size of attention window [height, width]
/// - `num_heads`: Number of attention heads
/// - `scale`: Scaling factor for attention scores (typically head_dim^-0.5)
/// - `relative_position_bias_table`: Learnable relative position bias parameters
/// - `relative_position_index`: Pre-computed relative position indices
/// - `qkv`: Combined Query-Key-Value projection layer
/// - `attn_drop`: Dropout for attention weights
/// - `proj`: Output projection layer
/// - `proj_drop`: Dropout for output projection
#[derive(Module, Debug)]
pub struct WindowAttention<B: Backend> {
    dim: usize,
    window_size: [usize; 2],
    num_heads: usize,
    scale: f64,
    relative_position_bias_table: Param<Tensor<B, 2>>,
    relative_position_index: Param<Tensor<B, 2>>,
    qkv: Linear<B>,
    attn_drop: Dropout,
    proj: Linear<B>,
    proj_drop: Dropout,
}

impl<B: Backend> WindowAttention<B> {
    /// Forward pass of window-based multi-head self-attention.
    ///
    /// Computes self-attention within windows with relative position bias.
    /// Supports optional attention masking for shifted window attention.
    ///
    /// # Arguments
    /// - `x`: Input tensor of shape `[num_windows * batch_size, window_size * window_size, channels]`
    /// - `mask`: Optional attention mask of shape `[num_windows, window_size * window_size, window_size * window_size]`
    ///   Used for shifted window attention to mask out invalid cross-window connections
    ///
    /// # Returns
    /// Output tensor of shape `[num_windows * batch_size, window_size * window_size, channels]`
    pub fn forward(&self, x: Tensor<B, 3>, mask: Option<Tensor<B, 3>>) -> Tensor<B, 3> {
        let [b, n, c] = x.dims();
        let qkv = self
            .qkv
            .forward(x)
            .reshape([b, n, 3, self.num_heads, c / self.num_heads])
            .permute([2, 0, 3, 1, 4]);
        let [_, d2, d3, d4m, d5] = qkv.dims();
        // Use reshape instead of squeeze to avoid backend-specific issues
        let q: Tensor<B, 4> = qkv
            .clone()
            .slice(s![0..1, .., .., .., ..])
            .reshape([d2, d3, d4m, d5]);
        let k: Tensor<B, 4> = qkv
            .clone()
            .slice(s![1..2, .., .., .., ..])
            .reshape([d2, d3, d4m, d5]);
        let v: Tensor<B, 4> = qkv
            .slice(s![2..3, .., .., .., ..])
            .reshape([d2, d3, d4m, d5]);

        let q = q * self.scale;

        let attn = q.matmul(k.swap_dims(2, 3));
        // Use stable tensor operations for relative position bias
        let indices = self.relative_position_index.val().flatten(0, 1).int();
        let relative_position_bias = self
            .relative_position_bias_table
            .val()
            .select(0, indices)
            .reshape([
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                self.num_heads,
            ])
            .permute([2, 0, 1]);

        // Add relative position bias with proper broadcasting
        let [bias_heads, bias_h, bias_w] = relative_position_bias.dims();
        let bias_expanded = relative_position_bias.reshape([1, bias_heads, bias_h, bias_w]);
        let attn = attn + bias_expanded;

        let attn = {
            match mask {
                Some(mask) => {
                    let [nw, mask_h, mask_w] = mask.dims();
                    // Reshape attention for mask application
                    let attn_reshaped = attn.reshape([b / nw, nw, self.num_heads, n, n]);

                    // Expand mask dimensions properly using reshape instead of unsqueeze
                    let mask_expanded = mask.reshape([1, nw, 1, mask_h, mask_w]);

                    let attn_masked = attn_reshaped + mask_expanded;
                    attn_masked.reshape([b, self.num_heads, n, n])
                }
                None => attn,
            }
        };

        let attn = softmax(attn, 3);

        let attn = self.attn_drop.forward(attn);
        let x = attn.matmul(v).swap_dims(1, 2).reshape([b, n, c]);

        let x = self.proj.forward(x);

        self.proj_drop.forward(x)
    }
}

/// Configuration for a Swin Transformer Block.
///
/// This is the fundamental building block of the Swin Transformer, combining
/// window-based multi-head self-attention with a feed-forward network.
///
/// # Architecture
/// Each block consists of:
/// 1. Layer normalization
/// 2. Window-based multi-head self-attention (W-MSA or SW-MSA)
/// 3. Residual connection
/// 4. Layer normalization  
/// 5. Multi-layer perceptron (MLP)
/// 6. Residual connection
///
/// # Window Shifting
/// - When `shift_size = 0`: Regular window-based multi-head self-attention (W-MSA)
/// - When `shift_size > 0`: Shifted window-based multi-head self-attention (SW-MSA)
///
/// # Arguments
/// - `dim`: Number of input channels
/// - `num_heads`: Number of attention heads
/// - `window_size`: Window size (default: 7)
/// - `shift_size`: Shift size for SW-MSA (default: 0, typically window_size/2 for shifted blocks)
/// - `mlp_ratio`: Ratio of MLP hidden dimension to embedding dimension (default: 4.0)
/// - `qkv_bias`: Whether to add bias to QKV projections (default: true)
/// - `qk_scale`: Override default QK scale if provided
/// - `drop`: Dropout rate (default: 0.0)
/// - `attn_drop`: Attention dropout rate (default: 0.0)
/// - `drop_path`: Stochastic depth rate (default: 0.0)
#[derive(Config, Debug)]
pub struct SwinTransformerBlockConfig {
    dim: usize,
    num_heads: usize,
    #[config(default = "7")]
    window_size: usize,
    #[config(default = "0")]
    shift_size: usize,
    #[config(default = "4.0")]
    mlp_ratio: f64,
    #[config(default = "true")]
    qkv_bias: bool,
    #[config(default = "None")]
    qk_scale: Option<f64>,
    #[config(default = "0.0")]
    drop: f64,
    #[config(default = "0.0")]
    attn_drop: f64,
    #[config(default = "0.0")]
    drop_path: f64,
    // act_layer
    // norm_layer
}

impl SwinTransformerBlockConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> SwinTransformerBlock<B> {
        let norm1 = LayerNormConfig::new(self.dim).init(device);
        let attn = WindowAttentionConfig::new(
            self.dim,
            [self.window_size, self.window_size],
            self.num_heads,
        )
        .with_qkv_bias(self.qkv_bias)
        .with_qk_scale(self.qk_scale)
        .with_attn_drop(self.attn_drop)
        .with_proj_drop(self.drop)
        .init(device);

        let drop_path = DropPathConfig::new().with_drop_prob(self.drop_path).init();
        let norm2 = LayerNormConfig::new(self.dim).init(device);
        let mlp_hidden_dim = ((self.dim as f64) * self.mlp_ratio) as usize;
        let mlp = MlpConfig::new(self.dim)
            .with_hidden_features(Some(mlp_hidden_dim))
            .with_drop(self.drop)
            .init(device);

        SwinTransformerBlock {
            window_size: self.window_size,
            shift_size: self.shift_size,
            norm1,
            attn,
            drop_path,
            norm2,
            mlp,
        }
    }
}

/// Swin Transformer Block implementation.
///
/// The core building block that combines window-based self-attention with
/// feed-forward network, following the transformer architecture with
/// pre-normalization and residual connections.
///
/// # Processing Flow
/// 1. Apply layer normalization to input
/// 2. Perform window partitioning and optional cyclic shifting
/// 3. Compute window-based multi-head self-attention  
/// 4. Reverse shifts and merge windows
/// 5. Apply residual connection with drop path
/// 6. Apply layer normalization and MLP
/// 7. Apply final residual connection with drop path
///
/// # Components
/// - `window_size`: Size of attention windows
/// - `shift_size`: Amount of cyclic shift (0 for W-MSA, window_size/2 for SW-MSA)
/// - `norm1`: First layer normalization (before attention)
/// - `attn`: Window-based multi-head self-attention module
/// - `norm2`: Second layer normalization (before MLP)
/// - `mlp`: Feed-forward network
/// - `drop_path`: Stochastic depth for regularization
#[derive(Module, Debug)]
pub struct SwinTransformerBlock<B: Backend> {
    window_size: usize,
    shift_size: usize,
    norm1: LayerNorm<B>,
    attn: WindowAttention<B>,
    norm2: LayerNorm<B>,
    mlp: Mlp<B>,
    drop_path: DropPath,
}

impl<B: Backend> SwinTransformerBlock<B> {
    /// Forward pass through a Swin Transformer block.
    ///
    /// Performs the complete Swin Transformer block computation including:
    /// window partitioning, attention computation, window merging, and MLP processing.
    ///
    /// # Arguments
    /// - `x`: Input tensor of shape `[batch_size, height * width, channels]`
    /// - `h`: Height of the feature map
    /// - `w`: Width of the feature map  
    /// - `mask_matrix`: Attention mask for shifted window attention of shape
    ///   `[num_windows, window_size * window_size, window_size * window_size]`
    ///
    /// # Returns
    /// Output tensor of shape `[batch_size, height * width, channels]`
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        h: usize,
        w: usize,
        mask_matrix: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let [b, _l, c] = x.dims();

        let shortcut = x.clone();
        let x = self.norm1.forward(x);
        let x = x.reshape([b, h, w, c]);

        // Pad feature maps to multiples of window size
        let pad_l = 0;
        let pad_t = 0;
        let pad_r = (self.window_size - w % self.window_size) % self.window_size;
        let pad_b = (self.window_size - h % self.window_size) % self.window_size;
        let x = x
            .permute([0, 3, 1, 2])
            .pad((pad_l, pad_r, pad_t, pad_b), B::FloatElem::from_elem(0.0))
            .permute([0, 2, 3, 1]);
        let [_, hp, wp, _] = x.dims();

        let (shifted_x, attn_mask) = {
            if self.shift_size > 0 {
                let shifted_x = x.roll(&[-(self.shift_size as i64), -(self.shift_size as i64)], &[
                    1, 2,
                ]);
                let attn_mask = mask_matrix;
                (shifted_x, Some(attn_mask))
            } else {
                (x, None)
            }
        };

        let x_window = window_partition(shifted_x, self.window_size);
        let num_windows = x_window.dims()[0];
        let x_window = x_window.reshape([num_windows, self.window_size * self.window_size, c]);

        let attn_window = self.attn.forward(x_window, attn_mask);
        let attn_num_windows = attn_window.dims()[0];

        let attn_window =
            attn_window.reshape([attn_num_windows, self.window_size, self.window_size, c]);
        let shifted_x = window_reverse(attn_window, self.window_size, hp, wp);

        let x = {
            if self.shift_size > 0 {
                shifted_x.roll(&[self.shift_size as i64, self.shift_size as i64], &[1, 2])
            } else {
                shifted_x
            }
        };

        let x = {
            if pad_r > 0 || pad_b > 0 {
                x.slice(s![.., 0..h, 0..w, ..])
            } else {
                x
            }
        };

        let x = x.reshape([b, h * w, c]);

        let x = shortcut + self.drop_path.forward(x);

        x.clone()
            + self
                .drop_path
                .forward(self.mlp.forward(self.norm2.forward(x)))
    }
}

/// Configuration for Patch Merging layer.
///
/// Patch merging reduces the spatial resolution while increasing the channel dimension,
/// acting as a downsampling operation between Swin Transformer stages.
///
/// # Operation
/// The patch merging operation:
/// 1. Takes 2x2 neighboring patches and concatenates their features
/// 2. Applies layer normalization to the concatenated features  
/// 3. Uses a linear layer to reduce the dimension from 4C to 2C
///
/// This effectively halves the spatial resolution (H/2, W/2) while doubling
/// the channel dimension.
///
/// # Arguments
/// - `dim`: Input channel dimension (output will be 2*dim channels)
#[derive(Config, Debug)]
pub struct PatchMergingConfig {
    dim: usize,
    // norm: nn::LayerNormConfig,
}

impl PatchMergingConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> PatchMerging<B> {
        PatchMerging {
            norm: LayerNormConfig::new(4 * self.dim).init(device),
            reduction: LinearConfig::new(4 * self.dim, 2 * self.dim)
                .with_bias(false)
                .init(device),
        }
    }
}

/// Patch Merging layer for downsampling in Swin Transformer.
///
/// This layer reduces spatial resolution by merging 2x2 neighboring patches
/// while doubling the channel dimension. It serves as the downsampling operation
/// between different stages of the Swin Transformer hierarchy.
///
/// # Processing Steps
/// 1. Reshape input from sequence format to spatial format
/// 2. Extract 2x2 neighboring patches: top-left, top-right, bottom-left, bottom-right
/// 3. Concatenate the 4 patches along channel dimension (C -> 4C)
/// 4. Apply layer normalization
/// 5. Apply linear reduction (4C -> 2C)
///
/// # Components
/// - `norm`: Layer normalization applied to concatenated features (4C dimensions)
/// - `reduction`: Linear layer that reduces channels from 4C to 2C without bias
#[derive(Module, Debug)]
pub struct PatchMerging<B: Backend> {
    norm: LayerNorm<B>,
    reduction: Linear<B>,
}

impl<B: Backend> PatchMerging<B> {
    /// Forward pass through patch merging layer.
    ///
    /// Merges 2x2 neighboring patches to reduce spatial resolution while
    /// increasing channel dimension.
    ///
    /// # Arguments
    /// - `x`: Input tensor of shape `[batch_size, height * width, channels]`
    /// - `h`: Height of the feature map
    /// - `w`: Width of the feature map
    ///
    /// # Returns
    /// Output tensor of shape `[batch_size, (height/2) * (width/2), 2*channels]`
    pub fn forward(&self, x: Tensor<B, 3>, h: usize, w: usize) -> Tensor<B, 3> {
        let device = x.device();

        let [b, _l, c] = x.dims();

        let x = x.reshape([b, h, w, c]);

        let pad_input = h % 2 == 1 || w % 2 == 1;

        let x = {
            if pad_input {
                x.permute([0, 3, 1, 2])
                    .pad((0, w % 2, 0, h % 2), B::FloatElem::from_elem(0.0))
                    .permute([0, 2, 3, 1])
            } else {
                x
            }
        };

        let top_idx = Tensor::arange_step(0..h as i64, 2, &device);
        let bottom_idx = Tensor::arange_step(1..h as i64, 2, &device);
        let left_idx = Tensor::arange_step(0..w as i64, 2, &device);
        let right_idx = Tensor::arange_step(1..w as i64, 2, &device);

        let x0 = x
            .clone()
            .select(1, top_idx.clone())
            .select(2, left_idx.clone());
        let x1 = x.clone().select(1, bottom_idx.clone()).select(2, left_idx);
        let x2 = x.clone().select(1, top_idx).select(2, right_idx.clone());
        let x3 = x.select(1, bottom_idx).select(2, right_idx);

        let x = Tensor::cat(vec![x0, x1, x2, x3], 3);
        let x = x.reshape([b, h.div_ceil(2) * w.div_ceil(2), 4 * c]);

        let x = self.norm.forward(x);

        self.reduction.forward(x)
    }
}

/// Configuration for a Basic Layer (stage) in Swin Transformer.
///
/// A basic layer represents one stage of the Swin Transformer hierarchy,
/// consisting of multiple Swin Transformer blocks followed by an optional
/// patch merging layer for downsampling.
///
/// # Stage Structure
/// Each stage alternates between:
/// - Even-indexed blocks: Regular window multi-head self-attention (W-MSA)  
/// - Odd-indexed blocks: Shifted window multi-head self-attention (SW-MSA)
///
/// This pattern ensures information exchange between different windows while
/// maintaining computational efficiency.
///
/// # Arguments
/// - `dim`: Number of input channels for this stage
/// - `depth`: Number of Swin Transformer blocks in this stage
/// - `num_heads`: Number of attention heads
/// - `window_size`: Window size for attention computation (default: 7)
/// - `mlp_ratio`: Ratio of MLP hidden dimension to embedding dimension (default: 4.0)
/// - `qkv_bias`: Whether to add bias to QKV projections (default: true)
/// - `qk_scale`: Override default QK scale if provided
/// - `drop`: Dropout rate (default: 0.0)
/// - `attn_drop`: Attention dropout rate (default: 0.0)
/// - `drop_path`: Stochastic depth rates for each block
/// - `downsample`: Whether to add patch merging at the end (default: false)
#[derive(Config, Debug)]
pub struct BasicLayerConfig {
    dim: usize,
    depth: usize,
    num_heads: usize,
    #[config(default = "7")]
    window_size: usize,
    #[config(default = "4.0")]
    mlp_ratio: f64,
    #[config(default = "true")]
    qkv_bias: bool,
    #[config(default = "None")]
    qk_scale: Option<f64>,
    #[config(default = "0.0")]
    drop: f64,
    #[config(default = "0.0")]
    attn_drop: f64,
    #[config(default = "Vec::new()")]
    drop_path: Vec<f64>,
    #[config(default = "false")]
    downsample: bool,
    // use_checkpoint: bool,
}

impl BasicLayerConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> BasicLayer<B> {
        let blocks = (0..self.depth)
            .map(|i| {
                SwinTransformerBlockConfig::new(self.dim, self.num_heads)
                    .with_window_size(self.window_size)
                    .with_shift_size(if i % 2 == 0 { 0 } else { self.window_size / 2 })
                    .with_mlp_ratio(self.mlp_ratio)
                    .with_qkv_bias(self.qkv_bias)
                    .with_qk_scale(self.qk_scale)
                    .with_drop(self.drop)
                    .with_attn_drop(self.attn_drop)
                    .with_drop_path(self.drop_path[i])
                    .init(device)
            })
            .collect::<Vec<_>>();
        let downsample = self
            .downsample
            .then(|| PatchMergingConfig::new(self.dim).init(device));

        BasicLayer {
            window_size: self.window_size,
            shift_size: self.window_size / 2,
            blocks,
            downsample,
        }
    }
}

/// Basic Layer representing one stage of the Swin Transformer hierarchy.
///
/// A basic layer groups multiple Swin Transformer blocks that operate at the same
/// spatial resolution, with alternating regular and shifted window attention.
/// It optionally includes a patch merging layer for downsampling to the next stage.
///
/// # Attention Pattern
/// The layer alternates between two types of attention:
/// - Block 0, 2, 4, ...: Regular window attention (W-MSA) with shift_size = 0
/// - Block 1, 3, 5, ...: Shifted window attention (SW-MSA) with shift_size = window_size/2
///
/// This alternating pattern allows information to flow between different windows
/// while keeping the computational complexity linear with respect to input size.
///
/// # Components
/// - `window_size`: Size of attention windows
/// - `shift_size`: Shift amount for SW-MSA (typically window_size/2)
/// - `blocks`: Sequence of Swin Transformer blocks
/// - `downsample`: Optional patch merging layer for resolution reduction
#[derive(Module, Debug)]
pub struct BasicLayer<B: Backend> {
    window_size: usize,
    shift_size: usize,
    blocks: Vec<SwinTransformerBlock<B>>,
    downsample: Option<PatchMerging<B>>,
}

impl<B: Backend> BasicLayer<B> {
    /// Forward pass through a basic layer.
    ///
    /// Processes input through all Swin Transformer blocks in this stage,
    /// computing attention masks for shifted window attention, and optionally
    /// applies downsampling via patch merging.
    ///
    /// # Arguments
    /// - `x`: Input tensor of shape `[batch_size, height * width, channels]`
    /// - `h`: Height of the feature map
    /// - `w`: Width of the feature map
    ///
    /// # Returns
    /// A tuple containing:
    /// - `x_out`: Output before downsampling `[batch_size, h * w, channels]`
    /// - `h_out`: Output height before downsampling
    /// - `w_out`: Output width before downsampling  
    /// - `x_down`: Output after optional downsampling `[batch_size, h_down * w_down, channels_down]`
    /// - `h_down`: Height after downsampling (h/2 if downsampling, else h)
    /// - `w_down`: Width after downsampling (w/2 if downsampling, else w)
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        h: usize,
        w: usize,
    ) -> (Tensor<B, 3>, usize, usize, Tensor<B, 3>, usize, usize) {
        let device = x.device();

        let hp = ((h as f64) / self.window_size as f64).ceil() as usize * self.window_size;
        let wp = ((w as f64) / self.window_size as f64).ceil() as usize * self.window_size;
        let mut img_mask: Tensor<B, 4> = Tensor::zeros([1, hp, wp, 1], &device);

        // Only create attention mask if shift_size > 0 and dimensions are large enough
        let h_slices = [
            0..-(self.window_size as isize),
            -(self.window_size as isize)..-(self.shift_size as isize),
            -(self.shift_size as isize)..(hp as isize),
        ];
        let w_slices = [
            0..-(self.window_size as isize),
            -(self.window_size as isize)..-(self.shift_size as isize),
            -(self.shift_size as isize)..(wp as isize),
        ];

        let mut cnt = 0;
        for h_slice in h_slices {
            for w_slice in w_slices.clone() {
                img_mask = img_mask.slice_fill(
                    s![.., h_slice.clone(), w_slice, ..],
                    B::FloatElem::from_elem(cnt as f64),
                );
                cnt += 1;
            }
        }

        let mask_windows = window_partition(img_mask, self.window_size);
        let mask_num_windows = mask_windows.dims()[0];
        let mask_windows =
            mask_windows.reshape([mask_num_windows, self.window_size * self.window_size]);

        let attn_mask: Tensor<B, 3> =
            mask_windows.clone().unsqueeze_dim(1) - mask_windows.unsqueeze_dim(2);
        let attn_mask = attn_mask
            .clone()
            .mask_fill(attn_mask.clone().not_equal_elem(0.0), -100.0)
            .mask_fill(attn_mask.equal_elem(0.0), 0.0);

        let mut x = x;
        for blk in &self.blocks {
            x = blk.forward(x, h, w, attn_mask.clone());
        }
        match &self.downsample {
            Some(downsample) => {
                let x_down = downsample.forward(x.clone(), h, w);
                let wh = h.div_ceil(2);
                let ww = w.div_ceil(2);
                (x, h, w, x_down, wh, ww)
            }
            None => (x.clone(), h, w, x, h, w),
        }
    }
}

/// Configuration for Patch Embedding layer.
///
/// The patch embedding layer converts input images into patch tokens by:
/// 1. Dividing the image into non-overlapping patches
/// 2. Projecting each patch to an embedding vector using a convolutional layer
/// 3. Optionally applying layer normalization
///
/// # Patch Tokenization
/// An image of size (H, W, C) is divided into patches of size (patch_size, patch_size),
/// resulting in (H/patch_size) * (W/patch_size) patches, each projected to embed_dim dimensions.
///
/// # Arguments
/// - `patch_size`: Size of each patch (default: 4)
/// - `in_channels`: Number of input image channels (default: 3 for RGB)
/// - `embed_dim`: Embedding dimension for each patch (default: 96)
/// - `norm_layer`: Whether to apply layer normalization (default: false)
#[derive(Config, Debug)]
pub struct PatchEmbedConfig {
    #[config(default = "4")]
    patch_size: usize,
    #[config(default = "3")]
    in_channels: usize,
    #[config(default = "96")]
    embed_dim: usize,
    #[config(default = "false")]
    norm_layer: bool,
}

impl PatchEmbedConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> PatchEmbed<B> {
        let proj = Conv2dConfig::new([self.in_channels, self.embed_dim], [
            self.patch_size,
            self.patch_size,
        ])
        .with_stride([self.patch_size, self.patch_size])
        .init(device);
        let norm = self
            .norm_layer
            .then(|| LayerNormConfig::new(self.embed_dim).init(device));

        PatchEmbed {
            embed_dim: self.embed_dim,
            patch_size: self.patch_size,
            proj,
            norm,
        }
    }
}

/// Patch Embedding layer for converting images to patch tokens.
///
/// This layer is the first component of the Swin Transformer that converts
/// input images into a sequence of patch embeddings using a convolutional projection.
///
/// # Processing Steps
/// 1. Pad input image if dimensions are not divisible by patch_size
/// 2. Apply 2D convolution with kernel_size=patch_size and stride=patch_size
/// 3. Optionally apply layer normalization if configured
///
/// # Tensor Transformations
/// - Input: `[batch_size, channels, height, width]`
/// - After conv: `[batch_size, embed_dim, height/patch_size, width/patch_size]`
/// - Output: Same shape as after conv (4D tensor format for compatibility)
///
/// # Components
/// - `embed_dim`: Output embedding dimension
/// - `patch_size`: Size of each patch
/// - `proj`: 2D convolution layer for patch projection
/// - `norm`: Optional layer normalization
#[derive(Module, Debug)]
pub struct PatchEmbed<B: Backend> {
    embed_dim: usize,
    patch_size: usize,
    proj: Conv2d<B>,
    norm: Option<LayerNorm<B>>,
}

impl<B: Backend> PatchEmbed<B> {
    /// Forward pass through patch embedding layer.
    ///
    /// Converts input images into patch embeddings by applying convolutional
    /// projection and optional normalization.
    ///
    /// # Arguments
    /// - `x`: Input tensor of shape `[batch_size, in_channels, height, width]`
    ///
    /// # Returns
    /// Output tensor of shape `[batch_size, embed_dim, height/patch_size, width/patch_size]`
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [_, _, h, w] = x.dims();
        let x = {
            if w % self.patch_size != 0 {
                x.pad(
                    (0, self.patch_size - (w % self.patch_size), 0, 0),
                    B::FloatElem::from_elem(0.0),
                )
            } else {
                x
            }
        };
        let x = {
            if h % self.patch_size != 0 {
                x.pad(
                    (0, 0, 0, self.patch_size - (h % self.patch_size)),
                    B::FloatElem::from_elem(0.0),
                )
            } else {
                x
            }
        };
        let x = self.proj.forward(x);

        match &self.norm {
            Some(norm) => {
                let [batch_size, _, wh, ww] = x.dims();
                let x: Tensor<B, 3> = x.flatten(2, 3).swap_dims(1, 2);
                let x = norm.forward(x);
                x.swap_dims(1, 2)
                    .reshape([batch_size, self.embed_dim, wh, ww])
            }
            None => x,
        }
    }
}

#[derive(Config, Debug)]
pub struct SwinTransformerConfig {
    #[config(default = "224")]
    pretrain_img_size: usize,
    #[config(default = "4")]
    patch_size: usize,
    #[config(default = "3")]
    in_channels: usize,
    #[config(default = "96")]
    embed_dim: usize,
    #[config(default = "[2, 2, 6, 2]")]
    depths: [usize; 4],
    #[config(default = "[3, 6, 12, 24]")]
    num_heads: [usize; 4],
    #[config(default = "7")]
    window_size: usize,
    #[config(default = "4.0")]
    mlp_ratio: f64,
    #[config(default = "true")]
    qkv_bias: bool,
    #[config(default = "None")]
    qk_scale: Option<f64>,
    #[config(default = "0.0")]
    drop_rate: f64,
    #[config(default = "0.0")]
    attn_drop_rate: f64,
    #[config(default = "0.2")]
    drop_path_rate: f64,
    // norm_layer: f64,
    #[config(default = "false")]
    ape: bool,
    #[config(default = "true")]
    patch_norm: bool,
    #[config(default = "[0, 1, 2, 3]")]
    out_indices: [usize; 4],
    #[config(default = "-1")]
    frozen_stages: i32,
}

fn linspace(start: f64, end: f64, steps: usize) -> Vec<f64> {
    if steps == 0 {
        return Vec::new();
    }
    let mut result = Vec::with_capacity(steps);
    let step_size = (end - start) / (steps as f64 - 1.0);

    for i in 0..steps {
        result.push((i as f64).mul_add(step_size, start));
    }

    result
}

impl SwinTransformerConfig {
    pub fn init<B: Backend>(
        &self,
        device: &Device<B>,
    ) -> SwinTransformerResult<SwinTransformer<B>> {
        let num_layers = self.depths.len();

        let patch_embed = PatchEmbedConfig::new()
            .with_patch_size(self.patch_size)
            .with_in_channels(self.in_channels)
            .with_embed_dim(self.embed_dim)
            .with_norm_layer(self.patch_norm)
            .init(device);

        let absolute_pos_embed = if self.ape {
            let patches_resolution = [
                self.pretrain_img_size / self.patch_size,
                self.pretrain_img_size / self.patch_size,
            ];
            let absolute_pos_embed: Tensor<B, 4> = {
                Tensor::zeros(
                    [
                        1,
                        self.embed_dim,
                        patches_resolution[0],
                        patches_resolution[1],
                    ],
                    device,
                )
            };
            Some(Param::from_tensor(trunc_normal(
                absolute_pos_embed,
                0.0,
                0.02,
                -2.0,
                2.0,
            )))
        } else {
            None
        };

        let dpr = linspace(0.0, self.drop_path_rate, self.depths.iter().sum());

        let mut layers = Vec::new();
        for i_layer in 0..num_layers {
            let start: usize = self.depths[..i_layer].iter().sum();
            let end: usize = self.depths[..=i_layer].iter().sum();
            let layer = BasicLayerConfig::new(
                ((self.embed_dim as i32) * 2_i32.pow(i_layer as u32)) as usize,
                self.depths[i_layer],
                self.num_heads[i_layer],
            )
            .with_window_size(self.window_size)
            .with_mlp_ratio(self.mlp_ratio)
            .with_qkv_bias(self.qkv_bias)
            .with_qk_scale(self.qk_scale)
            .with_drop(self.drop_rate)
            .with_attn_drop(self.attn_drop_rate)
            .with_drop_path(dpr[start..end].to_owned())
            .with_downsample(i_layer < num_layers - 1)
            .init(device);
            layers.push(layer);
        }

        let num_features: Vec<_> = (0..num_layers)
            .map(|i| ((self.embed_dim as i32) * 2_i32.pow(i as u32)) as usize)
            .collect();

        let mut norm_layers = Vec::new();
        for i_layer in self.out_indices {
            let layer = LayerNormConfig::new(num_features[i_layer]).init(device);
            norm_layers.push(layer);
        }

        let num_features = num_features.try_into().map_err(|_| {
            SwinTransformerError::ModelInitializationFailed {
                reason: "Failed to convert num_features to array".to_string(),
            }
        })?;

        Ok(SwinTransformer {
            patch_embed,
            pos_drop: DropoutConfig::new(self.drop_rate).init(),
            num_layers,
            layers,
            num_features,
            norm_layers,
            out_indices: self.out_indices,
            absolute_pos_embed,
        })
    }
}

#[derive(Module, Debug)]
pub struct SwinTransformer<B: Backend> {
    patch_embed: PatchEmbed<B>,
    absolute_pos_embed: Option<Param<Tensor<B, 4>>>,
    pos_drop: Dropout,
    num_layers: usize,
    layers: Vec<BasicLayer<B>>,
    norm_layers: Vec<LayerNorm<B>>,
    out_indices: [usize; 4],
    num_features: [usize; 4],
}

impl<B: Backend> SwinTransformer<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> SwinTransformerResult<[Tensor<B, 4>; 4]> {
        let x = self.patch_embed.forward(x);

        let [_, _, wh, ww] = x.dims();

        let x = {
            match &self.absolute_pos_embed {
                Some(absolute_pos_embed) => {
                    let absolute_pos_embed = interpolate(
                        absolute_pos_embed.val(),
                        [wh, ww],
                        InterpolateOptions::new(InterpolateMode::Bicubic),
                    );

                    x + absolute_pos_embed
                }
                None => x,
            }
        };

        // Pre-allocate output vector with known capacity
        let output_layers = self.out_indices.len();
        let mut outs = Vec::with_capacity(output_layers);

        let x: Tensor<B, 3> = x.flatten(2, 3).swap_dims(1, 2);
        let x = self.pos_drop.forward(x);
        let mut x = x;
        let mut wh = wh;
        let mut ww = ww;
        for i in 0..self.num_layers {
            let (x_out, h, w, x_, wh_, ww_) = self.layers[i].forward(x, wh, ww);
            x = x_;
            wh = wh_;
            ww = ww_;
            if self.out_indices.contains(&i) {
                let x_out = self.norm_layers[i].forward(x_out);
                let batch_size = x_out.dims()[0];
                let out = x_out
                    .reshape([batch_size, h, w, self.num_features[i]])
                    .permute([0, 3, 1, 2]);
                outs.push(out);
            }
        }
        outs.try_into().map_err(|_| {
            SwinTransformerError::TensorOperationFailed {
                operation: "Failed to convert outputs to array".to_string(),
            }
        })
    }
}

/// Creates a Swin Transformer Tiny (Swin-T) model.
///
/// # Architecture Parameters
/// * Embedding dimension: 96
/// * Layer depths: [2, 2, 6, 2]
/// * Number of attention heads: [3, 6, 12, 24]
/// * Window size: 7x7
///
/// # Arguments
/// * `device` - The device on which to initialize the model
///
/// # Returns
/// A result containing the initialized Swin-T model or an error
///
/// # Errors  
/// Returns `SwinTransformerError::ModelInitializationFailed` when:
/// * Device memory allocation fails
/// * Invalid tensor operations during weight initialization
/// * Backend-specific initialization errors occur
pub fn swin_v1_t<B: Backend>(device: &Device<B>) -> SwinTransformerResult<SwinTransformer<B>> {
    SwinTransformerConfig::new()
        .with_embed_dim(96)
        .with_depths([2, 2, 6, 2])
        .with_num_heads([3, 6, 12, 24])
        .with_window_size(7)
        .init(device)
}

/// Creates a Swin Transformer Small (Swin-S) model.
///
/// # Architecture Parameters
/// * Embedding dimension: 96
/// * Layer depths: [2, 2, 18, 2] (deeper than Swin-T)
/// * Number of attention heads: [3, 6, 12, 24]
/// * Window size: 7x7
///
/// # Arguments
/// * `device` - The device on which to initialize the model
///
/// # Returns
/// A result containing the initialized Swin-S model or an error
///
/// # Errors  
/// Returns `SwinTransformerError::ModelInitializationFailed` when initialization fails
pub fn swin_v1_s<B: Backend>(device: &Device<B>) -> SwinTransformerResult<SwinTransformer<B>> {
    SwinTransformerConfig::new()
        .with_embed_dim(96)
        .with_depths([2, 2, 18, 2])
        .with_num_heads([3, 6, 12, 24])
        .with_window_size(7)
        .init(device)
}

pub fn swin_v1_b<B: Backend>(device: &Device<B>) -> SwinTransformerResult<SwinTransformer<B>> {
    SwinTransformerConfig::new()
        .with_embed_dim(128)
        .with_depths([2, 2, 18, 2])
        .with_num_heads([4, 8, 16, 32])
        .with_window_size(12)
        .init(device)
}

pub fn swin_v1_l<B: Backend>(device: &Device<B>) -> SwinTransformerResult<SwinTransformer<B>> {
    SwinTransformerConfig::new()
        .with_embed_dim(192)
        .with_depths([2, 2, 18, 2])
        .with_num_heads([6, 12, 24, 48])
        .with_window_size(12)
        .init(device)
}

#[cfg(test)]
mod tests {
    use burn::tensor::{Distribution, Transaction};
    use rstest::rstest;

    use super::*;
    use crate::tests::TestBackend;

    #[rstest]
    #[case(64, Some(256), 2, 49)] // Small dimensions
    #[case(96, Some(384), 2, 49)] // Original test case
    #[case(128, None, 1, 64)] // No hidden features specified (should default to input)
    #[case(192, Some(768), 4, 196)] // Larger dimensions
    #[case(256, Some(1024), 3, 784)] // Even larger
    fn mlp_preserves_input_output_dimensions(
        #[case] input_dim: usize,
        #[case] hidden_features: Option<usize>,
        #[case] batch_size: usize,
        #[case] seq_len: usize,
    ) {
        let device = Default::default();
        let config = MlpConfig::new(input_dim).with_hidden_features(hidden_features);
        let mlp = config.init::<TestBackend>(&device);

        let input = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_len, input_dim],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let output = mlp.forward(input);

        assert_eq!(output.shape().dims, [batch_size, seq_len, input_dim]);
    }

    #[rstest]
    #[case(7, 14, 14, 2, 96)] // Original case: 7x7 windows on 14x14
    #[case(4, 16, 16, 1, 64)] // 4x4 windows on 16x16
    #[case(8, 24, 24, 3, 128)] // 8x8 windows on 24x24
    #[case(7, 28, 28, 2, 192)] // 7x7 windows on 28x28 (4x4 grid)
    #[case(5, 20, 15, 1, 32)] // Different h and w sizes
    fn window_partition_and_reverse_are_inverse_operations(
        #[case] window_size: usize,
        #[case] h: usize,
        #[case] w: usize,
        #[case] batch_size: usize,
        #[case] channels: usize,
    ) {
        let device = Default::default();

        let input = Tensor::<TestBackend, 4>::random(
            [batch_size, h, w, channels],
            Distribution::Normal(0.0, 1.0),
            &device,
        );

        // Test window partition
        let windows = window_partition(input.clone(), window_size);
        let expected_num_windows = (h / window_size) * (w / window_size);
        assert_eq!(windows.shape().dims, [
            batch_size * expected_num_windows,
            window_size,
            window_size,
            channels
        ]);

        // Test window reverse
        let reversed = window_reverse(windows, window_size, h, w);
        assert_eq!(reversed.shape().dims, [batch_size, h, w, channels]);

        // Test that the operations are truly inverse (values are preserved)
        let [input_data, reversed_data] = Transaction::default()
            .register(input)
            .register(reversed)
            .execute()
            .try_into()
            .expect("Correct amount of tensor data");

        // Compare the actual values to ensure they're the same
        for i in 0..input_data.as_slice::<f32>().unwrap().len() {
            let diff = (input_data.as_slice::<f32>().unwrap()[i]
                - reversed_data.as_slice::<f32>().unwrap()[i])
                .abs();
            assert!(
                diff < 1e-6,
                "Values don't match after partition/reverse: {} vs {}",
                input_data.as_slice::<f32>().unwrap()[i],
                reversed_data.as_slice::<f32>().unwrap()[i]
            );
        }
    }

    #[rstest]
    #[case(96, [7, 7], 3, 4)] // Original case: 96 dim, 7x7 window, 3 heads
    #[case(128, [8, 8], 4, 6)] // Larger: 128 dim, 8x8 window, 4 heads
    #[case(48, [4, 4], 3, 1)] // Small batch: 48 dim, 4x4 window, 3 heads
    fn window_attention_preserves_input_dimensions(
        #[case] dim: usize,
        #[case] window_size: [usize; 2],
        #[case] num_heads: usize,
        #[case] num_windows: usize,
    ) {
        let device = Default::default();
        let config = WindowAttentionConfig::new(dim, window_size, num_heads);
        let attention = config.init::<TestBackend>(&device);

        let window_area = window_size[0] * window_size[1];
        let input = Tensor::<TestBackend, 3>::random(
            [num_windows, window_area, dim],
            Distribution::Normal(0.0, 1.0),
            &device,
        );

        let output = attention.forward(input, None);
        assert_eq!(output.shape().dims, [num_windows, window_area, dim]);

        // Test that dimension is divisible by num_heads (required for multi-head attention)
        assert_eq!(
            dim % num_heads,
            0,
            "Dimension {} must be divisible by number of heads {}",
            dim,
            num_heads
        );
    }

    #[rstest]
    #[case(4, 3, 96, 224, 224, 2)] // Original case: patch_size=4, 3 channels, 96 dim, 224x224
    #[case(8, 3, 128, 256, 256, 3)] // Larger patches: patch_size=8, 3 channels, 128 dim, 256x256
    #[case(4, 3, 64, 128, 160, 2)] // Non-square input: patch_size=4, 3 channels, 64 dim, 128x160
    fn patch_embed_converts_image_to_patch_tokens(
        #[case] patch_size: usize,
        #[case] in_channels: usize,
        #[case] embed_dim: usize,
        #[case] height: usize,
        #[case] width: usize,
        #[case] batch_size: usize,
    ) {
        let device = Default::default();
        let config = PatchEmbedConfig::new()
            .with_patch_size(patch_size)
            .with_in_channels(in_channels)
            .with_embed_dim(embed_dim);
        let patch_embed = config.init::<TestBackend>(&device);

        let input = Tensor::<TestBackend, 4>::random(
            [batch_size, in_channels, height, width],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let output = patch_embed.forward(input);

        // Calculate expected output size considering potential padding
        let output_h = (height + patch_size - 1) / patch_size; // Ceiling division for padding
        let output_w = (width + patch_size - 1) / patch_size;

        assert_eq!(output.shape().dims, [
            batch_size, embed_dim, output_h, output_w
        ]);

        // Verify that the output dimensions make sense
        assert!(
            output_h > 0 && output_w > 0,
            "Output dimensions must be positive: {}x{}",
            output_h,
            output_w
        );
    }

    #[rstest]
    #[case(96, 56, 56, 2)] // Original case: 96 dim, 56x56 spatial
    #[case(128, 28, 28, 3)] // Larger dim: 128 dim, 28x28 spatial
    #[case(48, 64, 48, 1)] // Non-square: 48 dim, 64x48 spatial
    fn patch_merging_reduces_spatial_dims_doubles_channels(
        #[case] input_dim: usize,
        #[case] h: usize,
        #[case] w: usize,
        #[case] batch_size: usize,
    ) {
        let device = Default::default();
        let config = PatchMergingConfig::new(input_dim);
        let patch_merging = config.init::<TestBackend>(&device);

        let input = Tensor::<TestBackend, 3>::random(
            [batch_size, h * w, input_dim],
            Distribution::Normal(0.0, 1.0),
            &device,
        );

        let output = patch_merging.forward(input, h, w);

        // Calculate expected output dimensions
        let expected_h = (h + 1) / 2; // Ceiling division
        let expected_w = (w + 1) / 2; // Ceiling division
        let expected_channels = 2 * input_dim;

        assert_eq!(output.shape().dims, [
            batch_size,
            expected_h * expected_w,
            expected_channels
        ]);

        // Verify spatial reduction and channel doubling
        assert!(
            expected_h * expected_w <= h * w,
            "Spatial dimensions should be reduced: {}x{} -> {}x{}",
            h,
            w,
            expected_h,
            expected_w
        );
        assert_eq!(
            expected_channels,
            2 * input_dim,
            "Channels should be doubled: {} -> {}",
            input_dim,
            expected_channels
        );
    }

    #[rstest]
    #[case(96, 3, 56, 56, 2)] // Original case: 96 dim, 3 heads, 56x56 spatial
    #[case(128, 4, 28, 28, 3)] // Larger: 128 dim, 4 heads, 28x28 spatial
    #[case(48, 3, 64, 48, 2)] // Non-square: 48 dim, 3 heads, 64x48 spatial
    fn swin_transformer_block_preserves_sequence_length(
        #[case] dim: usize,
        #[case] num_heads: usize,
        #[case] h: usize,
        #[case] w: usize,
        #[case] batch_size: usize,
    ) {
        let device = Default::default();
        let config = SwinTransformerBlockConfig::new(dim, num_heads);
        let block = config.init::<TestBackend>(&device);

        let input = Tensor::<TestBackend, 3>::random(
            [batch_size, h * w, dim],
            Distribution::Normal(0.0, 1.0),
            &device,
        );

        // Create appropriate mask matrix for the test
        let window_size = 7; // Default window size
        let num_windows =
            ((h + window_size - 1) / window_size) * ((w + window_size - 1) / window_size);
        let mask_matrix = Tensor::<TestBackend, 3>::zeros(
            [
                num_windows,
                window_size * window_size,
                window_size * window_size,
            ],
            &device,
        );

        let output = block.forward(input, h, w, mask_matrix);
        assert_eq!(output.shape().dims, [batch_size, h * w, dim]);

        // Verify that dimension is divisible by num_heads
        assert_eq!(
            dim % num_heads,
            0,
            "Dimension {} must be divisible by number of heads {}",
            dim,
            num_heads
        );
    }

    #[rstest]
    #[case(96, 2, 3, true, vec![0.0, 0.1], 56, 56, 2)] // Original case: 96 dim, 2 blocks, 3 heads, downsample
    #[case(64, 1, 2, false, vec![0.0], 32, 32, 1)] // No downsample: 64 dim, 1 block, 2 heads
    #[case(128, 3, 4, true, vec![0.0, 0.05, 0.1], 28, 28, 3)] // More blocks: 128 dim, 3 blocks, 4 heads
    fn basic_layer_handles_various_configurations(
        #[case] dim: usize,
        #[case] depth: usize,
        #[case] num_heads: usize,
        #[case] downsample: bool,
        #[case] drop_path: Vec<f64>,
        #[case] h: usize,
        #[case] w: usize,
        #[case] batch_size: usize,
    ) {
        let device = Default::default();
        let config = BasicLayerConfig::new(dim, depth, num_heads)
            .with_downsample(downsample)
            .with_drop_path(drop_path);
        let layer = config.init::<TestBackend>(&device);

        let input = Tensor::<TestBackend, 3>::random(
            [batch_size, h * w, dim],
            Distribution::Normal(0.0, 1.0),
            &device,
        );

        let (x_out, h_out, w_out, x_down, h_down, w_down) = layer.forward(input, h, w);

        // Before downsampling - should preserve dimensions
        assert_eq!(x_out.shape().dims, [batch_size, h * w, dim]);
        assert_eq!(h_out, h);
        assert_eq!(w_out, w);

        if downsample {
            // After downsampling - spatial dims halved, channels doubled
            let expected_h = (h + 1) / 2; // Ceiling division
            let expected_w = (w + 1) / 2; // Ceiling division
            assert_eq!(x_down.shape().dims, [
                batch_size,
                expected_h * expected_w,
                2 * dim
            ]);
            assert_eq!(h_down, expected_h);
            assert_eq!(w_down, expected_w);
        } else {
            // No downsampling - should be identical to output
            assert_eq!(x_down.shape().dims, x_out.shape().dims);
            assert_eq!(h_down, h_out);
            assert_eq!(w_down, w_out);
        }

        // Verify dimension is divisible by num_heads
        assert_eq!(
            dim % num_heads,
            0,
            "Dimension {} must be divisible by number of heads {}",
            dim,
            num_heads
        );
    }

    #[rstest]
    #[case("swin_v1_t", [96, 192, 384, 768])]
    #[case("swin_v1_s", [96, 192, 384, 768])]
    #[case("swin_v1_b", [128, 256, 512, 1024])]
    #[case("swin_v1_l", [192, 384, 768, 1536])]
    fn swin_models_forward_returns_correct_shapes(
        #[case] model_variant: &str,
        #[case] expected_channels: [usize; 4],
    ) {
        let device = Default::default();

        let model = match model_variant {
            "swin_v1_t" => {
                swin_v1_t::<TestBackend>(&device).expect("Failed to create Swin-T model")
            }
            "swin_v1_s" => {
                swin_v1_s::<TestBackend>(&device).expect("Failed to create Swin-S model")
            }
            "swin_v1_b" => {
                swin_v1_b::<TestBackend>(&device).expect("Failed to create Swin-B model")
            }
            "swin_v1_l" => {
                swin_v1_l::<TestBackend>(&device).expect("Failed to create Swin-L model")
            }
            _ => panic!("Unknown model variant: {}", model_variant),
        };

        let input = Tensor::<TestBackend, 4>::random(
            [1, 3, 224, 224],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let outputs = model.forward(input).expect("Forward pass failed");

        assert_eq!(outputs.len(), 4);

        let expected_spatial_sizes = [56, 28, 14, 7];

        for (stage, output) in outputs.iter().enumerate() {
            let expected_shape = [
                1,
                expected_channels[stage],
                expected_spatial_sizes[stage],
                expected_spatial_sizes[stage],
            ];
            assert_eq!(
                output.shape().dims,
                expected_shape,
                "Stage {} output shape mismatch for {}: expected {:?}, got {:?}",
                stage,
                model_variant,
                expected_shape,
                output.shape().dims
            );
        }
    }

    #[rstest]
    #[case(224, 224, 1)] // Standard ImageNet size, batch=1
    #[case(320, 320, 1)] // Larger input, batch=1
    #[case(128, 128, 4)] // Smaller input, larger batch
    #[case(192, 256, 1)] // Non-square input, batch=1
    fn swin_transformer_handles_various_input_sizes(
        #[case] height: usize,
        #[case] width: usize,
        #[case] batch_size: usize,
    ) {
        let device = Default::default();
        let model = swin_v1_t::<TestBackend>(&device).expect("Failed to create Swin-T model");

        let input = Tensor::<TestBackend, 4>::random(
            [batch_size, 3, height, width],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let outputs = model.forward(input).expect("Forward pass failed");

        // Should always produce 4 outputs
        assert_eq!(outputs.len(), 4);

        // Check that outputs have reasonable shapes
        for (stage, output) in outputs.iter().enumerate() {
            let dims = output.shape().dims;
            assert_eq!(
                dims[0], batch_size,
                "Batch size should be preserved at stage {}",
                stage
            );
            assert!(
                dims[1] > 0,
                "Channels should be positive at stage {}",
                stage
            );
            assert!(dims[2] > 0, "Height should be positive at stage {}", stage);
            assert!(dims[3] > 0, "Width should be positive at stage {}", stage);

            // Verify that spatial dimensions are reasonable (should be getting smaller)
            if stage > 0 {
                let prev_spatial =
                    outputs[stage - 1].shape().dims[2] * outputs[stage - 1].shape().dims[3];
                let curr_spatial = dims[2] * dims[3];
                assert!(curr_spatial <= prev_spatial,
                    "Spatial dimensions should decrease between stages: stage {} = {}, stage {} = {}", 
                    stage - 1, prev_spatial, stage, curr_spatial);
            }
        }

        // Verify the expected channel progression for Swin-T: [96, 192, 384, 768]
        let expected_channels = [96, 192, 384, 768];
        for (stage, expected) in expected_channels.iter().enumerate() {
            assert_eq!(
                outputs[stage].shape().dims[1],
                *expected,
                "Stage {} should have {} channels, got {}",
                stage,
                expected,
                outputs[stage].shape().dims[1]
            );
        }
    }
}
