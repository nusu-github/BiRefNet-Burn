//! Window-based multi-head self-attention with relative position bias.

use birefnet_extra_ops::trunc_normal;
use burn::{
    module::{Param, ParamId},
    nn::{Dropout, DropoutConfig, Linear, LinearConfig},
    prelude::*,
    tensor::{
        activation::softmax,
        grid::{IndexPos, meshgrid_stack},
    },
};

use crate::backbones::{usize_to_f64, usize_to_i64};

/// Creates a 2D coordinate grid `[2, height, width]` matching `PyTorch`
/// `torch.meshgrid([coords_h, coords_w], indexing="ij")`.
fn create_coordinate_grid(height: usize, width: usize, device: &Device) -> Tensor<3> {
    let h_coords = Tensor::arange(0..usize_to_i64(height), device).float();
    let w_coords = Tensor::arange(0..usize_to_i64(width), device).float();
    meshgrid_stack::<2, 3, Float>(&[h_coords, w_coords], IndexPos::First)
}

/// Configuration for [`WindowAttention`].
#[derive(Config, Debug, PartialEq)]
pub struct WindowAttentionConfig {
    dim: usize,
    window_size: [usize; 2],
    num_heads: usize,
    #[config(default = true)]
    qkv_bias: bool,
    #[config(default = "None")]
    qk_scale: Option<f64>,
    #[config(default = 0.0)]
    attn_drop: f64,
    #[config(default = 0.0)]
    proj_drop: f64,
}

impl WindowAttentionConfig {
    /// Initializes a [`WindowAttention`] module.
    #[must_use]
    pub fn init(&self, device: &Device) -> WindowAttention {
        let head_dim = self.dim / self.num_heads;
        let table_len = (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1);
        let relative_position_bias_table = Param::from_tensor(trunc_normal(
            [table_len, self.num_heads],
            0.0,
            0.02,
            -2.0,
            2.0,
            device,
        ));

        let coords = create_coordinate_grid(self.window_size[0], self.window_size[1], device);
        let coords_flatten: Tensor<2> = coords.flatten(1, 2);
        let relative_coords: Tensor<3> =
            coords_flatten.clone().unsqueeze_dim::<3>(2) - coords_flatten.unsqueeze_dim::<3>(1);
        let relative_coords = relative_coords.permute([1, 2, 0]);

        let [num_positions, _, _] = relative_coords.dims();
        let h_coords: Tensor<2> = relative_coords
            .clone()
            .slice([0..num_positions, 0..num_positions, 0..1])
            .squeeze_dim(2);
        let w_coords: Tensor<2> = relative_coords
            .slice([0..num_positions, 0..num_positions, 1..2])
            .squeeze_dim(2);

        let h_coords_shifted = h_coords + usize_to_f64(self.window_size[0] - 1);
        let w_coords_shifted = w_coords + usize_to_f64(self.window_size[1] - 1);
        let h_index = h_coords_shifted * usize_to_f64(2 * self.window_size[1] - 1);
        let relative_position_index =
            Param::initialized(ParamId::new(), (h_index + w_coords_shifted).int());

        WindowAttention {
            dim: self.dim,
            window_size: self.window_size,
            num_heads: self.num_heads,
            scale: self
                .qk_scale
                .unwrap_or_else(|| usize_to_f64(head_dim).powf(-0.5)),
            relative_position_bias_table,
            relative_position_index,
            qkv: LinearConfig::new(self.dim, self.dim * 3)
                .with_bias(self.qkv_bias)
                .init(device),
            attn_drop: DropoutConfig::new(self.attn_drop).init(),
            proj: LinearConfig::new(self.dim, self.dim).init(device),
            proj_drop: DropoutConfig::new(self.proj_drop).init(),
        }
    }
}

/// Window-based multi-head self-attention with learnable relative position bias.
#[derive(Module, Debug)]
pub struct WindowAttention {
    dim: usize,
    window_size: [usize; 2],
    num_heads: usize,
    scale: f64,
    relative_position_bias_table: Param<Tensor<2>>,
    relative_position_index: Param<Tensor<2, Int>>,
    qkv: Linear,
    attn_drop: Dropout,
    proj: Linear,
    proj_drop: Dropout,
}

impl WindowAttention {
    /// Applies windowed multi-head self-attention to `[num_windows * B, Wh * Ww, C]`.
    #[must_use]
    pub fn forward(&self, x: Tensor<3>, mask: Option<Tensor<3>>) -> Tensor<3> {
        let [batch_windows, seq_len, channels] = x.dims();
        let head_dim = channels / self.num_heads;
        let qkv = self
            .qkv
            .forward(x)
            .reshape([batch_windows, seq_len, 3, self.num_heads, head_dim])
            .permute([2, 0, 3, 1, 4]);
        let query = qkv.clone().narrow(0, 0, 1).squeeze_dim::<4>(0);
        let key = qkv.clone().narrow(0, 1, 1).squeeze_dim::<4>(0);
        let value = qkv.narrow(0, 2, 1).squeeze_dim::<4>(0);

        let scaled_query = query * self.scale;
        let attn = scaled_query.matmul(key.swap_dims(2, 3));

        let indices = self.relative_position_index.val().flatten(0, 1);
        let window_area = self.window_size[0] * self.window_size[1];
        let relative_position_bias = self
            .relative_position_bias_table
            .val()
            .select(0, indices)
            .reshape([window_area, window_area, self.num_heads])
            .permute([2, 0, 1]);

        let attn = attn + relative_position_bias.unsqueeze();

        let attn = match mask {
            Some(mask) => {
                let [num_windows, mask_h, mask_w] = mask.dims();
                let attn = attn.reshape([
                    batch_windows / num_windows,
                    num_windows,
                    self.num_heads,
                    seq_len,
                    seq_len,
                ]);
                let mask = mask.reshape([1, num_windows, 1, mask_h, mask_w]);
                (attn + mask).reshape([batch_windows, self.num_heads, seq_len, seq_len])
            }
            None => attn,
        };

        let attn = self.attn_drop.forward(softmax(attn, 3));
        let out = attn
            .matmul(value)
            .swap_dims(1, 2)
            .reshape([batch_windows, seq_len, channels]);
        self.proj_drop.forward(self.proj.forward(out))
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::IntDType;

    use super::*;

    #[test]
    fn relative_position_index_matches_swin_formula_and_is_int() {
        let device = Device::flex();
        let attn = WindowAttentionConfig::new(8, [2, 2], 2).init(&device);
        let index = attn.relative_position_index.val();

        assert!(index.dtype().is_int());
        assert_eq!(index.dims(), [4, 4]);
        let values = index
            .into_data()
            .try_to_vec_as::<i64>()
            .expect("integer index should convert to i64");
        assert_eq!(values, vec![4, 3, 1, 0, 5, 4, 2, 1, 7, 6, 4, 3, 8, 7, 5, 4]);
    }

    #[test]
    fn window_attention_forward_accepts_i64_relative_position_index() {
        let device = Device::flex();
        let mut attn = WindowAttentionConfig::new(8, [2, 2], 2).init(&device);
        attn.relative_position_index = Param::initialized(
            ParamId::new(),
            attn.relative_position_index.val().cast(IntDType::I64),
        );

        let output = attn.forward(Tensor::zeros([1, 4, 8], &device), None);
        assert_eq!(output.dims(), [1, 4, 8]);
        let values = output
            .into_data()
            .try_to_vec_as::<f32>()
            .expect("output should convert to f32");
        assert!(values.iter().all(|v| v.is_finite()));
    }
}
