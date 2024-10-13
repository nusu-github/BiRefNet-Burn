use burn::prelude::*;

pub fn roll<B: Backend, const D: usize>(
    input: Tensor<B, D>,
    shifts: &[i64],
    dims: &[usize],
) -> Tensor<B, D> {
    if dims.len() == 1 {
        let dim = dims[0];
        let size = input.dims()[dim] as i64;
        let mut start = (size - shifts[0]) % size;
        if start < 0 {
            start += size;
        }
        let t0 = input
            .clone()
            .narrow(dim, start as usize, (size - start) as usize);
        let t1 = input.narrow(dim, 0, start as usize);
        return Tensor::cat(vec![t0, t1], dim);
    }
    roll_common(input, shifts, dims)
}

fn roll_common<B: Backend, const D: usize>(
    input: Tensor<B, D>,
    shifts: &[i64],
    dims: &[usize],
) -> Tensor<B, D> {
    assert!(dims.len() > 1, "dimension must be > 1");
    // other checks ...
    let tail_shifts = shifts[1];
    let tail_dims = dims[1];
    let first_dim_rolled = roll::<B, D>(input, &[shifts[0]], &[dims[0]]);
    roll::<B, D>(first_dim_rolled, &[tail_shifts], &[tail_dims])
}
