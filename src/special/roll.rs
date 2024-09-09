use burn::prelude::*;

pub fn roll<B: Backend, const D: usize>(
    input: Tensor<B, D>,
    shifts: Vec<i64>,
    dims: Vec<usize>,
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
        let t1 = input.clone().narrow(dim, 0, start as usize);
        return Tensor::cat([t0, t1].to_vec(), dim);
    }
    roll_common(input, shifts, dims)
}

fn roll_common<B: Backend, const D: usize>(
    input: Tensor<B, D>,
    shifts: Vec<i64>,
    dims: Vec<usize>,
) -> Tensor<B, D> {
    if dims.len() <= 1 {
        panic!("dimension must be > 1");
    }
    // other checks ...
    let tail_shifts = shifts[1];
    let tail_dims = dims[1];
    let first_dim_rolled = roll::<B, D>(input, [shifts[0]].to_vec(), [dims[0]].to_vec());
    roll::<B, D>(
        first_dim_rolled,
        [tail_shifts].to_vec(),
        [tail_dims].to_vec(),
    )
}