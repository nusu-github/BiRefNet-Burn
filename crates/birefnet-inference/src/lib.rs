pub mod postprocessing;

pub use postprocessing::*;

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;

    pub type TestBackend = NdArray;
}
