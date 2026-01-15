pub mod postprocessing;

pub use postprocessing::*;

#[cfg(test)]
mod tests {
    use burn::backend::Cpu;

    pub type TestBackend = Cpu;
}
