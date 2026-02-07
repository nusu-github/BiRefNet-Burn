pub mod postprocessing;

#[doc(inline)]
pub use postprocessing::{
    apply_threshold, fill_holes, gaussian_blur, morphological_closing, morphological_opening,
    postprocess_mask, remove_small_components, resize_tensor, tensor_to_image_data,
};

#[cfg(test)]
mod tests {
    use burn::backend::Cpu;

    pub type TestBackend = Cpu;
}
