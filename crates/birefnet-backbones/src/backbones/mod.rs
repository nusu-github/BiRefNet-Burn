pub mod pvt_v2;
pub mod resnet;
pub mod swin;
pub mod vgg;

pub(crate) fn usize_to_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).expect("dimension exceeds u32::MAX"))
}

pub(crate) fn usize_to_i64(value: usize) -> i64 {
    i64::try_from(value).expect("index exceeds i64::MAX")
}
