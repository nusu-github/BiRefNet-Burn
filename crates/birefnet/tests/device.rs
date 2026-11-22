//! Device and precision selection tests.

use birefnet::device::{AVAILABLE_DEVICES, Precision, configure_precision, select_device};

#[test]
fn unknown_device_lists_available_ones() {
    let err = select_device("tpu").unwrap_err();
    assert!(err.is_unsupported_device());
    let msg = err.to_string();
    assert!(msg.contains("unknown or disabled device `tpu`"), "{msg}");
    for name in AVAILABLE_DEVICES {
        assert!(msg.contains(name), "{msg}");
    }
}

#[test]
fn malformed_and_misplaced_indices_are_rejected() {
    let bad_index = select_device("cuda:x").unwrap_err();
    assert!(bad_index.is_invalid_index());
    assert!(std::error::Error::source(&bad_index).is_some());

    assert!(
        select_device("default:0")
            .unwrap_err()
            .is_unsupported_device()
    );
    assert!(
        select_device(String::from("flex:1"))
            .unwrap_err()
            .is_unsupported_device()
    );
}

#[test]
fn precision_round_trips_display_and_from_str() {
    for precision in [Precision::F32, Precision::F16, Precision::Bf16] {
        let parsed: Precision = precision.to_string().to_uppercase().parse().unwrap();
        assert_eq!(parsed, precision);
    }
    let err = "fp64".parse::<Precision>().unwrap_err();
    assert!(err.is_invalid_precision() && !err.is_configuration());
}

#[cfg(feature = "flex")]
#[test]
fn flex_precision_locks_after_first_allocation() {
    let mut device = select_device("flex").unwrap();
    assert_eq!(device, burn::tensor::Device::flex());
    configure_precision(&mut device, Precision::F32).unwrap();
    let _ = burn::tensor::Tensor::<1>::zeros([1], &device);
    let err = configure_precision(&mut device, Precision::F16).unwrap_err();
    assert!(err.is_configuration(), "{err}");
}

#[cfg(not(any(
    feature = "flex",
    feature = "cpu",
    feature = "wgpu",
    feature = "cuda",
    feature = "rocm"
)))]
#[test]
fn default_without_backends_is_an_error() {
    assert!(select_device("default").unwrap_err().is_no_backend());
    let _ = configure_precision;
}
