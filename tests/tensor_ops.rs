use approx::assert_ulps_eq;
use half::bf16;
use helium::{Device, Tensor};

const DEVICE: Device = Device::Cuda(0);

#[test]
fn add() {
    let a = Tensor::from_vec(vec![2.0f32; 100], [50, 2], DEVICE);
    let b = Tensor::from_vec(vec![bf16::ONE; 100], [50, 2], DEVICE);

    let result = (a + b).into_vec::<f32>();

    assert_ulps_eq!(result.as_slice(), &[3.0f32; 100][..]);
}
