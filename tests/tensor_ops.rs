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

#[test]
fn multiply() {
    let a = Tensor::from_vec(vec![2.0f32; 100], [50, 2], DEVICE);
    let b = Tensor::from_vec(vec![bf16::from_f32(3.0); 100], [50, 2], DEVICE);

    let result = (a * b).into_vec::<f32>();

    assert_ulps_eq!(result.as_slice(), &[6.0f32; 100][..]);
}

#[test]
fn divide() {
    let a = Tensor::from_vec(vec![6.0f32; 100], [50, 2], DEVICE);
    let b = Tensor::from_vec(vec![bf16::from_f32(2.0); 100], [50, 2], DEVICE);

    let result = (a / b).into_vec::<f32>();

    assert_ulps_eq!(result.as_slice(), &[3.0f32; 100][..]);
}

#[test]
fn multiply_by_scalar() {
    let a = Tensor::from_vec(vec![2.0f32; 100], [50, 2], DEVICE);
    let scalar = 3.0f32;

    let result = (a * scalar).into_vec::<f32>();

    assert_ulps_eq!(result.as_slice(), &[6.0f32; 100][..]);
}

#[test]
fn divide_by_scalar() {
    let a = Tensor::from_vec(vec![6.0f32; 100], [50, 2], DEVICE);
    let scalar = 2.0f32;

    let result = (a / scalar).into_vec::<f32>();

    assert_ulps_eq!(result.as_slice(), &[3.0f32; 100][..]);
}

#[test]
fn recip() {
    let a = Tensor::from_vec(vec![2.0f32; 100], [50, 2], DEVICE);

    let result = a.recip().into_vec::<f32>();

    assert_ulps_eq!(result.as_slice(), &[0.5f32; 100][..]);
}

#[test]
fn complex_operation_chain() {
    let a = Tensor::from_vec(vec![1.0f32; 100], [50, 2], DEVICE);
    let b = Tensor::from_vec(vec![2.0f32; 100], [50, 2], DEVICE);
    let c = Tensor::from_vec(vec![0.5f32; 100], [50, 2], DEVICE);
    let d = Tensor::from_vec(vec![3.0f32; 100], [50, 2], DEVICE);

    // result = (a + b) * (c - d).recip() / 2 + a
    let result = ((a.clone() + b) * (c - d).recip() / 2.0 + a).into_vec::<f32>();

    let expected = vec![0.4f32; 100];

    assert_ulps_eq!(result.as_slice(), expected.as_slice(), epsilon = 1e-6);
}
