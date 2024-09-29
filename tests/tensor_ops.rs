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

#[test]
fn reduce_sum() {
    let x: Tensor<2> = Tensor::from_vec(vec![10.0f32; 100], [25, 4], DEVICE);
    let sum_all: Tensor<1> = x.clone().reduce_sum(2);
    let sum_dim1: Tensor<2> = x.reduce_sum(1);

    assert_ulps_eq!(sum_all.into_scalar::<f32>(), 1000.0);
    assert_ulps_eq!(sum_dim1.into_vec::<f32>().as_slice(), &[40.0f32; 25][..]);
}

#[test]
fn reduce_mean() {
    let x: Tensor<2> = Tensor::from_vec(vec![10.0f32; 100], [25, 4], DEVICE);
    let mean_all: Tensor<1> = x.clone().reduce_mean(2);
    let mean_dim1: Tensor<2> = x.reduce_mean(1);

    assert_ulps_eq!(mean_all.into_scalar::<f32>(), 10.0);
    assert_ulps_eq!(mean_dim1.into_vec::<f32>().as_slice(), &[10.0f32; 25][..]);
}

#[test]
fn reduce_max() {
    let x: Tensor<2> = Tensor::from_vec(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        [3, 4],
        DEVICE,
    );
    let max_all: Tensor<1> = x.clone().reduce_max(2);
    let max_dim1: Tensor<2> = x.reduce_max(1);

    assert_ulps_eq!(max_all.into_scalar::<f32>(), 12.0);
    assert_ulps_eq!(max_dim1.into_vec::<f32>().as_slice(), &[4.0, 8.0, 12.0][..]);
}

#[test]
fn reduce_min() {
    let x: Tensor<2> = Tensor::from_vec(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        [3, 4],
        DEVICE,
    );
    let min_all: Tensor<1> = x.clone().reduce_min(2);
    let min_dim1: Tensor<2> = x.reduce_min(1);

    assert_ulps_eq!(min_all.into_scalar::<f32>(), 1.0);
    assert_ulps_eq!(min_dim1.into_vec::<f32>().as_slice(), &[1.0, 5.0, 9.0][..]);
}
