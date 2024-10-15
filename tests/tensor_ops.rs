use approx::assert_ulps_eq;
use half::{bf16, f16};
use helium::{Device, Tensor};

const DEVICE: Device = Device::Cuda(0);

#[test]
fn upload_download() {
    let tensor = Tensor::from_vec(vec![5.0f32; 100], [25, 4], DEVICE);
    assert_eq!(tensor.shape(), [25, 4]);
    assert_eq!(tensor.into_vec::<f32>(), vec![5.0f32; 100],);
}

#[test]
fn add_simple() {
    let a = Tensor::from_vec(vec![2.0f32; 100], [50, 2], DEVICE);
    let b = Tensor::from_vec(vec![bf16::ONE; 100], [50, 2], DEVICE);

    let result = (a + b).into_vec::<f32>();

    assert_ulps_eq!(result.as_slice(), &[3.0f32; 100][..]);
}

#[test]
fn add_with_broadcast() {
    let a = Tensor::<2>::from_array([[3.0f32; 5]; 5], DEVICE);
    let b = Tensor::<1>::from_array([1.0f32], DEVICE);

    let result = (a + b.broadcast_to([5, 5])).into_vec::<f32>();

    assert_ulps_eq!(result.as_slice(), &[4.0f32; 25][..]);
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

    assert_eq!(min_all.shape(), [1]);
    assert_eq!(min_dim1.shape(), [3, 1]);

    assert_ulps_eq!(min_all.into_scalar::<f32>(), 1.0);
    assert_ulps_eq!(min_dim1.into_vec::<f32>().as_slice(), &[1.0, 5.0, 9.0][..]);
}

#[test]
fn reduce_large() {
    let mut data = Vec::new();
    data.extend_from_slice(&[1.0f32; 100_000]);
    data.extend_from_slice(&[2.0f32; 100_000]);

    let x = Tensor::<3>::from_vec(data, [2, 1000, 100], DEVICE);

    let sum_all = x.clone().reduce_sum::<1>(3).into_scalar::<f32>();
    let sum_partial = x.reduce_sum::<2>(2).into_vec::<f32>();

    assert_ulps_eq!(sum_all, 300_000.0f32);
    assert_ulps_eq!(sum_partial.as_slice(), &[100_000.0f32, 200_000.0][..]);
}

#[test]
fn matmul_simple() {
    let a = Tensor::<2>::from_array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]], DEVICE)
        .transpose();
    let b = Tensor::<2>::from_array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]], DEVICE).transpose();

    let result = a.matmul(b).transpose().into_vec::<f32>();

    assert_ulps_eq!(result.as_slice(), &[2.0, 4.0, 6.0, 4.0, 8.0, 12.0][..]);
}

#[test]
fn matmul_bf16() {
    let a = Tensor::<2>::from_array(
        [
            [bf16::from_f32(1.0), bf16::from_f32(2.0)],
            [bf16::from_f32(3.0), bf16::from_f32(4.0)],
        ],
        DEVICE,
    );
    let b = Tensor::<2>::from_array(
        [
            [bf16::from_f32(1.0), bf16::from_f32(2.0)],
            [bf16::from_f32(3.0), bf16::from_f32(4.0)],
        ],
        DEVICE,
    );

    let result = a.matmul(b).into_vec::<f32>();

    assert_ulps_eq!(
        result.as_slice(),
        &[7.0, 10.0, 15.0, 22.0][..],
        epsilon = 1e-3
    );
}

#[test]
fn matmul_f16() {
    let a = Tensor::<2>::from_array(
        [
            [f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)],
            [f16::from_f32(4.0), f16::from_f32(5.0), f16::from_f32(6.0)],
        ],
        DEVICE,
    )
    .transpose();
    let b = Tensor::<2>::from_array(
        [
            [f16::from_f32(1.0), f16::from_f32(2.0)],
            [f16::from_f32(3.0), f16::from_f32(4.0)],
            [f16::from_f32(5.0), f16::from_f32(6.0)],
        ],
        DEVICE,
    )
    .transpose();

    let result = a.matmul(b).transpose().into_vec::<f32>();

    assert_ulps_eq!(
        result.as_slice(),
        &[9.0, 12.0, 15.0, 19.0, 26.0, 33.0, 29.0, 40.0, 51.0][..],
        epsilon = 1e-3
    );
}

#[test]
fn matmul_large_matrices() {
    let a = Tensor::<2>::from_vec(vec![1.0f32; 10000], [100, 100], DEVICE);
    let b = Tensor::<2>::from_vec(vec![0.5f32; 10000], [100, 100], DEVICE);

    let result = a.matmul(b).into_vec::<f32>();

    assert_ulps_eq!(result.as_slice(), &[50.0f32; 10000][..], epsilon = 1e-3);
}

#[test]
fn matmul_batched() {
    let a = Tensor::<3>::from_array(
        [
            // first batch
            [
                [
                    bf16::from_f32(1.0),
                    bf16::from_f32(2.0),
                    bf16::from_f32(3.0),
                ],
                [
                    bf16::from_f32(4.0),
                    bf16::from_f32(5.0),
                    bf16::from_f32(6.0),
                ],
            ],
            // second batch
            [
                [
                    bf16::from_f32(7.0),
                    bf16::from_f32(8.0),
                    bf16::from_f32(9.0),
                ],
                [
                    bf16::from_f32(10.0),
                    bf16::from_f32(11.0),
                    bf16::from_f32(12.0),
                ],
            ],
        ],
        DEVICE,
    )
    .transpose();

    let b = Tensor::<3>::from_array(
        [
            // first batch
            [
                [bf16::from_f32(1.0), bf16::from_f32(2.0)],
                [bf16::from_f32(3.0), bf16::from_f32(4.0)],
                [bf16::from_f32(5.0), bf16::from_f32(6.0)],
            ],
            // second batch
            [
                [bf16::from_f32(7.0), bf16::from_f32(8.0)],
                [bf16::from_f32(9.0), bf16::from_f32(10.0)],
                [bf16::from_f32(11.0), bf16::from_f32(12.0)],
            ],
        ],
        DEVICE,
    )
    .transpose();

    let result = a.matmul(b).transpose().into_vec::<f32>();

    let expected = vec![
        9.0, 12.0, 15.0, 19.0, 26.0, 33.0, 29.0, 40.0, 51.0, // first batch
        129.0, 144.0, 159.0, 163.0, 182.0, 201.0, 197.0, 220.0, 243.0, // second batch
    ];

    assert_ulps_eq!(result.as_slice(), expected.as_slice(), epsilon = 1e-2);
}

#[test]
fn broadcast() {
    let x = Tensor::<2>::from_array([[1.0], [2.0]], DEVICE);
    let result = x.broadcast_to([2, 2, 4]);

    assert_eq!(result.shape(), [2, 2, 4]);
    assert_eq!(
        result.into_vec::<f32>().as_slice(),
        &[1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0][..]
    );
}

#[test]
fn reshape() {
    let x: Tensor<1> = Tensor::<1>::from_array([1.0, 2.0, 3.0, 4.0], DEVICE);
    let result: Tensor<2> = x.reshape([2, 2]);

    assert_eq!(result.shape(), [2, 2]);
    assert_eq!(
        result.into_vec::<f32>().as_slice(),
        &[1.0, 2.0, 3.0, 4.0][..]
    );
}

#[test]
fn transpose() {
    let x = Tensor::<3>::from_array(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ],
        DEVICE,
    );
    let result = x.transpose().into_vec::<f32>();
    assert_eq!(
        result,
        &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0, 7.0, 10.0, 8.0, 11.0, 9.0, 12.0]
    );
}
