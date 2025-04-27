use approx::assert_ulps_eq;
use half::{bf16, f16};
use helium::{
    DataType, Device, Param, Tensor,
    op::conv::{Conv2dParams, PaddingMode},
};
use helium_ir::opgraph::op::precision::Precision;

const DEVICE: Device = Device::Cuda(0);

#[test]
fn upload_download() {
    let tensor = Tensor::from_slice(vec![5.0f32; 100], [25, 4], DEVICE);
    assert_eq!(tensor.shape(), [25, 4]);
    assert_eq!(tensor.to_vec::<f32>(), vec![5.0f32; 100],);
}

#[test]
fn add_simple() {
    let a = Tensor::from_slice(vec![2.0f32; 100], [50, 2], DEVICE);
    let b = Tensor::from_slice(vec![bf16::ONE; 100], [50, 2], DEVICE);

    let result = (a + b).to_vec::<f32>();

    assert_ulps_eq!(result.as_slice(), &[3.0f32; 100][..]);
}

#[test]
fn add_with_broadcast() {
    let a = Tensor::<2>::from_array([[3.0f32; 5]; 5], DEVICE);
    let b = Tensor::<1>::from_array([1.0f32], DEVICE);

    let result = (a + b.broadcast_to([5, 5])).to_vec::<f32>();

    assert_ulps_eq!(result.as_slice(), &[4.0f32; 25][..]);
}

#[test]
fn multiply() {
    let a = Tensor::from_slice(vec![2.0f32; 100], [50, 2], DEVICE);
    let b = Tensor::from_slice(vec![bf16::from_f32(3.0); 100], [50, 2], DEVICE);

    let result = (a * b).to_vec::<f32>();

    assert_ulps_eq!(result.as_slice(), &[6.0f32; 100][..]);
}

#[test]
fn divide() {
    let a = Tensor::from_slice(vec![6.0f32; 100], [50, 2], DEVICE);
    let b = Tensor::from_slice(vec![bf16::from_f32(2.0); 100], [50, 2], DEVICE);

    let result = (a / b).to_vec::<f32>();

    assert_ulps_eq!(result.as_slice(), &[3.0f32; 100][..]);
}

#[test]
fn multiply_by_scalar() {
    let a = Tensor::from_slice(vec![2.0f32; 100], [50, 2], DEVICE);
    let scalar = 3.0f32;

    let result = (a * scalar).to_vec::<f32>();

    assert_ulps_eq!(result.as_slice(), &[6.0f32; 100][..]);
}

#[test]
fn divide_by_scalar() {
    let a = Tensor::from_slice(vec![6.0f32; 100], [50, 2], DEVICE);
    let scalar = 2.0f32;

    let result = (a / scalar).to_vec::<f32>();

    assert_ulps_eq!(result.as_slice(), &[3.0f32; 100][..]);
}

#[test]
fn recip() {
    let a = Tensor::from_slice(vec![2.0f32; 100], [50, 2], DEVICE);

    let result = a.recip().to_vec::<f32>();

    assert_ulps_eq!(result.as_slice(), &[0.5f32; 100][..]);
}

#[test]
fn complex_operation_chain() {
    let a = Tensor::from_slice(vec![1.0f32; 100], [50, 2], DEVICE);
    let b = Tensor::from_slice(vec![2.0f32; 100], [50, 2], DEVICE);
    let c = Tensor::from_slice(vec![0.5f32; 100], [50, 2], DEVICE);
    let d = Tensor::from_slice(vec![3.0f32; 100], [50, 2], DEVICE);

    // result = (a + b) * (c - d).recip() / 2 + a
    let result = ((a.clone() + b) * (c - d).recip() / 2.0 + a).to_vec::<f32>();

    let expected = vec![0.4f32; 100];

    assert_ulps_eq!(result.as_slice(), expected.as_slice(), epsilon = 1e-6);
}

#[test]
fn reduce_sum() {
    let x: Tensor<2> = Tensor::from_slice(vec![10.0f32; 100], [25, 4], DEVICE);
    let sum_all: Tensor<1> = x.clone().reduce_sum(2);
    let sum_dim1: Tensor<2> = x.reduce_sum(1);

    assert_ulps_eq!(sum_all.to_scalar::<f32>(), 1000.0);
    assert_ulps_eq!(sum_dim1.to_vec::<f32>().as_slice(), &[40.0f32; 25][..]);
}

#[test]
fn reduce_mean() {
    let x: Tensor<2> = Tensor::from_slice(vec![10.0f32; 100], [25, 4], DEVICE);
    let mean_all: Tensor<1> = x.clone().reduce_mean(2);
    let mean_dim1: Tensor<2> = x.reduce_mean(1);

    assert_ulps_eq!(mean_all.to_scalar::<f32>(), 10.0);
    assert_ulps_eq!(mean_dim1.to_vec::<f32>().as_slice(), &[10.0f32; 25][..]);
}

#[test]
fn reduce_max() {
    let x: Tensor<2> = Tensor::from_slice(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        [3, 4],
        DEVICE,
    );
    let max_all: Tensor<1> = x.clone().reduce_max(2);
    let max_dim1: Tensor<2> = x.reduce_max(1);

    assert_ulps_eq!(max_all.to_scalar::<f32>(), 12.0);
    assert_ulps_eq!(max_dim1.to_vec::<f32>().as_slice(), &[4.0, 8.0, 12.0][..]);
}

#[test]
fn reduce_min() {
    let x: Tensor<2> = Tensor::from_slice(
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

    assert_ulps_eq!(min_all.to_scalar::<f32>(), 1.0);
    assert_ulps_eq!(min_dim1.to_vec::<f32>().as_slice(), &[1.0, 5.0, 9.0][..]);
}

#[test]
fn reduce_large() {
    let mut data = Vec::new();
    data.extend_from_slice(&[1.0f32; 100_000]);
    data.extend_from_slice(&[2.0f32; 100_000]);

    let x = Tensor::<3>::from_slice(data, [2, 1000, 100], DEVICE);

    let sum_all = x.clone().reduce_sum::<1>(3).to_scalar::<f32>();
    let sum_partial = x.reduce_sum::<2>(2).to_vec::<f32>();

    assert_ulps_eq!(sum_all, 300_000.0f32);
    assert_ulps_eq!(sum_partial.as_slice(), &[100_000.0f32, 200_000.0][..]);
}

#[test]
fn matmul_simple() {
    let a = Tensor::<2>::from_array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]], DEVICE)
        .transpose();
    let b = Tensor::<2>::from_array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]], DEVICE).transpose();

    let result = a
        .matmul(b, Precision::MulTf32AccumF32)
        .transpose()
        .to_vec::<f32>();

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

    let result = a.matmul(b, Precision::MulBf16AccumF32).to_vec::<f32>();

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

    let result = a
        .matmul(b, Precision::MulF16AccumF32)
        .transpose()
        .to_vec::<f32>();

    assert_ulps_eq!(
        result.as_slice(),
        &[9.0, 12.0, 15.0, 19.0, 26.0, 33.0, 29.0, 40.0, 51.0][..],
        epsilon = 1e-3
    );
}

#[test]
fn matmul_large_matrices() {
    let a = Tensor::<2>::from_slice(vec![1.0f32; 10000], [100, 100], DEVICE);
    let b = Tensor::<2>::from_slice(vec![0.5f32; 10000], [100, 100], DEVICE);

    let result = a.matmul(b, Precision::MulTf32AccumF32).to_vec::<f32>();

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

    let result = a.matmul(b, Precision::MulBf16AccumF32).transpose();
    assert_eq!(result.shape(), [2, 3, 3]);
    let result = result.to_vec::<f32>();

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
        result.to_vec::<f32>().as_slice(),
        &[
            1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0
        ][..]
    );
}

#[test]
fn reshape() {
    let x: Tensor<1> = Tensor::<1>::from_array([1.0, 2.0, 3.0, 4.0], DEVICE);
    let result: Tensor<2> = x.reshape([2, 2]);

    assert_eq!(result.shape(), [2, 2]);
    assert_eq!(result.to_vec::<f32>().as_slice(), &[1.0, 2.0, 3.0, 4.0][..]);
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
    let result = x.transpose().to_vec::<f32>();
    assert_eq!(
        result,
        &[
            1.0, 4.0, 2.0, 5.0, 3.0, 6.0, 7.0, 10.0, 8.0, 11.0, 9.0, 12.0
        ]
    );
}

#[test]
fn swap_dims() {
    let x = Tensor::<4>::from_array(
        [
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            [[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]],
        ],
        DEVICE,
    );
    let result = x.swap_dims(1, 2).to_vec::<f32>();
    assert_eq!(
        result,
        &[
            1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0, 9.0, 10.0, 13.0, 14.0, 11.0, 12.0, 15.0, 16.0
        ]
    );
}

#[test]
fn conv_simple() {
    let x = Tensor::<1>::from_array([1.0, 2.0], DEVICE).reshape([1, 1, 1, 2]);
    let w = Tensor::<1>::from_array([10.0, 20.0, 30.0, 40.0], DEVICE).reshape([2, 1, 1, 2]);
    let y = x.conv2d(
        &w,
        Conv2dParams {
            in_channels: 2,
            out_channels: 2,
            kernel_size: [1, 1],
            stride: [1, 1],
            dilation: [1, 1],
            padding_mode: PaddingMode::Same,
        },
    );
    assert_eq!(y.shape(), [1, 1, 1, 2]);

    assert_ulps_eq!(y.to_vec::<f32>().as_slice(), &[50.0, 110.0,][..]);
}

#[test]
fn relu() {
    let x = Tensor::<1>::from_array([-1.0, -0.5, 1.0, 0.0], DEVICE);
    assert_eq!(x.relu().to_vec::<f32>(), vec![0.0, 0.0, 1.0, 0.0]);
}

#[test]
fn test_2d_tensor_swap() {
    let x = Tensor::<2>::from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], DEVICE);
    let result = x.swap_dims(0, 1).to_vec::<f32>();
    assert_eq!(result, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_3d_tensor_swap() {
    let x = Tensor::<3>::from_array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], DEVICE);

    let result1 = x.swap_dims(0, 1).to_vec::<f32>();
    assert_eq!(result1, &[1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);

    let result2 = x.swap_dims(1, 2).to_vec::<f32>();
    assert_eq!(result2, &[1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0]);
}

#[test]
fn test_chained_swaps() {
    let x = Tensor::<3>::from_array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], DEVICE);

    let result = x
        .swap_dims(0, 1)
        .swap_dims(1, 2)
        .swap_dims(0, 2)
        .detach()
        .to_vec::<f32>();
    assert_eq!(result, &[1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0]);
}

#[test]
fn test_same_dimension_swap() {
    let x = Tensor::<3>::from_array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], DEVICE);

    let result = x.swap_dims(1, 1).to_vec::<f32>();
    assert_eq!(result, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn test_different_sized_dimensions() {
    let x = Tensor::<3>::from_array([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]], DEVICE);

    let result = x.swap_dims(1, 2).to_vec::<f32>();
    assert_eq!(result, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_singleton_dimension_swap() {
    let x = Tensor::<4>::from_array([[[[1.0], [2.0]], [[3.0], [4.0]]]], DEVICE);

    let result = x.swap_dims(0, 3).to_vec::<f32>();
    assert_eq!(result, &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn multiple_iterations() {
    let mut mean: Tensor<1> = Tensor::<1>::from_array([1.0f32; 10], DEVICE);
    mean = Param::new(mean).into_value();
    for _ in 0..10 {
        let x = Tensor::<2>::from_array([[2.0f32; 20]; 10], DEVICE);
        let sample_mean = x.reduce_mean::<2>(1).reshape(mean.shape());
        mean = &mean * 0.9 + &sample_mean * 0.1;
        let y = x.pow_scalar(2.0);
        y.to_vec::<f32>();
    }
    mean.to_vec::<f32>();
}

#[test]
fn constant() {
    let x = Tensor::from_constant(10.0f32, [10, 20], DEVICE);
    assert_eq!(x.shape(), [10, 20]);
    assert_eq!(x.data_type(), DataType::F32);
    assert_eq!(x.to_vec::<f32>(), vec![10.0f32; 200]);

    let y = Tensor::from_constant(bf16::from_f32(200.0), [2, 3], DEVICE);
    assert_eq!(y.shape(), [2, 3]);
    assert_eq!(y.data_type(), DataType::Bf16);
    assert_eq!(y.to_vec::<bf16>(), vec![bf16::from_f32(200.0); 6]);
}

#[test]
fn max() {
    let a = Tensor::from_constant(-1.0f32, [3], DEVICE);
    let b = Tensor::from_slice(&[-3.0, -1.0, 2.0][..], [3], DEVICE);

    let result = a.max(b).to_vec::<f32>();
    assert_eq!(result, &[-1.0, -1.0, 2.0]);
}

#[test]
fn min() {
    let a = Tensor::from_constant(-1.0f32, [3], DEVICE);
    let b = Tensor::from_slice(&[-3.0, -1.0, 2.0][..], [3], DEVICE);

    let result = a.min(b).to_vec::<f32>();
    assert_eq!(result, &[-3.0, -1.0, -1.0]);
}
