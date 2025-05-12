use approx::relative_eq;
use cudarc::driver::{CudaContext, CudaSlice, DevicePtr, DevicePtrMut};
use faer::Mat;
use half::{bf16, f16};
use helium_ir::{
    data_type::DataType,
    opgraph::{
        Descriptor, OpGraph,
        op::{ChangeDataType, Matmul, Op, precision::Precision},
        subgraph::OpSubgraph,
    },
    shape::Shape,
};
use helium_kernel_generator::{architecture::Architecture, generators::matmul::MatmulGenerator};
use rstest::rstest;
use std::sync::Arc;

fn test_matmul(
    a: &Mat<f32>,
    b: &Mat<f32>,
    input_dtype_a: DataType,
    input_dtype_b: DataType,
    precision: Precision,
    output_dtype: DataType,
) {
    let mut opgraph = OpGraph::new();
    let input_a = opgraph.new_input(Descriptor {
        shape: Shape::new([a.nrows(), a.ncols()]),
        data_type: input_dtype_a,
    });
    let input_b = opgraph.new_input(Descriptor {
        shape: Shape::new([b.nrows(), b.ncols()]),
        data_type: input_dtype_b,
    });
    let matmul = opgraph.new_op(Op::Matmul(Matmul {
        input_a,
        input_b,
        precision,
    }));
    let output = opgraph.new_op(Op::ChangeDataType(ChangeDataType {
        input: matmul,
        target_type: output_dtype,
    }));
    opgraph.new_output(output);

    let kernel = MatmulGenerator::new(&OpSubgraph::from_nodes(
        &Arc::new(opgraph),
        vec![matmul, output],
    ))
    .unwrap()
    .generate(Architecture::Sm89) // TODO don't hardcode
    .unwrap();

    let device = CudaContext::new(0).unwrap();
    let stream = device.default_stream();
    let module = kernel.load_on_device(&device).unwrap();

    let mut device_a: CudaSlice<u8> = stream
        .alloc_zeros(a.nrows() * a.nrows() * input_dtype_a.size_in_bits() / 8)
        .unwrap();
    let host_a = (0..a.nrows())
        .flat_map(|i| (0..a.ncols()).map(move |j| (i, j)))
        .map(|(i, j)| a.get(i, j))
        .flat_map(|x| match input_dtype_a {
            DataType::Bf16 => bf16::from_f32(*x).to_le_bytes().to_vec(),
            DataType::F16 => f16::from_f32(*x).to_le_bytes().to_vec(),
            DataType::F32 => x.to_le_bytes().to_vec(),
            _ => unreachable!(),
        })
        .collect::<Vec<u8>>();
    stream.memcpy_htod(&host_a, &mut device_a).unwrap();

    let mut device_b: CudaSlice<u8> = stream
        .alloc_zeros(b.nrows() * b.ncols() * input_dtype_b.size_in_bits() / 8)
        .unwrap();
    let host_b = (0..b.nrows())
        .flat_map(|i| (0..b.ncols()).map(move |j| (i, j)))
        .map(|(i, j)| b.get(i, j))
        .flat_map(|x| match input_dtype_b {
            DataType::Bf16 => bf16::from_f32(*x).to_le_bytes().to_vec(),
            DataType::F16 => f16::from_f32(*x).to_le_bytes().to_vec(),
            DataType::F32 => x.to_le_bytes().to_vec(),
            _ => unreachable!(),
        })
        .collect::<Vec<u8>>();
    stream.memcpy_htod(&host_b, &mut device_b).unwrap();

    let mut device_out: CudaSlice<u8> = stream
        .alloc_zeros(a.nrows() * b.ncols() * output_dtype.size_in_bits() / 8)
        .unwrap();

    {
        let (device_a, _guard) = device_a.device_ptr(&stream);
        let (device_b, _guard) = device_b.device_ptr(&stream);
        let (device_out, _guard) = device_out.device_ptr_mut(&stream);

        unsafe {
            kernel
                .execute(&module, &stream, device_a, device_b, |node| {
                    if node == output {
                        device_out
                    } else {
                        panic!("not an output node")
                    }
                })
                .unwrap();
        }
    }

    stream.synchronize().unwrap();

    let host_out: Vec<u8> = stream.memcpy_dtov(&device_out).unwrap();
    let expected_out = a * b;

    for i in 0..expected_out.nrows() {
        for j in 0..expected_out.ncols() {
            let expected = *expected_out.get(i, j);
            let actual_bytes = &host_out[i * expected_out.ncols() * output_dtype.size_in_bits()
                / 8
                + j * output_dtype.size_in_bits() / 8..];
            let actual = match output_dtype {
                DataType::F16 => f16::from_le_bytes(actual_bytes[..2].try_into().unwrap()).to_f32(),
                DataType::Bf16 => {
                    bf16::from_le_bytes(actual_bytes[..2].try_into().unwrap()).to_f32()
                }
                DataType::F32 => f32::from_le_bytes(actual_bytes[..4].try_into().unwrap()),
                _ => unreachable!(),
            };

            if !relative_eq!(expected, actual, epsilon = 1e-3, max_relative = 1e-2) {
                panic!("failed at ({i}, {j}): expected = {expected}, actual = {actual}");
            }
        }
    }
}

#[rstest]
fn matmul_no_fusion(
    #[values(DataType::F32, DataType::Bf16, DataType::F16)] input_dtype_a: DataType,
    #[values(DataType::F32, DataType::Bf16, DataType::F16)] input_dtype_b: DataType,
    #[values(DataType::F32, DataType::Bf16, DataType::F16)] output_dtype: DataType,
    #[values(
        Precision::MulF32AccumF32,
        Precision::MulTf32AccumF32,
        Precision::MulBf16AccumF32,
        Precision::MulF16AccumF32,
        Precision::MulF16AccumF16
    )]
    precision: Precision,
) {
    let mat_a = Mat::from_fn(16, 16, |_, _| rand::random::<f32>());
    let mat_b = Mat::from_fn(16, 16, |_, _| rand::random::<f32>());

    test_matmul(
        &mat_a,
        &mat_b,
        input_dtype_a,
        input_dtype_b,
        precision,
        output_dtype,
    );
}
