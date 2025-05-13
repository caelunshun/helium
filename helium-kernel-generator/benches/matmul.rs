use cudarc::{
    cublaslt::{CudaBlasLT, Matmul as _, MatmulConfig},
    driver::{CudaContext, DevicePtr, DevicePtrMut, sys::CUevent_flags},
};
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
use std::{sync::Arc, time::Duration};

const WARMUP: u32 = 1;
const ITERATIONS: u32 = 1;

fn main() {
    let device = CudaContext::new(0).unwrap();
    println!(
        "Device architecture: {:?}",
        Architecture::from_device(&device).unwrap()
    );

    for (m, n, k, precision) in [
        (4096, 4096, 4096, Precision::MulTf32AccumF32),
        (4096, 4096, 4096, Precision::MulBf16AccumF32),
        (4096, 4096, 4096, Precision::MulF16AccumF32),
    ] {
        let helium_time = bench_generated_matmul(&device, m, n, k, precision);
        let cublaslt_time = bench_cublaslt_matmul(&device, m, n, k, precision);
        let speedup = cublaslt_time.as_secs_f64() / helium_time.as_secs_f64();

        println!(
            "Matmul {m}x{n}x{k} with precision {precision:?}: helium = {helium_time:.2?}, cublasLt = {cublaslt_time:2?}, speedup = {speedup:.2}x"
        );
    }
}

fn bench_generated_matmul(
    device: &Arc<CudaContext>,
    m: u32,
    n: u32,
    k: u32,
    precision: Precision,
) -> Duration {
    let input_type = match precision {
        Precision::MulBf16AccumF32 => DataType::Bf16,
        Precision::MulF16AccumF32 | Precision::MulF16AccumF16 => DataType::F16,
        Precision::MulTf32AccumF32 | Precision::MulF32AccumF32 => DataType::F32,
        Precision::MulF8AccumF32 { .. } | Precision::MulF8AccumF16 { .. } => unimplemented!(),
    };
    let output_type = input_type;

    let mut opgraph = OpGraph::new();
    let input_a = opgraph.new_input(Descriptor {
        shape: Shape::new([m as usize, k as usize]),
        data_type: input_type,
    });
    let input_b = opgraph.new_input(Descriptor {
        shape: Shape::new([n as usize, k as usize]),
        data_type: input_type,
    });
    let matmul = opgraph.new_op(Op::Matmul(Matmul {
        input_a,
        input_b,
        precision,
    }));
    let cvt = opgraph.new_op(Op::ChangeDataType(ChangeDataType {
        input: matmul,
        target_type: output_type,
    }));
    opgraph.new_output(cvt);
    let op_subgraph = OpSubgraph::from_nodes(&Arc::new(opgraph), vec![matmul, cvt]);

    let kernel = MatmulGenerator::new(&op_subgraph)
        .unwrap()
        .generate(Architecture::from_device(device).unwrap())
        .unwrap();
    let module = kernel.load_on_device(device).unwrap();

    let stream = device.new_stream().unwrap();
    let mut input_a = stream
        .alloc_zeros::<u8>(m as usize * k as usize * input_type.size_in_bits() / 8)
        .unwrap();
    let mut input_b = stream
        .alloc_zeros::<u8>(n as usize * k as usize * input_type.size_in_bits() / 8)
        .unwrap();
    let mut output = stream
        .alloc_zeros::<u8>(n as usize * m as usize * output_type.size_in_bits() / 8)
        .unwrap();

    let host_input_a = (0..input_a.len())
        .map(|_| rand::random::<u8>())
        .collect::<Vec<_>>();
    let host_input_b = (0..input_b.len())
        .map(|_| rand::random::<u8>())
        .collect::<Vec<_>>();
    stream.memcpy_htod(&host_input_a, &mut input_a).unwrap();
    stream.memcpy_htod(&host_input_b, &mut input_b).unwrap();

    let start = device
        .new_event(Some(CUevent_flags::CU_EVENT_BLOCKING_SYNC))
        .unwrap();
    let end = device
        .new_event(Some(CUevent_flags::CU_EVENT_BLOCKING_SYNC))
        .unwrap();

    // warmup
    for _ in 0..WARMUP {
        unsafe {
            let (input_a, _guard) = input_a.device_ptr(&stream);
            let (input_b, _guard) = input_b.device_ptr(&stream);
            let (output, _guard) = output.device_ptr_mut(&stream);
            kernel
                .execute(&module, &stream, input_a, input_b, |node| {
                    if node == cvt { output } else { unreachable!() }
                })
                .unwrap();
        }
    }
    stream.synchronize().unwrap();

    start.record(&stream).unwrap();

    for _ in 0..ITERATIONS {
        unsafe {
            let (input_a, _guard) = input_a.device_ptr(&stream);
            let (input_b, _guard) = input_b.device_ptr(&stream);
            let (output, _guard) = output.device_ptr_mut(&stream);
            kernel
                .execute(&module, &stream, input_a, input_b, |node| {
                    if node == cvt { output } else { unreachable!() }
                })
                .unwrap();
        }
    }
    end.record(&stream).unwrap();

    end.synchronize().unwrap();
    let time = start.elapsed_ms(&end).unwrap();
    Duration::from_secs_f64(f64::from(time) / 1000.0 / ITERATIONS as f64)
}

fn bench_cublaslt_matmul(
    device: &Arc<CudaContext>,
    m: u32,
    n: u32,
    k: u32,
    precision: Precision,
) -> Duration {
    let stream = device.new_stream().unwrap();
    let cublas = CudaBlasLT::new(stream.clone()).unwrap();

    let time = match precision {
        Precision::MulTf32AccumF32 => {
            let mut input_a = stream.alloc_zeros::<f32>(m as usize * k as usize).unwrap();
            let mut input_b = stream.alloc_zeros::<f32>(n as usize * k as usize).unwrap();
            let mut output = stream.alloc_zeros::<f32>(n as usize * m as usize).unwrap();
            let host_input_a = (0..input_a.len())
                .map(|_| rand::random::<f32>())
                .collect::<Vec<_>>();
            let host_input_b = (0..input_b.len())
                .map(|_| rand::random::<f32>())
                .collect::<Vec<_>>();
            stream.memcpy_htod(&host_input_a, &mut input_a).unwrap();
            stream.memcpy_htod(&host_input_b, &mut input_b).unwrap();

            // warmup
            for _ in 0..WARMUP {
                unsafe {
                    cublas
                        .matmul(
                            MatmulConfig {
                                transa: true,
                                transb: true,
                                transc: false, // TODO wrong
                                m: m as _,
                                n: n as _,
                                k: k as _,
                                alpha: 1.0,
                                lda: m as _,
                                ldb: n as _,
                                beta: 0.0,
                                ldc: m as _,
                                stride_a: None,
                                stride_b: None,
                                stride_c: None,
                                stride_bias: None,
                                batch_size: None,
                            },
                            &input_a,
                            &input_b,
                            &mut output,
                            None,
                            None,
                        )
                        .unwrap();
                }
            }

            stream.synchronize().unwrap();
            let start = stream
                .record_event(Some(CUevent_flags::CU_EVENT_BLOCKING_SYNC))
                .unwrap();
            for _ in 0..ITERATIONS {
                unsafe {
                    cublas
                        .matmul(
                            MatmulConfig {
                                transa: true,
                                transb: true,
                                transc: false, // TODO wrong
                                m: m as _,
                                n: n as _,
                                k: k as _,
                                alpha: 1.0,
                                lda: m as _,
                                ldb: n as _,
                                beta: 0.0,
                                ldc: m as _,
                                stride_a: None,
                                stride_b: None,
                                stride_c: None,
                                stride_bias: None,
                                batch_size: None,
                            },
                            &input_a,
                            &input_b,
                            &mut output,
                            None,
                            None,
                        )
                        .unwrap();
                }
            }
            let end = stream
                .record_event(Some(CUevent_flags::CU_EVENT_BLOCKING_SYNC))
                .unwrap();
            end.synchronize().unwrap();
            start.elapsed_ms(&end).unwrap()
        }
        Precision::MulBf16AccumF32 => {
            let mut input_a = stream.alloc_zeros::<bf16>(m as usize * k as usize).unwrap();
            let mut input_b = stream.alloc_zeros::<bf16>(n as usize * k as usize).unwrap();
            let mut output = stream.alloc_zeros::<bf16>(n as usize * m as usize).unwrap();
            let host_input_a = (0..input_a.len())
                .map(|_| rand::random::<bf16>())
                .collect::<Vec<_>>();
            let host_input_b = (0..input_b.len())
                .map(|_| rand::random::<bf16>())
                .collect::<Vec<_>>();
            stream.memcpy_htod(&host_input_a, &mut input_a).unwrap();
            stream.memcpy_htod(&host_input_b, &mut input_b).unwrap();

            // warmup
            for _ in 0..WARMUP {
                unsafe {
                    cublas
                        .matmul(
                            MatmulConfig {
                                transa: true,
                                transb: true,
                                transc: false, // TODO wrong
                                m: m as _,
                                n: n as _,
                                k: k as _,
                                alpha: 1.0,
                                lda: m as _,
                                ldb: n as _,
                                beta: 0.0,
                                ldc: m as _,
                                stride_a: None,
                                stride_b: None,
                                stride_c: None,
                                stride_bias: None,
                                batch_size: None,
                            },
                            &input_a,
                            &input_b,
                            &mut output,
                            None,
                            None,
                        )
                        .unwrap();
                }
            }

            stream.synchronize().unwrap();
            let start = stream
                .record_event(Some(CUevent_flags::CU_EVENT_BLOCKING_SYNC))
                .unwrap();
            for _ in 0..ITERATIONS {
                unsafe {
                    cublas
                        .matmul(
                            MatmulConfig {
                                transa: true,
                                transb: true,
                                transc: false, // TODO wrong
                                m: m as _,
                                n: n as _,
                                k: k as _,
                                alpha: 1.0,
                                lda: m as _,
                                ldb: n as _,
                                beta: 0.0,
                                ldc: m as _,
                                stride_a: None,
                                stride_b: None,
                                stride_c: None,
                                stride_bias: None,
                                batch_size: None,
                            },
                            &input_a,
                            &input_b,
                            &mut output,
                            None,
                            None,
                        )
                        .unwrap();
                }
            }
            let end = stream
                .record_event(Some(CUevent_flags::CU_EVENT_BLOCKING_SYNC))
                .unwrap();
            end.synchronize().unwrap();
            start.elapsed_ms(&end).unwrap()
        }
        Precision::MulF16AccumF16 | Precision::MulF16AccumF32 => {
            let mut input_a = stream.alloc_zeros::<f16>(m as usize * k as usize).unwrap();
            let mut input_b = stream.alloc_zeros::<f16>(n as usize * k as usize).unwrap();
            let mut output = stream.alloc_zeros::<f16>(n as usize * m as usize).unwrap();
            let host_input_a = (0..input_a.len())
                .map(|_| rand::random::<f16>())
                .collect::<Vec<_>>();
            let host_input_b = (0..input_b.len())
                .map(|_| rand::random::<f16>())
                .collect::<Vec<_>>();
            stream.memcpy_htod(&host_input_a, &mut input_a).unwrap();
            stream.memcpy_htod(&host_input_b, &mut input_b).unwrap();

            // warmup
            for _ in 0..WARMUP {
                unsafe {
                    cublas
                        .matmul(
                            MatmulConfig {
                                transa: true,
                                transb: true,
                                transc: false, // TODO wrong
                                m: m as _,
                                n: n as _,
                                k: k as _,
                                alpha: 1.0,
                                lda: m as _,
                                ldb: n as _,
                                beta: 0.0,
                                ldc: m as _,
                                stride_a: None,
                                stride_b: None,
                                stride_c: None,
                                stride_bias: None,
                                batch_size: None,
                            },
                            &input_a,
                            &input_b,
                            &mut output,
                            None,
                            None,
                        )
                        .unwrap();
                }
            }

            stream.synchronize().unwrap();
            let start = stream
                .record_event(Some(CUevent_flags::CU_EVENT_BLOCKING_SYNC))
                .unwrap();
            for _ in 0..ITERATIONS {
                unsafe {
                    cublas
                        .matmul(
                            MatmulConfig {
                                transa: true,
                                transb: true,
                                transc: false, // TODO wrong
                                m: m as _,
                                n: n as _,
                                k: k as _,
                                alpha: 1.0,
                                lda: m as _,
                                ldb: n as _,
                                beta: 0.0,
                                ldc: m as _,
                                stride_a: None,
                                stride_b: None,
                                stride_c: None,
                                stride_bias: None,
                                batch_size: None,
                            },
                            &input_a,
                            &input_b,
                            &mut output,
                            None,
                            None,
                        )
                        .unwrap();
                }
            }
            let end = stream
                .record_event(Some(CUevent_flags::CU_EVENT_BLOCKING_SYNC))
                .unwrap();
            end.synchronize().unwrap();
            start.elapsed_ms(&end).unwrap()
        }
        _ => unimplemented!(),
    };
    Duration::from_secs_f64(f64::from(time) / 1000.0 / ITERATIONS as f64)
}
