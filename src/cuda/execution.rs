use crate::{
    cuda::{
        context::{CudaContext, CudaStream},
        cuda_data_type,
        error::CudaError,
        kernel::KernelParam,
        plan::{Instr, Plan},
        tensor::{Data, RawTensor},
    },
    data_type::DataType,
    opgraph::NodeId,
};
use bumpalo::Bump;
use cudarc::{
    cublas::sys::cublasOperation_t::CUBLAS_OP_T,
    cublaslt::sys::{
        cublasComputeType_t::{CUBLAS_COMPUTE_16F, CUBLAS_COMPUTE_32F_FAST_TF32},
        cublasLtMatmulDescAttributes_t::{
            CUBLASLT_MATMUL_DESC_EPILOGUE, CUBLASLT_MATMUL_DESC_TRANSA, CUBLASLT_MATMUL_DESC_TRANSB,
        },
        cudaDataType::CUDA_R_32F,
        cudaDataType_t::CUDA_R_16F,
    },
    driver,
    driver::{
        sys::{CUevent_flags, CUevent_wait_flags},
        DriverError, LaunchAsync, LaunchConfig,
    },
};
use half::f16;
use slotmap::SecondaryMap;
use std::{ffi::c_void, mem, ptr};

/// Maps nodes to the tensors containing their output data.
#[derive(Default)]
pub struct TensorMap {
    map: SecondaryMap<NodeId, RawTensor>,
}

impl TensorMap {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, node: NodeId, tensor: RawTensor) {
        self.map.insert(node, tensor);
    }

    pub fn remove(&mut self, node: NodeId) -> Option<RawTensor> {
        self.map.remove(node)
    }

    pub fn get(&self, node: NodeId) -> &RawTensor {
        &self.map[node]
    }
}

/// Executes a `Plan` in the given context.
///
/// Returns a `TensorMap` containing the outputs.
pub fn execute_plan(
    plan: &Plan,
    cx: &CudaContext,
    inputs: TensorMap,
) -> Result<TensorMap, CudaError> {
    let mut tensors = inputs;
    let mut bump = Bump::new();
    let streams = cx.stream_pool()?;
    let events = streams
        .iter()
        .map(|_| driver::result::event::create(CUevent_flags::CU_EVENT_DEFAULT))
        .collect::<Result<Vec<_>, DriverError>>()?;

    for (i, step) in plan.steps().enumerate() {
        if i != 0 {
            for stream in streams {
                for event in &events {
                    unsafe {
                        driver::result::stream::wait_event(
                            stream.raw(),
                            *event,
                            CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT,
                        )?;
                    }
                }
            }
        }

        for (i, instr) in step.instrs().enumerate() {
            let stream = &streams[i % streams.len()];
            execute_instr(instr, stream, cx, &bump, &mut tensors)?;
        }

        for (stream, event) in streams.iter().zip(&events) {
            unsafe {
                driver::result::event::record(*event, stream.raw())?;
            }
        }

        bump.reset();
    }

    Ok(tensors)
}

fn execute_instr(
    instr: &Instr,
    stream: &CudaStream,
    cx: &CudaContext,
    bump: &Bump,
    tensors: &mut TensorMap,
) -> Result<(), CudaError> {
    match instr {
        Instr::FreeTensor(node_id) => {
            tensors.remove(*node_id);
        }
        Instr::PointwiseKernel { kernel, output } => {
            let func = cx
                .device()
                .get_func(&kernel.module_name, kernel.func_name)
                .expect("pointwise module not found?");

            let first_input = tensors.get(kernel.first_tensor_input());
            let len = u32::try_from(first_input.num_elements()).expect("tensor too large");

            let mut output_tensor = RawTensor::new(
                unsafe { Data::alloc(cx.device(), kernel.output_type, len as usize)? },
                first_input.shape().to_vec(),
            );

            let mut params =
                build_params(tensors, &kernel.params, len, None, &mut output_tensor, bump);

            unsafe {
                func.launch_on_stream(stream.cudarc_stream(), launch_config(len), &mut params)?;
            }

            tensors.insert(*output, output_tensor);
        }
        Instr::ReductionKernel {
            kernel,
            reduction_depth,
            output,
        } => {
            let func = cx
                .device()
                .get_func(&kernel.module_name, kernel.func_name)
                .expect("reduction module not found?");

            let first_input = tensors.get(kernel.first_tensor_input());
            let len = u32::try_from(first_input.num_elements()).expect("tensor too large");

            let output_shape = first_input.shape()
                [..first_input.shape().len() - *reduction_depth as usize]
                .to_vec();
            let output_len = u32::try_from(output_shape.iter().copied().sum::<usize>()).unwrap();

            let reduction_stride = first_input.shape()
                [first_input.shape().len() - *reduction_depth as usize..]
                .iter()
                .copied()
                .sum::<usize>() as u32;

            let mut output_tensor = RawTensor::new(
                unsafe { Data::alloc(cx.device(), kernel.output_type, output_len as usize)? },
                output_shape,
            );

            let mut params = build_params(
                tensors,
                &kernel.params,
                len,
                Some(reduction_stride),
                &mut output_tensor,
                bump,
            );

            unsafe {
                func.launch_on_stream(
                    stream.cudarc_stream(),
                    launch_config_for_reduction(len),
                    &mut params,
                )?;
            }

            tensors.insert(*output, output_tensor);
        }
        Instr::Matmul(config) => {
            use cudarc::cublaslt::result as cublaslt;

            let a_input = tensors.get(config.a_input);
            let b_input = tensors.get(config.b_input);
            let bias_input = config.bias_input.map(|id| tensors.get(id));

            let (compute_type, scale_type) =
                if a_input.data_type() == DataType::F16 && b_input.data_type() == DataType::F16 {
                    (CUBLAS_COMPUTE_16F, CUDA_R_16F)
                } else {
                    (CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F)
                };

            let matmul_desc = cublaslt::create_matmul_desc(compute_type, scale_type)?;

            unsafe {
                cublaslt::set_matmul_desc_attribute(
                    matmul_desc,
                    CUBLASLT_MATMUL_DESC_EPILOGUE,
                    &config.epilogue as *const _ as *const _,
                    size_of_val(&config.epilogue),
                )?;

                if config.transpose_a {
                    cublaslt::set_matmul_desc_attribute(
                        matmul_desc,
                        CUBLASLT_MATMUL_DESC_TRANSA,
                        &CUBLAS_OP_T as *const _ as *const _,
                        size_of_val(&CUBLAS_OP_T),
                    )?;
                }
                if config.transpose_b {
                    cublaslt::set_matmul_desc_attribute(
                        matmul_desc,
                        CUBLASLT_MATMUL_DESC_TRANSB,
                        &CUBLAS_OP_T as *const _ as *const _,
                        size_of_val(&CUBLAS_OP_T),
                    )?;
                }
            }

            let a_layout = cublaslt::create_matrix_layout(
                cuda_data_type(a_input.data_type()),
                a_input.dim_at(-1) as u64,
                a_input.dim_at(-2) as u64,
                a_input.dim_at(-1) as i64,
            )?;

            let b_layout = cublaslt::create_matrix_layout(
                cuda_data_type(b_input.data_type()),
                b_input.dim_at(-1) as u64,
                b_input.dim_at(-2) as u64,
                b_input.dim_at(-1) as i64,
            )?;

            let c_layout = match bias_input {
                Some(bias) => {
                    cublaslt::create_matrix_layout(
                        cuda_data_type(bias.data_type()),
                        bias.dim_at(-1) as u64,
                        bias.dim_at(-2) as u64,
                        0, // column broadcast
                    )?
                }
                None => ptr::null_mut(),
            };

            let out_rows = a_input.dim_at(-1) as u64;
            let out_cols = a_input.dim_at(-2) as u64;
            let out_len = out_rows * out_cols;

            let d_layout = cublaslt::create_matrix_layout(
                cuda_data_type(config.output_type),
                out_rows,
                out_cols,
                out_rows as i64,
            )?;

            // TODO: batched matmul support
            assert_eq!(a_input.shape().len(), 2);
            assert_eq!(b_input.shape().len(), 2);

            let mut output_tensor = RawTensor::new(
                unsafe { Data::alloc(cx.device(), config.output_type, out_len as usize)? },
                vec![out_cols as usize, out_rows as usize],
            );

            let alpha = if scale_type == CUDA_R_16F {
                bump.alloc(f16::ONE) as *const _ as *const c_void
            } else {
                bump.alloc(1.0f32) as *const _ as *const c_void
            };
            let beta = if config.bias_input.is_some() {
                1.0
            } else {
                0.0
            };
            let beta = if scale_type == CUDA_R_16F {
                bump.alloc(f16::from_f32(beta)) as *const _ as *const c_void
            } else {
                bump.alloc(beta) as *const _ as *const c_void
            };

            unsafe {
                cublaslt::matmul(
                    cx.cublaslt_handle()?.raw(),
                    matmul_desc,
                    alpha,
                    beta,
                    a_input.data().device_ptr().cast(),
                    a_layout,
                    b_input.data().device_ptr().cast(),
                    b_layout,
                    bias_input.map_or(ptr::null(), |tensor| tensor.data().device_ptr().cast()),
                    c_layout,
                    output_tensor.data_mut().device_ptr_mut().cast(),
                    d_layout,
                    ptr::null(),
                    ptr::null_mut(),
                    0,
                    stream.raw().cast(),
                )?;
            }

            tensors.insert(config.output, output_tensor);
        }
    }
    Ok(())
}

fn build_params(
    tensors: &TensorMap,
    params: &[KernelParam],
    input_size: u32,
    reduction_stride: Option<u32>,
    output: &mut RawTensor,
    bump: &Bump,
) -> Vec<*mut c_void> {
    let mut raw_params = Vec::new();

    for param in params {
        let raw = match param {
            KernelParam::Node(node) => {
                bump.alloc(tensors.get(*node).data().device_ptr()) as *mut *const u8 as *mut c_void
            }
            KernelParam::Var(_) => todo!(),
            KernelParam::Output => {
                bump.alloc(output.data_mut().device_ptr_mut()) as *mut *mut u8 as *mut c_void
            }
            KernelParam::Size => bump.alloc(input_size) as *mut u32 as *mut c_void,
            KernelParam::ReductionStride => {
                bump.alloc(reduction_stride.unwrap()) as *mut u32 as *mut c_void
            }
        };
        raw_params.push(raw);
    }

    raw_params
}

const BLOCK_DIM: u32 = 256;

fn launch_config(len: u32) -> LaunchConfig {
    let grid_dim = (len + BLOCK_DIM - 1) / BLOCK_DIM;
    LaunchConfig {
        grid_dim: (grid_dim, 1, 1),
        block_dim: (BLOCK_DIM, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn launch_config_for_reduction(len: u32) -> LaunchConfig {
    LaunchConfig {
        shared_mem_bytes: BLOCK_DIM * mem::size_of::<f32>() as u32,
        ..launch_config(len)
    }
}
