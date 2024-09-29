use crate::{
    cuda::{
        context::{CudaContext, CudaStream, LoadedKernel},
        cuda_data_type,
        error::CudaError,
        kernel::KernelParam,
        plan::{Instr, Plan},
        tensor::{Data, RawTensor},
    },
    data_type::{DataType, DataVec},
    opgraph::{NodeId, VarMap},
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
use half::{bf16, f16};
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

    pub fn get_mut(&mut self, node: NodeId) -> &mut RawTensor {
        &mut self.map[node]
    }
}

/// Executes a `Plan` in the given context.
///
/// Returns a `TensorMap` containing the outputs.
pub fn execute_plan(
    plan: &Plan,
    cx: &CudaContext,
    inputs: TensorMap,
    vars: &VarMap,
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
            execute_instr(instr, stream, cx, &bump, &mut tensors, vars)?;
        }

        for (stream, event) in streams.iter().zip(&events) {
            unsafe {
                driver::result::event::record(*event, stream.raw())?;
            }
        }

        bump.reset();
    }

    // PERF: use more fine-grained synchronization here
    for stream in streams {
        unsafe {
            driver::result::stream::synchronize(stream.raw())?;
        }
    }

    Ok(tensors)
}

fn execute_instr(
    instr: &Instr,
    stream: &CudaStream,
    cx: &CudaContext,
    bump: &Bump,
    tensors: &mut TensorMap,
    vars: &VarMap,
) -> Result<(), CudaError> {
    match instr {
        Instr::FreeTensor(node_id) => unsafe {
            tensors
                .remove(*node_id)
                .unwrap()
                .into_data()
                .free_async(stream)?;
        },
        Instr::PointwiseKernel { kernel } => {
            let func = cx
                .device()
                .get_func(&kernel.module_name, kernel.func_name)
                .expect("pointwise module not found?");

            let first_input = tensors.get(kernel.first_tensor_input());
            let shape = first_input.shape().to_vec();
            let len = u32::try_from(first_input.num_elements()).expect("tensor too large");

            alloc_outputs(kernel, tensors, len, cx, stream, &shape)?;

            let mut params = build_params(tensors, vars, &kernel.params, len, None, bump);

            unsafe {
                func.launch_on_stream(stream.cudarc_stream(), launch_config(len), &mut params)?;
            }
        }
        Instr::ReductionKernel {
            kernel,
            reduction_depth,
            initial_reduced_value,
        } => {
            let func = cx
                .device()
                .get_func(&kernel.module_name, kernel.func_name)
                .expect("reduction module not found?");

            let first_input = tensors.get(kernel.first_tensor_input());
            let shape = first_input.shape().to_vec();
            let len = u32::try_from(first_input.num_elements()).expect("tensor too large");

            let output_shape = first_input.shape()
                [..first_input.shape().len() - *reduction_depth as usize]
                .to_vec();
            let output_len =
                u32::try_from(output_shape.iter().copied().product::<usize>()).unwrap();

            let reduction_stride = first_input.shape()
                [first_input.shape().len() - *reduction_depth as usize..]
                .iter()
                .copied()
                .product::<usize>() as u32;

            alloc_outputs(kernel, tensors, len, cx, stream, &shape)?;

            let mut reduction_output_tensor = RawTensor::new(
                unsafe {
                    Data::alloc_async(cx.device(), stream, DataType::F32, output_len as usize)?
                },
                output_shape,
            );
            reduction_output_tensor.fill(*initial_reduced_value, stream)?;

            tensors.insert(kernel.reduction_output.unwrap(), reduction_output_tensor);

            let mut params = build_params(
                tensors,
                vars,
                &kernel.params,
                len,
                Some(reduction_stride),
                bump,
            );

            unsafe {
                func.launch_on_stream(
                    stream.cudarc_stream(),
                    launch_config_for_reduction(len, reduction_stride),
                    &mut params,
                )?;
            }
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

            let out_rows = a_input.dim_at(-1) as u64;
            let out_cols = b_input.dim_at(-2) as u64;
            let out_len = out_rows * out_cols;

            let d_layout = cublaslt::create_matrix_layout(
                cuda_data_type(config.output_type),
                out_rows,
                out_cols,
                out_rows as i64,
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
                None => d_layout,
            };

            // TODO: batched matmul support
            assert_eq!(a_input.shape().len(), 2);
            assert_eq!(b_input.shape().len(), 2);

            let mut output_tensor = RawTensor::new(
                unsafe {
                    Data::alloc_async(cx.device(), stream, config.output_type, out_len as usize)?
                },
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
                    bias_input.map_or(ptr::null(), |t| t.data().device_ptr().cast()),
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
        Instr::UploadTensor(instr) => {
            let (data, shape) = vars.get(instr.data_var).expect_tensor();
            let tensor = match data {
                DataVec::F32(data) => RawTensor::from_slice_async::<f32>(
                    data,
                    instr.data_type,
                    shape,
                    stream,
                    cx.device(),
                )?,
                DataVec::Bf16(data) => RawTensor::from_slice_async::<bf16>(
                    data,
                    instr.data_type,
                    shape,
                    stream,
                    cx.device(),
                )?,
                DataVec::F16(data) => RawTensor::from_slice_async::<f16>(
                    data,
                    instr.data_type,
                    shape,
                    stream,
                    cx.device(),
                )?,
            };
            tensors.insert(instr.output, tensor);
        }
    }
    Ok(())
}

fn alloc_outputs(
    kernel: &LoadedKernel,
    tensors: &mut TensorMap,
    len: u32,
    cx: &CudaContext,
    stream: &CudaStream,
    shape: &[usize],
) -> Result<(), CudaError> {
    for (node, data_type) in &kernel.output_types {
        if kernel.reduction_output != Some(*node) {
            let tensor = RawTensor::new(
                unsafe { Data::alloc_async(cx.device(), stream, *data_type, len as usize)? },
                shape,
            );
            tensors.insert(*node, tensor);
        }
    }
    Ok(())
}

fn build_params(
    tensors: &mut TensorMap,
    vars: &VarMap,
    params: &[KernelParam],
    input_size: u32,
    reduction_stride: Option<u32>,
    bump: &Bump,
) -> Vec<*mut c_void> {
    let mut raw_params = Vec::new();

    for param in params {
        let raw = match param {
            KernelParam::Node(node) => {
                bump.alloc(tensors.get(*node).data().device_ptr()) as *mut *const u8 as *mut c_void
            }
            KernelParam::Var(id) => {
                let x = vars.get(*id).expect_scalar();
                bump.alloc(x) as *mut f32 as *mut c_void
            }
            KernelParam::Output(node) => bump
                .alloc(tensors.get_mut(*node).data_mut().device_ptr_mut())
                as *mut *mut u8 as *mut c_void,
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

fn launch_config_for_reduction(len: u32, stride: u32) -> LaunchConfig {
    let blocks_per_group = (stride + BLOCK_DIM - 1) / BLOCK_DIM;
    let groups = len / stride;
    let blocks = blocks_per_group * groups;
    LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (BLOCK_DIM, 1, 1),
        shared_mem_bytes: BLOCK_DIM * mem::size_of::<f32>() as u32,
    }
}
