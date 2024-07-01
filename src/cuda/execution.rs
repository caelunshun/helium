use crate::{
    cuda::{
        context::{CudaContext, CudaStream},
        error::CudaError,
        kernel::KernelParam,
        plan::{Instr, Plan},
        tensor::{Data, RawTensor},
    },
    opgraph::NodeId,
};
use bumpalo::Bump;
use cudarc::{
    driver,
    driver::{
        sys::{CUevent_flags, CUevent_wait_flags},
        DriverError, LaunchAsync, LaunchConfig,
    },
};
use slotmap::SecondaryMap;
use std::{ffi::c_void, mem};

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
        }

        for (stream, event) in streams.iter().zip(&events) {
            unsafe {
                driver::result::event::record(*event, stream.raw())?;
            }
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
) -> Result<(), CudaError> {
    match instr {
        Instr::FreeTensor(node_id) => {
            tensors.remove(*node_id);
        }
        Instr::PointwiseKernel { kernel } => {
            let func = cx
                .device()
                .get_func(&kernel.module_name, kernel.func_name)
                .expect("pointwise module not found?");

            let first_input = tensors.get(kernel.first_tensor_input());
            let len = u32::try_from(first_input.num_elements()).expect("tensor too large");

            let mut output = RawTensor::new(
                unsafe { Data::alloc(cx.device(), kernel.output_type, len as usize)? },
                first_input.shape().to_vec(),
            );

            let mut params = build_params(tensors, &kernel.params, len, None, &mut output, bump);

            unsafe {
                func.launch_on_stream(stream.cudarc_stream(), launch_config(len), &mut params)?;
            }
        }
        Instr::ReductionKernel {
            kernel,
            reduction_depth,
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

            let mut output = RawTensor::new(
                unsafe { Data::alloc(cx.device(), kernel.output_type, output_len as usize)? },
                output_shape,
            );

            let mut params = build_params(
                tensors,
                &kernel.params,
                len,
                Some(reduction_stride),
                &mut output,
                bump,
            );

            unsafe {
                func.launch_on_stream(
                    stream.cudarc_stream(),
                    launch_config_for_reduction(len),
                    &mut params,
                )?;
            }
        }
        Instr::Matmul(config) => {
            todo!()
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
