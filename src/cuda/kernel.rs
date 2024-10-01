//! Runtime generation of fused kernels.

use crate::{
    cuda::error::CudaError,
    data_type::DataType,
    opgraph::{NodeId, VarId},
};
use ahash::AHashMap;
use cudarc::{
    driver::{result::device, sys::CUdevice_attribute, CudaDevice},
    nvrtc::{compile_ptx_with_opts, CompileOptions, Ptx},
};
use std::cell::Cell;

pub mod pointwise;
pub mod reduction;
pub mod restructure;

#[derive(Debug, Clone)]
pub struct Kernel {
    /// CUDA C++ code
    pub code: String,
    /// List of inputs to pass to kernel.
    params: Vec<KernelParam>,
    entrypoint_name: &'static str,
}

pub struct CompiledKernel {
    ptx: Ptx,
    entrypoint_name: &'static str,
    params: Vec<KernelParam>,
}

impl CompiledKernel {
    pub fn new(kernel: &Kernel, device: &CudaDevice) -> Result<Self, CudaError> {
        let compute_major = unsafe {
            device::get_attribute(
                *device.cu_device(),
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            )?
        };
        let compute_minor = unsafe {
            device::get_attribute(
                *device.cu_device(),
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            )?
        };
        let device_arch = format!("compute_{compute_major}{compute_minor}");

        let ptx = compile_ptx_with_opts(
            &kernel.code,
            CompileOptions {
                arch: Some(device_arch.leak()), // TODO remove leak
                include_paths: vec!["/usr/local/cuda/include".into()], // TODO make cross-platform
                ..Default::default()
            },
        )?;
        Ok(Self {
            ptx,
            entrypoint_name: kernel.entrypoint_name,
            params: kernel.params.clone(),
        })
    }

    pub fn ptx(&self) -> &Ptx {
        &self.ptx
    }

    pub fn entrypoint_name(&self) -> &'static str {
        self.entrypoint_name
    }

    pub fn params(&self) -> impl Iterator<Item = KernelParam> + '_ {
        self.params.iter().copied()
    }
}

#[derive(Debug, Copy, Clone)]
pub enum KernelParam {
    /// Tensor data pointer from a previous node's output
    Node(NodeId),
    /// Scalar float variable
    Var(VarId),
    /// An output tensor
    Output(NodeId),
    /// Number of elements to operate on; used for pointwise
    /// and reduction
    Size,
    /// For reductions, specifies the stride between reduction
    /// groups
    ReductionStride,
}

/// Context for building kernel source code.
#[derive(Default)]
pub struct Context {
    /// Maps variables in the opgraph (i.e. dynamic scalar inputs)
    /// to the C++ variable name storing their values.
    var_map: AHashMap<VarId, String>,
    /// Maps input tensor arrays to their C++ identifiers.
    input_map: AHashMap<NodeId, String>,
    /// Maps intermediate outputs of nodes to their C++ identifiers,
    /// if they are present. This is used for fused operations.
    intermediate_map: AHashMap<NodeId, String>,
    next_identifier: Cell<u64>,
}

impl Context {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert_var(&mut self, var: VarId) -> String {
        let id = self.generate_identifier();
        self.var_map.insert(var, id.clone());
        id
    }

    pub fn insert_input(&mut self, input: NodeId) -> String {
        let id = self.generate_identifier();
        self.input_map.insert(input, id.clone());
        id
    }

    pub fn insert_intermediate(&mut self, intermediate: NodeId) -> String {
        let id = self.generate_identifier();
        self.intermediate_map.insert(intermediate, id.clone());
        id
    }

    pub fn var(&self, id: VarId) -> &str {
        &self.var_map[&id]
    }

    #[expect(unused)]
    pub fn input(&self, id: NodeId) -> &str {
        &self.input_map[&id]
    }

    pub fn intermediate(&self, id: NodeId) -> &str {
        &self.intermediate_map[&id]
    }

    /// Generates a new unique identifier
    /// for use in variable declarations, function
    /// names, etc.
    pub fn generate_identifier(&self) -> String {
        let x = self.next_identifier.get();
        self.next_identifier.set(x + 1);
        format!("ident{x}")
    }
}

fn cpp_type_name(datatype: DataType) -> &'static str {
    match datatype {
        DataType::F32 => "float",
        DataType::Bf16 => "nv_bfloat16",
        DataType::F16 => "half",
    }
}
