use crate::{
    cuda::{context::LoadedKernel, kernel::KernelParam},
    data_type::DataType,
    opgraph::{NodeId, VarId},
};
use cudarc::cublaslt::sys::cublasLtEpilogue_t;
use std::sync::Arc;

/// Compiled version of an `OpGraph` specifying what
/// sequence of CUDA kernels and operations to execute.
#[derive(Debug, Clone, Default)]
pub struct Plan {
    steps: Vec<Step>,
}

impl Plan {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push_step(&mut self, step: Step) {
        self.steps.push(step);
    }

    pub fn steps(&self) -> impl Iterator<Item = &Step> + '_ {
        self.steps.iter()
    }

    pub fn max_parallelism(&self) -> usize {
        self.steps.iter().map(|s| s.0.len()).max().unwrap()
    }
}

/// On each step, one or more `Instr` execute, potentially
/// in parallel.
#[derive(Debug, Clone)]
pub struct Step(Vec<Instr>);

impl Step {
    pub fn new(instr: impl IntoIterator<Item = Instr>) -> Self {
        Self(instr.into_iter().collect())
    }

    pub fn push(&mut self, instr: Instr) {
        self.0.push(instr);
    }

    pub fn instrs(&self) -> impl Iterator<Item = &Instr> + '_ {
        self.0.iter()
    }
}

/// CUDA kernel or operation.
#[derive(Debug, Clone)]
pub enum Instr {
    UploadTensor(UploadTensorInstr),
    /// Free the device memory owned by a node's output tensor.
    FreeTensor(NodeId),
    /// Execute a generated kernel.
    PointwiseKernel {
        kernel: Arc<LoadedKernel>,
    },
    ReductionKernel {
        kernel: Arc<LoadedKernel>,
        reduction_depth: u32,
    },
    /// Execute a matmul with cublasLT.
    Matmul(MatmulInstr),
}

impl Instr {
    pub fn dependencies(&self) -> Vec<NodeId> {
        match self {
            Instr::FreeTensor(id) => vec![*id],
            Instr::PointwiseKernel { kernel, .. } | Instr::ReductionKernel { kernel, .. } => {
                let mut deps = Vec::new();
                for param in &kernel.params {
                    if let KernelParam::Node(id) = param {
                        deps.push(*id);
                    }
                }
                deps
            }
            Instr::Matmul(config) => {
                let mut deps = vec![config.a_input, config.b_input];
                if let Some(bias) = config.bias_input {
                    deps.push(bias);
                }
                deps
            }
            Instr::UploadTensor(_) => vec![],
        }
    }

    pub fn outputs(&self) -> Vec<NodeId> {
        match self {
            Instr::FreeTensor(_) => Vec::new(),
            Instr::PointwiseKernel { kernel, .. } | Instr::ReductionKernel { kernel, .. } => {
                kernel.output_types.keys().copied().collect()
            }
            Instr::Matmul(config) => vec![config.output],
            Instr::UploadTensor(instr) => vec![instr.output],
        }
    }
}

#[derive(Debug, Clone)]
pub struct UploadTensorInstr {
    pub data_var: VarId,
    pub data_type: DataType,
    pub output: NodeId,
}

#[derive(Debug, Clone)]
pub struct MatmulInstr {
    pub a_input: NodeId,
    pub b_input: NodeId,
    pub output_type: DataType,
    /// Optionally feed the matrices as transposed
    pub transpose_a: bool,
    pub transpose_b: bool,
    /// Optional fused addition of bias vector
    /// (broadcast)
    pub bias_input: Option<NodeId>,
    /// Optional fused ReLU or GeLU on the output
    pub epilogue: cublasLtEpilogue_t,
    pub output: NodeId,
}
