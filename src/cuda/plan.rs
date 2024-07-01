use crate::{cuda::context::LoadedKernel, opgraph::NodeId};
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
}

/// On each step, one or more `Instr` execute, potentially
/// in parallel.
#[derive(Debug, Clone)]
pub struct Step(Vec<Instr>);

impl Step {
    pub fn new(instr: impl IntoIterator<Item = Instr>) -> Self {
        Self(instr.into_iter().collect())
    }

    pub fn instrs(&self) -> impl Iterator<Item = &Instr> + '_ {
        self.0.iter()
    }
}

/// CUDA kernel or operation.
#[derive(Debug, Clone)]
pub enum Instr {
    /// Free the device memory owned by a node's output tensor.
    FreeTensor(NodeId),
    /// Execute a generated kernel.
    PointwiseKernel(Arc<LoadedKernel>),
    ReductionKernel {
        kernel: Arc<LoadedKernel>,
        reduction_depth: u32,
    },
    /// Execute a matmul with cublasLT.
    Matmul(MatmulInstr),
}

#[derive(Debug, Clone)]
pub struct MatmulInstr {
    /// Optionally feed the matrices as transposed
    pub transpose_a: bool,
    pub transpose_b: bool,
    /// Optional fused addition of bias vector
    /// (broadcast)
    pub bias_input: Option<NodeId>,
    /// Optional fused ReLU or GeLU on the output
    pub epilogue: cublasLtEpilogue_t,
}
