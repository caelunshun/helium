use crate::opgraph::NodeId;

/// Compiled version of an `OpGraph` specifying what
/// sequence of CUDA kernels and operations to execute.
#[derive(Debug, Clone)]
pub struct Plan {
    steps: Vec<Step>,
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
    /// Free the memory owned by a node's output tensor.
    FreeTensor(NodeId),
}
