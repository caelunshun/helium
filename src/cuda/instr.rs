use crate::{
    backend::{InstrPerf, Instruction, TensorMap},
    cuda::{
        allocator::Memory,
        context::{CudaContext, CudaStream},
        instr::cudnn_graph::CudnnGraph,
        Cuda,
    },
    opgraph::{NodeId, OpGraph},
};
use pointwise::PointwiseGraph;
use std::sync::Arc;

pub mod cudnn_graph;
pub mod pointwise;

#[derive(Debug, Clone)]
pub enum Instr {
    CudnnGraph(CudnnGraph),
    PointwiseGraph(PointwiseGraph),
}

impl Instr {
    pub fn execute(
        &self,
        tensors: &TensorMap<Cuda>,
        stream: &CudaStream,
        cx: &CudaContext,
        hold_allocations: &mut Vec<Memory>,
    ) {
        match self {
            Instr::CudnnGraph(instr) => instr.execute(tensors, stream, cx, hold_allocations),
            Instr::PointwiseGraph(instr) => instr.execute(tensors, stream, cx),
        }
    }
}

impl Instruction<Cuda> for Instr {
    fn inputs(&self) -> Vec<NodeId> {
        match self {
            Instr::CudnnGraph(instr) => instr.inputs(),
            Instr::PointwiseGraph(instr) => instr.inputs(),
        }
    }

    fn outputs(&self) -> Vec<NodeId> {
        match self {
            Instr::CudnnGraph(instr) => instr.outputs(),
            Instr::PointwiseGraph(instr) => instr.outputs(),
        }
    }

    fn can_fuse_with(&self, next: &Self, op_graph: &Arc<OpGraph>) -> bool {
        match (self, next) {
            (Instr::CudnnGraph(instr1), Instr::CudnnGraph(instr2)) => {
                instr1.can_fuse_with(instr2, op_graph)
            }
            (Instr::PointwiseGraph(instr1), Instr::PointwiseGraph(instr2)) => {
                instr1.can_fuse_with(instr2, op_graph)
            }
            _ => false,
        }
    }

    fn fuse_with(&self, next: &Self, op_graph: &Arc<OpGraph>) -> Self {
        match (self, next) {
            (Instr::CudnnGraph(instr1), Instr::CudnnGraph(instr2)) => {
                Instr::CudnnGraph(instr1.fuse_with(instr2, op_graph))
            }
            (Instr::PointwiseGraph(instr1), Instr::PointwiseGraph(instr2)) => {
                Instr::PointwiseGraph(instr1.fuse_with(instr2, op_graph))
            }
            _ => unreachable!("can_fuse_with() is false"),
        }
    }

    fn perf(&self) -> InstrPerf {
        match self {
            Instr::CudnnGraph(instr) => instr.perf(),
            Instr::PointwiseGraph(instr) => instr.perf(),
        }
    }
}
