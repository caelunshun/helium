use super::allocator::StreamId;
use crate::{
    backend::{InstrPerf, Instruction, TensorMap},
    cuda::{
        Cuda,
        allocator::DeviceMemory,
        context::{CudaContext, CudaStream},
        instr::{cudnn_graph::CudnnGraph, permute_dims::PermuteDims},
    },
    opgraph::{NodeId, OpGraph},
};
use pointwise::PointwiseGraph;
use std::sync::Arc;

pub mod cudnn_graph;
pub mod permute_dims;
pub mod pointwise;

#[derive(Debug, Clone)]
pub enum Instr {
    CudnnGraph(CudnnGraph),
    PointwiseGraph(PointwiseGraph),
    PermuteDims(PermuteDims),
}

impl Instr {
    pub fn execute(
        &self,
        tensors: &TensorMap<Cuda>,
        stream: &CudaStream,
        cx: &CudaContext,
        hold_allocations: &mut Vec<DeviceMemory>,
        allocation_stream: StreamId,
    ) {
        match self {
            Instr::CudnnGraph(instr) => {
                instr.execute(tensors, stream, cx, hold_allocations, allocation_stream)
            }
            Instr::PointwiseGraph(instr) => instr.execute(tensors, stream, cx),
            Instr::PermuteDims(instr) => instr.execute(tensors, stream, cx),
        }
    }

    #[profiling::function]
    pub fn precompile(&self, cx: &CudaContext) {
        match self {
            Instr::CudnnGraph(instr) => instr.precompile(cx),
            Instr::PointwiseGraph(instr) => instr.precompile(cx),
            Instr::PermuteDims(instr) => instr.precompile(cx),
        }
    }
}

impl Instruction<Cuda> for Instr {
    fn inputs(&self) -> Vec<NodeId> {
        match self {
            Instr::CudnnGraph(instr) => instr.inputs(),
            Instr::PointwiseGraph(instr) => instr.inputs(),
            Instr::PermuteDims(instr) => instr.inputs(),
        }
    }

    fn outputs(&self) -> Vec<NodeId> {
        match self {
            Instr::CudnnGraph(instr) => instr.outputs(),
            Instr::PointwiseGraph(instr) => instr.outputs(),
            Instr::PermuteDims(instr) => instr.outputs(),
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
            (Instr::PermuteDims(instr1), Instr::PermuteDims(instr2)) => {
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
            (Instr::PermuteDims(instr1), Instr::PermuteDims(instr2)) => {
                Instr::PermuteDims(instr1.fuse_with(instr2, op_graph))
            }
            _ => unreachable!("can_fuse_with() is false"),
        }
    }

    fn perf(&self) -> InstrPerf {
        match self {
            Instr::CudnnGraph(instr) => instr.perf(),
            Instr::PointwiseGraph(instr) => instr.perf(),
            Instr::PermuteDims(instr) => instr.perf(),
        }
    }
}
