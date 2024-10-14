use crate::{
    backend::{InstrPerf, Instruction, TensorMap},
    cuda::{
        allocator::Memory,
        context::{CudaContext, CudaStream},
        instr::cudnn_graph::CudnnGraph,
        Cuda,
    },
    data_type::DataVec,
    opgraph::{NodeId, OpGraph},
};
use std::sync::Arc;

pub mod cudnn_graph;
pub mod pointwise;

#[derive(Debug, Clone)]
pub enum Instr {
    UploadTensor { node: NodeId, data: DataVec },
    CopyTensor { from: NodeId, to: NodeId },
    CudnnGraph(CudnnGraph),
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
            Instr::UploadTensor { node, data } => {
                tensors
                    .get_storage(*node)
                    .initialize_with_data(data, stream)
                    .expect("failed to copy data");
            }
            Instr::CopyTensor { from, to } => {
                tensors
                    .get_storage(*to)
                    .copy_from(tensors.get_storage(*from), stream)
                    .expect("failed to copy data");
            }
        }
    }
}

impl Instruction<Cuda> for Instr {
    fn inputs(&self) -> Vec<NodeId> {
        match self {
            Instr::UploadTensor { .. } => vec![],
            Instr::CudnnGraph(instr) => instr.inputs(),
            Instr::CopyTensor { from, .. } => vec![*from],
        }
    }

    fn outputs(&self) -> Vec<NodeId> {
        match self {
            Instr::UploadTensor { node, .. } => vec![*node],
            Instr::CudnnGraph(instr) => instr.outputs(),
            Instr::CopyTensor { to, .. } => vec![*to],
        }
    }

    fn can_fuse_with(&self, next: &Self, op_graph: &Arc<OpGraph>) -> bool {
        match (self, next) {
            (Instr::CudnnGraph(instr1), Instr::CudnnGraph(instr2)) => {
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
            _ => unreachable!("can_fuse_with() is false"),
        }
    }

    fn perf(&self) -> InstrPerf {
        match self {
            Instr::CudnnGraph(instr) => instr.perf(),
            Instr::UploadTensor { .. } | Instr::CopyTensor { .. } => InstrPerf::MemoryBound,
        }
    }
}
