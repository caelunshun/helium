use crate::{
    backend::{InstrPerf, Instruction, TensorMap},
    cuda::{
        context::{CudaContext, CudaStream},
        Cuda,
    },
    opgraph::{subgraph::OpSubgraph, NodeId, OpGraph},
};
use std::sync::Arc;

mod jit;

/// Instruction that handles graphs of elementwise operations
/// by JIT-compiling specialized kernels.
///
/// This instruction handles op subgraphs that meet the following requirements:
/// 1. Consist only of pointwise, reduction, and restructuring ops (e.g., swap dim,
/// broadcast, reshape).
/// 2. All reduction ops must be leaves.
/// 3. All reductions must have the same size.
/// 4. All outputs must have the same size.
pub struct PointwiseGraph {
    subgraph: OpSubgraph,
}

impl PointwiseGraph {
    pub fn new(subgraph: OpSubgraph) -> Self {
        Self { subgraph }
    }

    pub fn execute(&self, tensors: &TensorMap<Cuda>, stream: &CudaStream, cx: &CudaContext) {}
}

impl Instruction<Cuda> for PointwiseGraph {
    fn inputs(&self) -> Vec<NodeId> {
        self.subgraph.inputs().collect()
    }

    fn outputs(&self) -> Vec<NodeId> {
        self.subgraph.leafs().collect()
    }

    fn can_fuse_with(&self, next: &Self, op_graph: &Arc<OpGraph>) -> bool {
        todo!()
    }

    fn fuse_with(&self, next: &Self, op_graph: &Arc<OpGraph>) -> Self {
        todo!()
    }

    fn perf(&self) -> InstrPerf {
        todo!()
    }
}
