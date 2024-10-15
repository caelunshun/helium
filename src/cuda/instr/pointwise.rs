use crate::{
    backend::{InstrPerf, Instruction, TensorMap},
    cuda::{
        context::{CudaContext, CudaStream},
        Cuda,
    },
    opgraph::{
        op::{Op, ReduceOp},
        subgraph::OpSubgraph,
        Intermediate, Node, NodeId, OpGraph,
    },
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
#[derive(Debug, Clone)]
pub struct PointwiseGraph {
    subgraph: OpSubgraph,
}

impl PointwiseGraph {
    pub fn new(subgraph: OpSubgraph) -> Self {
        Self { subgraph }
    }

    pub fn execute(&self, tensors: &TensorMap<Cuda>, stream: &CudaStream, cx: &CudaContext) {
        // Reduction outputs need to be initialized to their
        // initial values (e.g. zero for sum, -inf for max).
        for output in self.subgraph.leafs() {
            if let Node::Intermediate(Intermediate {
                op: Op::Reduce(op), ..
            }) = self.subgraph.graph().get(output)
            {
                tensors
                    .get_storage(output)
                    .fill(initial_reduction_val(op.op), stream)
                    .expect("failed to fill reduction tensor with initial value");
            }
        }

        let kernel = jit::generate_kernel(&self.subgraph)
            .build("pointwise_kernel", cx)
            .expect("failed to compile kernel");
        let grid_size = jit::compute_grid_size(&self.subgraph);
        kernel
            .execute(
                |node| tensors.get_storage(node),
                stream,
                cx,
                grid_size as u32,
                jit::BLOCK_SIZE as u32,
            )
            .expect("failed to execute generated kernel");
    }
}

impl Instruction<Cuda> for PointwiseGraph {
    fn inputs(&self) -> Vec<NodeId> {
        self.subgraph.inputs().collect()
    }

    fn outputs(&self) -> Vec<NodeId> {
        self.subgraph.leafs().collect()
    }

    fn can_fuse_with(&self, _next: &Self, _op_graph: &Arc<OpGraph>) -> bool {
        todo!()
    }

    fn fuse_with(&self, _next: &Self, _op_graph: &Arc<OpGraph>) -> Self {
        todo!()
    }

    fn perf(&self) -> InstrPerf {
        InstrPerf::MemoryBound
    }
}

fn initial_reduction_val(op: ReduceOp) -> f32 {
    match op {
        ReduceOp::Sum | ReduceOp::Mean => 0.0,
        ReduceOp::Max => f32::NEG_INFINITY,
        ReduceOp::Min => f32::INFINITY,
    }
}
