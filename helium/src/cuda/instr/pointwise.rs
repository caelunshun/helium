use crate::{
    DataType,
    backend::{InstrPerf, Instruction, TensorMap},
    cache::Cache,
    cuda::{
        Cuda,
        context::{CudaContext, CudaStream},
        kernel_jit::JitKernel,
    },
    opgraph::{
        Intermediate, Node, NodeId, OpGraph,
        op::{Op, ReduceOp},
        subgraph::OpSubgraph,
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

    #[profiling::function]
    pub fn precompile(&self, cx: &CudaContext) {
        self.get_kernel(cx);
    }

    fn get_kernel(&self, cx: &CudaContext) -> Arc<JitKernel> {
        static KERNEL_CACHE: Cache<OpSubgraph, Arc<JitKernel>> = Cache::with_capacity(1024);

        KERNEL_CACHE.get_or_insert(&self.subgraph, || {
            Arc::new(
                jit::generate_kernel(&self.subgraph)
                    .build("pointwise_kernel", cx)
                    .expect("failed to compile kernel"),
            )
        })
    }

    #[profiling::function]
    pub fn execute(&self, tensors: &TensorMap<Cuda>, stream: &CudaStream, cx: &CudaContext) {
        // Reduction outputs need to be initialized to their
        // initial values (e.g. zero for sum, -inf for max).
        // Boolean (bitset) outputs need to be initialized
        // because of write access granularity of 1 bit.
        for output in self.subgraph.leafs() {
            if let Node::Intermediate(Intermediate {
                op: Op::Reduce(op), ..
            }) = self.subgraph.graph().get(output)
            {
                tensors
                    .get_storage(output)
                    .fill(initial_reduction_val(op.op), stream)
                    .expect("failed to fill reduction tensor with initial value");
            } else if self.subgraph.graph().get(output).descriptor().data_type == DataType::Bool {
                tensors.get_storage(output).fill(0.0, stream).unwrap();
            }
        }

        let grid_size = jit::compute_grid_size(&self.subgraph);
        self.get_kernel(cx)
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

    fn can_fuse_with(&self, next: &Self, _op_graph: &Arc<OpGraph>) -> bool {
        let graph = self.subgraph.merge_with(&next.subgraph);
        supports_subgraph(&graph)
    }

    fn fuse_with(&self, next: &Self, _op_graph: &Arc<OpGraph>) -> Self {
        let subgraph = self.subgraph.merge_with(&next.subgraph);
        Self { subgraph }
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

fn supports_subgraph(subgraph: &OpSubgraph) -> bool {
    let mut output_size = None;
    let mut reduction_shape = None;
    let mut x = 0;

    for node in subgraph.nodes() {
        let Node::Intermediate(Intermediate { op, .. }) = subgraph.graph().get(node) else {
            unreachable!("node must be intermediate")
        };

        let check_output_size_node = match op {
            Op::Reduce(op) => op.input,
            _ => node,
        };

        if subgraph.leafs().any(|l| l == node) {
            let descriptor = subgraph.graph().get(check_output_size_node).descriptor();
            match output_size {
                None => output_size = Some(descriptor.shape.num_elements()),
                Some(output_size) => {
                    if output_size != descriptor.shape.num_elements() {
                        // All output sizes must match
                        return false;
                    }
                }
            }
        }

        match op {
            Op::Select(_)
            | Op::Reshape(_)
            | Op::Compare(_)
            | Op::UnaryPointwise(_)
            | Op::BinaryPointwise(_)
            | Op::ChangeDataType(_) => continue,
            Op::Broadcast(_) => x += 1,
            Op::SwapDims(_) => {
                x += 1;
                if x > 1 {
                    return false;
                }
            }
            Op::Reduce(op) => {
                if !subgraph.leafs().any(|l| l == node) {
                    // Reduction must be output
                    return false;
                }

                let input_shape = &subgraph.graph().get(op.input).descriptor().shape;

                match &reduction_shape {
                    None => reduction_shape = Some((input_shape.clone(), op.depth)),
                    Some((shape, depth)) => {
                        if shape != input_shape || op.depth != *depth {
                            // All reduction shapes + depths must be equal
                            return false;
                        }
                    }
                }
            }
            _ => return false, // unsupported op
        }
    }

    true
}
