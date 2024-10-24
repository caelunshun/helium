use crate::{
    backend::{InstrPerf, Instruction, TensorMap},
    cuda::{
        context::{CudaContext, CudaStream},
        kernel_jit::JitKernel,
        Cuda,
    },
    opgraph::{op::Op, subgraph::OpSubgraph, Intermediate, Node, NodeId, OpGraph},
    shape::Shape,
};
use ahash::AHashMap;
use parking_lot::Mutex;
use std::sync::{Arc, OnceLock};

mod jit;

/// Specialized tiled kernel JIT
/// that can perform a sequence of `swap_dim`
/// operations. This kernel is much more efficient
/// than generated `pointwise` kernels
/// when the last dimension of the input tensor is involved
/// in the swap, since cuDNN does some magic with shared
/// memory so that shared memory accesses remain coalesced.
#[derive(Debug, Clone)]
pub struct PermuteDims {
    subgraph: OpSubgraph,
}

impl PermuteDims {
    pub fn new(subgraph: OpSubgraph) -> Self {
        Self { subgraph }
    }

    pub fn execute(&self, tensors: &TensorMap<Cuda>, stream: &CudaStream, cx: &CudaContext) {
        let kernel = self.get_cached_kernel(cx);
        let in_descriptor = self
            .subgraph
            .graph()
            .get(self.subgraph.inputs().next().unwrap())
            .descriptor();
        let shape = &in_descriptor.shape;
        let strides = compute_output_strides(&enumerate_swaps(&self.subgraph).unwrap(), shape);
        let grid_size = jit::compute_grid_size(shape, &strides);

        kernel
            .execute2d(
                |id| tensors.get_storage(id),
                stream,
                cx,
                grid_size.map(|x| x as u32),
                jit::BLOCK_SIZE.map(|x| x as u32),
            )
            .expect("failed to execute kernel");
    }

    fn get_cached_kernel(&self, cx: &CudaContext) -> Arc<JitKernel> {
        static CACHE: OnceLock<Mutex<AHashMap<OpSubgraph, Arc<JitKernel>>>> = OnceLock::new();
        CACHE
            .get_or_init(Default::default)
            .lock()
            .entry(self.subgraph.clone())
            .or_insert_with(|| Arc::new(self.build_kernel(cx)))
            .clone()
    }

    fn build_kernel(&self, cx: &CudaContext) -> JitKernel {
        let in_descriptor = self
            .subgraph
            .graph()
            .get(self.subgraph.inputs().next().unwrap())
            .descriptor();
        let shape = &in_descriptor.shape;
        let strides = compute_output_strides(&enumerate_swaps(&self.subgraph).unwrap(), shape);
        let kernel = jit::generate_kernel(
            shape,
            &strides,
            self.subgraph.inputs().next().unwrap(),
            self.subgraph.leafs().next().unwrap(),
            in_descriptor.data_type,
        );
        kernel
            .build("permute_dims", cx)
            .expect("failed to compile kernel")
    }
}

impl Instruction<Cuda> for PermuteDims {
    fn inputs(&self) -> Vec<NodeId> {
        self.subgraph.inputs().collect()
    }

    fn outputs(&self) -> Vec<NodeId> {
        self.subgraph.leafs().collect()
    }

    fn can_fuse_with(&self, next: &Self, _op_graph: &Arc<OpGraph>) -> bool {
        let new_graph = self.subgraph.merge_with(&next.subgraph);
        enumerate_swaps(&new_graph).is_some()
    }

    fn fuse_with(&self, next: &Self, _op_graph: &Arc<OpGraph>) -> Self {
        let new_graph = self.subgraph.merge_with(&next.subgraph);
        Self {
            subgraph: new_graph,
        }
    }

    fn perf(&self) -> InstrPerf {
        InstrPerf::MemoryBound
    }
}

fn enumerate_swaps(subgraph: &OpSubgraph) -> Option<Vec<(usize, usize)>> {
    let mut swaps = Vec::new();
    if subgraph.inputs().count() > 1 {
        return None;
    }

    let mut current = entry_node(subgraph);
    let mut visited = vec![current];
    loop {
        let Node::Intermediate(Intermediate {
            op: Op::SwapDims(op),
            ..
        }) = subgraph.graph().get(current)
        else {
            // Unsupported op
            return None;
        };
        swaps.push((op.axis_a, op.axis_b));
        let next = subgraph.graph().outbound_edges(current);
        if next.len() == 1 && subgraph.contains_node(next[0]) {
            current = next[0];
            visited.push(current);
        } else {
            break;
        }
    }
    if subgraph.nodes().any(|n| !visited.contains(&n)) {
        return None;
    }
    Some(swaps)
}

fn entry_node(subgraph: &OpSubgraph) -> NodeId {
    let input = subgraph.inputs().next().unwrap();
    subgraph
        .graph()
        .outbound_edges(input)
        .iter()
        .copied()
        .find(|id| subgraph.contains_node(*id))
        .unwrap()
}

/// Given a sequence of swap operations, computes
/// output strides such that memory layout is transformed
/// to match the swaps.
fn compute_output_strides(swaps: &[(usize, usize)], in_shape: &Shape) -> Vec<usize> {
    let mut permutation = (0..in_shape.num_dims()).collect::<Vec<_>>();
    for (a, b) in swaps {
        permutation.swap(*a, *b);
    }

    let mut strides = vec![1usize; in_shape.num_dims()];
    let mut stride = 1;
    for i in (0..in_shape.num_dims()).rev() {
        let j = permutation.iter().copied().position(|x| x == i).unwrap();
        strides[j] = stride;
        stride *= in_shape.dims()[j];
    }
    strides
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_output_strides() {
        assert_eq!(
            compute_output_strides(&[(0, 1)], &Shape::new([100, 1000])),
            vec![1, 100],
        );
        assert_eq!(
            compute_output_strides(&[(3, 2), (2, 1)], &Shape::new([32, 64, 16, 4])),
            vec![4096, 1, 256, 64],
        );
    }
}
