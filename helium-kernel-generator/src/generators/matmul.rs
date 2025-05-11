//! Generator for fast fused matrix multiplication kernels.
//!
//! These kernels compute `C = f(A)g(B)` in a tiled fashion and then write to memory
//! one or more tensors derived from the tiles of `C` (epilogue fusions). `f` and `g` (mainloop fusions)
//! can be shape operations, unary pointwise operations, binary pointwise operations with constant
//! operands, or compositions thereof. Epilogue fusions can be shape operations except broadcast, unary or binary
//! pointwise operations (supporting additional tensor inputs), reductions (yet to be implemented),
//! or compositions thereof. Reductions must be leaf nodes, i.e. no further operations can be applied
//! to the output of a reduction in a fused kernel.
//!
//! These kernels do away with the notion of "generalized" matrix-multiply that accepts
//! scalar alpha and beta values and an accumulator matrix. Those extra operations are more
//! elegantly expressed as optional mainloop (for alpha/beta) and epilogue (for accumulator matrix) fusions.

use crate::cute::Layout;
use ahash::{AHashMap, AHashSet};
use helium_ir::{
    opgraph::{Intermediate, Node, NodeId, OpGraph, op::Op, subgraph::OpSubgraph},
    shape::Shape,
};

mod emit;

/// Generator for a valid matmul fusion.
#[derive(Debug)]
pub struct MatmulGenerator {
    op_subgraph: OpSubgraph,
    input_a_root: NodeId,
    input_b_root: NodeId,
    matmul_node: NodeId,
    leafs: Vec<Leaf>,
}

/// An output tensor written to memory.
#[derive(Debug)]
struct Leaf {
    /// Output node in the `OpGraph` that corresponds to the leaf.
    node: NodeId,
    /// Layout object representing the mapping from coordinates in
    /// the logical output space (i.e. `M` by `N` C matrix) to the
    /// index in the output tensor.
    ///
    /// For example, this layout can encode a transpose operation
    /// by using column-major instead of row-major strides.
    /// More complex operations are of course possible.
    output_mapping: Layout,
}

#[derive(Debug)]
pub struct UnsupportedFusion;

impl MatmulGenerator {
    /// Attempts to construct a kernel generator for the given
    /// operation subgraph. Returns `Err(UnsupportedFusion)` if
    /// the operation graph violates the rules in the module-level documentation.
    pub fn new(op_subgraph: &OpSubgraph) -> Result<Self, UnsupportedFusion> {
        let matmul_node = find_matmul(op_subgraph).ok_or(UnsupportedFusion)?;
        let Node::Intermediate(Intermediate {
            op: Op::Matmul(matmul),
            ..
        }) = op_subgraph.graph().get(matmul_node)
        else {
            unreachable!()
        };

        let input_a_root = find_root(matmul.input_a, op_subgraph).ok_or(UnsupportedFusion)?;
        let input_b_root = find_root(matmul.input_b, op_subgraph).ok_or(UnsupportedFusion)?;

        let leafs = find_and_verify_leafs(op_subgraph, matmul_node).ok_or(UnsupportedFusion)?;

        Ok(Self {
            op_subgraph: op_subgraph.clone(),
            input_a_root,
            input_b_root,
            matmul_node,
            leafs,
        })
    }
}

fn find_matmul(op_subgraph: &OpSubgraph) -> Option<NodeId> {
    op_subgraph.nodes().find(|id| {
        matches!(
            op_subgraph.get(*id),
            Node::Intermediate(Intermediate {
                op: Op::Matmul(_),
                ..
            })
        )
    })
}

fn find_root(node: NodeId, op_subgraph: &OpSubgraph) -> Option<NodeId> {
    let mut stack = vec![node];
    stack.sort_unstable();
    stack.dedup();
    let mut roots = Vec::new();
    let mut visited: AHashSet<_> = stack.iter().copied().collect();
    while let Some(current) = stack.pop() {
        if op_subgraph.is_input(current) {
            roots.push(current);
        } else {
            let Node::Intermediate(Intermediate { op, .. }) = op_subgraph.get(current) else {
                unreachable!()
            };
            if !is_supported_mainloop_fusion_op(op) {
                return None;
            }
            for prev in op_subgraph.inbound_edges(current) {
                if visited.insert(prev) {
                    stack.push(prev);
                }
            }
        }
    }

    if roots.len() > 1 {
        // currently no auxiliary mainloop inputs allowed
        None
    } else {
        roots.first().copied()
    }
}

fn find_and_verify_leafs(op_subgraph: &OpSubgraph, matmul_node: NodeId) -> Option<Vec<Leaf>> {
    let mut leafs: Vec<Leaf> = Vec::new();

    let matmul_output_shape = &op_subgraph.get(matmul_node).descriptor().shape;

    let mut visited = AHashSet::new();

    let mut stack = vec![matmul_node];
    let mut output_mappings = AHashMap::new();
    output_mappings.insert(matmul_node, Layout::from_tensor_shape(matmul_output_shape));

    visited.extend(stack.iter().copied());

    while let Some(node) = stack.pop() {
        if op_subgraph.is_leaf(node) {
            leafs.push(Leaf {
                node,
                output_mapping: output_mappings.get(&node).unwrap().clone(),
            });
        }

        for next in op_subgraph.outbound_edges(node) {
            if op_subgraph
                .inbound_edges(next)
                .all(|dep| visited.contains(&dep))
                && visited.insert(next)
            {
                let Node::Intermediate(Intermediate { op, .. }) = op_subgraph.get(next) else {
                    unreachable!()
                };

                if !is_supported_epilogue_fusion_op(op) {
                    return None;
                }

                let new_output_mapping = match op {
                    Op::Reshape(_) | Op::SwapDims(_) => todo!(),
                    _ => output_mappings.get(&node).unwrap().clone(),
                };

                stack.push(next);
                output_mappings.insert(next, new_output_mapping);
            }
        }
    }

    if leafs.len() != op_subgraph.leafs().count() {
        return None;
    }

    Some(leafs)
}

fn is_supported_epilogue_fusion_op(op: &Op) -> bool {
    matches!(
        op,
        Op::Constant(_)
            | Op::UnaryPointwise(_)
            | Op::BinaryPointwise(_)
            | Op::Compare(_)
            | Op::Select(_)
            | Op::ChangeDataType(_)
            | Op::Reshape(_)
            | Op::SwapDims(_)
    )
}

fn is_supported_mainloop_fusion_op(op: &Op) -> bool {
    matches!(
        op,
        Op::Constant(_)
            | Op::UnaryPointwise(_)
            | Op::BinaryPointwise(_)
            | Op::ChangeDataType(_)
            | Op::Reshape(_)
            | Op::SwapDims(_)
    )
}

fn is_shape_op(op: &Op) -> bool {
    matches!(op, Op::SwapDims(_) | Op::Reshape(_))
}

#[cfg(test)]
mod tests {
    use super::*;
    use helium_ir::{
        data_type::DataType,
        opgraph::{
            Descriptor,
            op::{Matmul, UnaryPointwise, UnaryPointwiseOp, precision::Precision},
        },
    };
    use std::sync::Arc;

    #[test]
    fn trivial_kernel_is_valid() {
        let mut graph = OpGraph::new();
        let input = graph.new_input(Descriptor {
            shape: Shape::new([8, 8]),
            data_type: DataType::F32,
        });
        let matmul = graph.new_op(Op::Matmul(Matmul {
            input_a: input,
            input_b: input,
            precision: Precision::MulBf16AccumF32,
        }));
        graph.new_output(matmul);

        let generator =
            MatmulGenerator::new(&OpSubgraph::from_nodes(&Arc::new(graph), vec![matmul])).unwrap();
        assert_eq!(generator.leafs.len(), 1);
    }

    #[test]
    fn multi_leaf_kernel_is_valid() {
        let mut graph = OpGraph::new();
        let input = graph.new_input(Descriptor {
            shape: Shape::new([8, 8]),
            data_type: DataType::F32,
        });
        let matmul = graph.new_op(Op::Matmul(Matmul {
            input_a: input,
            input_b: input,
            precision: Precision::MulBf16AccumF32,
        }));
        graph.new_output(matmul);
        let pointwise = graph.new_op(Op::UnaryPointwise(UnaryPointwise {
            op: UnaryPointwiseOp::Exp,
            input: matmul,
        }));
        graph.new_output(pointwise);

        let generator = MatmulGenerator::new(&OpSubgraph::from_nodes(
            &Arc::new(graph),
            vec![matmul, pointwise],
        ))
        .unwrap();
        assert_eq!(generator.leafs.len(), 2);
    }
}
