use crate::{
    cuda::kernel_jit::{Ident, KernelBuilder, KernelParam},
    opgraph::{
        op::{self, BinaryPointwiseOp, Broadcast, Op, UnaryPointwiseOp},
        subgraph::OpSubgraph,
        Intermediate, Node, NodeId,
    },
    shape::Shape,
};
use ahash::AHashMap;
use indoc::formatdoc;
use std::fmt::Write;

pub const BLOCK_SIZE: usize = 256;

/// Generates a kernel to execute the given pointwise op graph.
pub fn generate_kernel(subgraph: &OpSubgraph) -> KernelBuilder {
    let mut kernel = KernelBuilder::new();
    let mut cx = Context::default();

    for input in subgraph.inputs() {
        cx.input_vars.insert(
            input,
            kernel.param(
                KernelParam::Input(input),
                subgraph.graph().get(input).descriptor().data_type,
            ),
        );
    }

    if is_reduction_graph(subgraph) {
        kernel.statement(format!("__shared__ float reduction_mem[{BLOCK_SIZE}];"));
        //kernel.register_shared_memory_bytes((BLOCK_SIZE * mem::size_of::<f32>()) as u32);
    }

    let ReductionStrides { group_size, .. } = compute_reduction_stride(subgraph);
    let blocks_per_group = (group_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let threads_per_group = blocks_per_group * BLOCK_SIZE;
    let group_size_rounded_up =
        (group_size + threads_per_group - 1) / threads_per_group * threads_per_group;

    kernel.statement(format!(
        "uint32_t group = (blockDim.x * blockIdx.x) / {group_size_rounded_up};"
    ));
    kernel.statement(format!(
        "uint32_t block_group_start = blockIdx.x / {blocks_per_group} * {blocks_per_group};"
    ));
    kernel.statement(format!(
        "uint32_t index_in_group = (blockIdx.x - block_group_start) * blockDim.x + threadIdx.x;"
    ));
    kernel.statement(format!(
        "uint32_t out_index = {group_size} * group + index_in_group;"
    ));
    kernel.statement(format!("if (index_in_group >= {group_size}) return;"));

    for output in subgraph.leafs() {
        let val = compute_node_output(
            subgraph,
            &Position {
                node: output,
                index_mapping: IndexMapping::Identity,
            },
            &mut kernel,
            &mut cx,
        );
        let dtype = subgraph.graph().get(output).descriptor().data_type;
        let param = kernel.param(KernelParam::Output(output), dtype);
        let typ = KernelBuilder::cpp_data_type(dtype);
        kernel.statement(format!("{param}[out_index] = static_cast<{typ}>({val});"));
    }

    kernel
}

struct ReductionStrides {
    group_size: usize,
    num_groups: usize,
}

fn compute_reduction_stride(subgraph: &OpSubgraph) -> ReductionStrides {
    let output_shape = compute_output_shape(subgraph);
    for node in subgraph.nodes() {
        if let Node::Intermediate(Intermediate {
            op: Op::Reduce(reduce),
            ..
        }) = subgraph.graph().get(node)
        {
            let group_size = output_shape.dims()
                [output_shape.num_dims() - reduce.depth as usize - 1..]
                .iter()
                .copied()
                .product();
            let num_groups = output_shape.num_elements() / group_size;
            return ReductionStrides {
                group_size,
                num_groups,
            };
        }
    }
    // No reduction; group size is same as output size
    ReductionStrides {
        group_size: output_shape.num_elements(),
        num_groups: 1,
    }
}

fn compute_output_shape(subgraph: &OpSubgraph) -> &Shape {
    let mut node = subgraph.leafs().next().expect("no leaf in subgraph");

    if let Node::Intermediate(Intermediate {
        op: Op::Reduce(reduce),
        ..
    }) = subgraph.graph().get(node)
    {
        node = reduce.input;
    }

    &subgraph.graph().get(node).descriptor().shape
}

fn is_reduction_graph(subgraph: &OpSubgraph) -> bool {
    subgraph.nodes().any(|node| {
        matches!(
            subgraph.graph().get(node),
            Node::Intermediate(Intermediate {
                op: Op::Reduce(_),
                ..
            })
        )
    })
}

#[derive(Default, Debug)]
struct Context {
    input_vars: AHashMap<NodeId, Ident>,
    results_at_position: AHashMap<Position, Ident>,
}

fn compute_node_output(
    subgraph: &OpSubgraph,
    position: &Position,
    kernel: &mut KernelBuilder,
    cx: &mut Context,
) -> Ident {
    if let Some(res) = cx.results_at_position.get(position) {
        return res.clone();
    }

    let Position {
        node,
        index_mapping,
    } = position;
    let node = *node;

    let ident = kernel.new_ident();
    cx.results_at_position
        .insert(position.clone(), ident.clone());

    if subgraph.inputs().any(|x| x == node) {
        // Load from memory.
        let index = index_mapping.derive_index("out_index", kernel);
        let array = &cx.input_vars[&node];
        kernel.statement(format!(
            "float {ident} = static_cast<float>({array}[{index}]);"
        ));
        return ident;
    }

    let Node::Intermediate(Intermediate { op, .. }) = subgraph.graph().get(node) else {
        unreachable!("internal node must be an intermediate")
    };

    match op {
        Op::Transpose(op::Transpose { input }) => {
            let new_mapping = IndexMapping::Compose {
                first: Box::new(index_mapping.clone()),
                second: Box::new(IndexMapping::Transpose {
                    in_shape: subgraph.graph().get(*input).descriptor().shape.clone(),
                }),
            };
            return compute_node_output(
                subgraph,
                &Position {
                    node: *input,
                    index_mapping: new_mapping,
                },
                kernel,
                cx,
            );
        }
        Op::Broadcast(broadcast) => {
            let in_shape = subgraph
                .graph()
                .get(broadcast.input)
                .descriptor()
                .shape
                .clone();
            let new_mapping = IndexMapping::Compose {
                first: Box::new(index_mapping.clone()),
                second: Box::new(IndexMapping::Broadcast {
                    broadcast: broadcast.clone(),
                    in_shape,
                }),
            };
            return compute_node_output(
                subgraph,
                &Position {
                    node: broadcast.input,
                    index_mapping: new_mapping,
                },
                kernel,
                cx,
            );
        }
        Op::UnaryPointwise(op::UnaryPointwise { input, op }) => {
            let input = compute_node_output(
                subgraph,
                &Position {
                    node: *input,
                    index_mapping: index_mapping.clone(),
                },
                kernel,
                cx,
            );
            kernel.statement(format!(
                "float {ident} = {};",
                unary_pointwise_op(&input, *op)
            ));
        }
        Op::BinaryPointwise(op::BinaryPointwise { lhs, rhs, op }) => {
            let lhs = compute_node_output(
                subgraph,
                &Position {
                    node: *lhs,
                    index_mapping: index_mapping.clone(),
                },
                kernel,
                cx,
            );
            let rhs = compute_node_output(
                subgraph,
                &Position {
                    node: *rhs,
                    index_mapping: index_mapping.clone(),
                },
                kernel,
                cx,
            );
            kernel.statement(format!(
                "float {ident} = {};",
                binary_pointwise_op(&lhs, &rhs, *op)
            ));
        }
        Op::Reduce(_) => todo!(),
        // No-ops in the context of pointwise kernel
        // (ChangeDataType happens when loading inputs / storing
        // outputs, all compute is in float32)
        Op::Reshape(op::Reshape { input, .. })
        | Op::ChangeDataType(op::ChangeDataType { input, .. }) => {
            return compute_node_output(
                subgraph,
                &Position {
                    node: *input,
                    index_mapping: index_mapping.clone(),
                },
                kernel,
                cx,
            );
        }
        _ => unreachable!("op {op:#?} not supported by PointwiseGraph"),
    }

    ident
}

fn unary_pointwise_op(input: &str, op: UnaryPointwiseOp) -> String {
    match op {
        UnaryPointwiseOp::Recip => format!("1.0f / {input}"),
        UnaryPointwiseOp::Neg => format!("-{input}"),
        UnaryPointwiseOp::Exp => format!("expf({input})"),
        UnaryPointwiseOp::Sin => format!("sinf({input})"),
        UnaryPointwiseOp::Cos => format!("cosf({input})"),
        UnaryPointwiseOp::Tan => format!("tanf({input})"),
        UnaryPointwiseOp::Sigmoid => format!("1.0f / (1.0f + expf(-{input})"),
        UnaryPointwiseOp::Tanh => format!("tanhf({input})"),
    }
}

fn binary_pointwise_op(lhs: &str, rhs: &str, op: BinaryPointwiseOp) -> String {
    match op {
        BinaryPointwiseOp::Add => format!("{lhs} + {rhs}"),
        BinaryPointwiseOp::Mul => format!("{lhs} * {rhs}"),
        BinaryPointwiseOp::Pow => format!("powf({lhs}, {rhs})"),
        BinaryPointwiseOp::Min => format!("fminf({lhs}, {rhs})"),
        BinaryPointwiseOp::Max => format!("fmaxf({lhs}, {rhs})"),
    }
}

/// Memoization key.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Position {
    node: NodeId,
    /// Index mapping up to this point.
    index_mapping: IndexMapping,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum IndexMapping {
    Identity,
    Transpose {
        in_shape: Shape,
    },
    Broadcast {
        broadcast: Broadcast,
        in_shape: Shape,
    },
    Compose {
        first: Box<IndexMapping>,
        second: Box<IndexMapping>,
    },
}

impl IndexMapping {
    pub fn derive_index(&self, current: &str, kernel: &mut KernelBuilder) -> Ident {
        let ident = kernel.new_ident();
        let expr = match self {
            IndexMapping::Identity => format!("{current}"),
            IndexMapping::Transpose { in_shape } => {
                let x = in_shape.dim_at(-1);
                let y = in_shape.dim_at(-2);
                format!("({current} % {y}) * {x} + ({current} / {y})")
            }
            IndexMapping::Broadcast {
                broadcast,
                in_shape,
            } => {
                let out_shape = broadcast.output_shape(in_shape);

                let temp_var = kernel.new_ident();

                let mut out_stride = 1;
                let mut in_stride = 1;
                let mut compute_dims = String::new();
                for (i, dim) in out_shape.dims().iter().rev().copied().enumerate() {
                    writeln!(
                        compute_dims,
                        "uint32_t coord_out_dim{i} = ({current} / {out_stride}) % {dim};"
                    )
                    .unwrap();
                    out_stride *= dim;

                    let j = in_shape
                        .num_dims()
                        .checked_sub(i)
                        .and_then(|x| x.checked_sub(1));
                    match j {
                        Some(_) => {
                            if broadcast.is_axis_broadcasted(out_shape.num_dims() - i - 1) {
                                writeln!(compute_dims, "uint32_t coord_in_dim{i} = 0;").unwrap();
                            } else {
                                writeln!(
                                    compute_dims,
                                    "uint32_t coord_in_dim{i} = coord_out_dim{i};"
                                )
                                .unwrap();
                            }
                        }
                        None => {
                            writeln!(compute_dims, "uint32_t coord_in_dim{i} = 0;").unwrap();
                        }
                    }

                    let in_dim = j.map(|j| in_shape.dims()[j]).unwrap_or(1);
                    writeln!(compute_dims, "in_index += coord_in_dim{i} * {in_stride};").unwrap();
                    in_stride *= in_dim;
                }

                kernel.statement(formatdoc! {"
                    uint32_t {temp_var};
                    {{
                        uint32_t in_index = 0;
                        {compute_dims}
                        {temp_var} = in_index;
                    }}
                "});
                temp_var
            }
            IndexMapping::Compose { first, second } => {
                let current2 = first.derive_index(current, kernel);
                second.derive_index(&current2, kernel)
            }
        };
        kernel.statement(format!("uint32_t {ident} = {expr};"));
        ident
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cuda::context::CudaContext,
        opgraph::{Descriptor, OpGraph},
        DataType,
    };
    use op::BroadcastAxis;
    use std::sync::Arc;

    #[test]
    fn no_reduction() {
        let mut graph = OpGraph::new();
        let in_a = graph.new_input(Descriptor {
            shape: Shape::new([5, 10, 10]),
            data_type: DataType::Bf16,
        });
        let in_b = graph.new_input(Descriptor {
            shape: Shape::new([1, 10]),
            data_type: DataType::F16,
        });

        let broadcast_b = graph.new_op(Op::Broadcast(Broadcast {
            input: in_b,
            new_dim_count: 3,
            broadcast_axes: vec![
                BroadcastAxis {
                    axis: 0,
                    new_size: 5,
                },
                BroadcastAxis {
                    axis: 1,
                    new_size: 10,
                },
            ],
        }));
        assert_eq!(
            graph.get(broadcast_b).descriptor().shape,
            Shape::new([5, 10, 10])
        );

        let x = graph.new_op(Op::BinaryPointwise(op::BinaryPointwise {
            lhs: in_a,
            rhs: broadcast_b,
            op: BinaryPointwiseOp::Pow,
        }));
        let y = graph.new_op(Op::UnaryPointwise(op::UnaryPointwise {
            input: in_b,
            op: UnaryPointwiseOp::Cos,
        }));
        graph.new_output(x);
        graph.new_output(y);

        let kernel = generate_kernel(&OpSubgraph::from_nodes(
            &Arc::new(graph),
            vec![broadcast_b, x, y],
        ));
        insta::assert_snapshot!(kernel.build_source("test"));
        kernel
            .build("test", &CudaContext::global(0).unwrap())
            .unwrap();
    }
}
