use crate::{
    cuda::kernel_jit::{Ident, KernelBuilder, KernelParam},
    data_type::DataClass,
    opgraph::{
        op::{self, BinaryPointwiseOp, Broadcast, CompareOp, Op, ReduceOp, UnaryPointwiseOp},
        subgraph::OpSubgraph,
        Intermediate, Node, NodeId,
    },
    shape::Shape,
    DataType,
};
use ahash::AHashMap;
use indoc::{formatdoc, indoc};
use std::fmt::Write;

pub const BLOCK_SIZE: usize = 256;

/// Generates a kernel to execute the given pointwise op graph.
pub fn generate_kernel(subgraph: &OpSubgraph) -> KernelBuilder {
    let mut kernel = KernelBuilder::new();
    let mut cx = Context::default();

    kernel.statement(format!("/* {subgraph:#?} */"));

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
        kernel.include("math_constants.h");
        kernel.item(indoc! {"
        __device__ __forceinline__ float atomicMinFloat(float *addr, float value) {
            float old;
            old = (value >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(value))) :
                __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));
    
            return old;
        }

        __device__ __forceinline__ float atomicMaxFloat(float *addr, float value) {
            float old;
            old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
                __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));
    
            return old;
        }
        "});
    }

    let ReductionStrides { group_size, .. } = compute_reduction_stride(subgraph);
    let blocks_per_group = group_size.div_ceil(BLOCK_SIZE);

    kernel.statement(format!("uint32_t group_size = {group_size};"));
    kernel.statement(format!("uint32_t group = blockIdx.x / {blocks_per_group};"));
    kernel.statement(format!(
        "uint32_t block_group_start = blockIdx.x / {blocks_per_group} * {blocks_per_group};"
    ));
    kernel.statement(format!(
        "uint32_t index_in_group = (blockIdx.x - block_group_start) * blockDim.x + threadIdx.x;"
    ));
    kernel.statement(format!(
        "uint32_t out_index = {group_size} * group + index_in_group;"
    ));
    kernel.statement(format!("bool active = index_in_group < {group_size};"));
    kernel.statement(format!(
        "uint32_t active_mask = __ballot_sync(0xffffffff, active);"
    ));
    kernel.statement(format!("if (!active) return;"));

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

        // Store result to memory.
        // If a reduction, then the store was already performed in compute_node_output.
        if !matches!(
            subgraph.graph().get(output),
            Node::Intermediate(Intermediate {
                op: Op::Reduce(_),
                ..
            })
        ) {
            let dtype = subgraph.graph().get(output).descriptor().data_type;
            let param = kernel.param(KernelParam::Output(output), dtype);
            let typ = KernelBuilder::cpp_data_type(dtype);

            match dtype.class() {
                DataClass::Float | DataClass::Int => {
                    kernel.statement(format!("{param}[out_index] = static_cast<{typ}>({val});"));
                }
                DataClass::Bool => {
                    kernel.statement(formatdoc! {"
                    if ({val}) {{
                        atomicOr({param} + out_index / 32, 1 << (out_index % 32));
                    }} else {{
                        atomicAnd({param} + out_index / 32, ~(1 << (out_index % 32)));
                    }}
                    "});
                }
            }
        }
    }

    kernel
}

pub fn compute_grid_size(subgraph: &OpSubgraph) -> usize {
    let ReductionStrides {
        group_size,
        num_groups,
    } = compute_reduction_stride(subgraph);
    let blocks_per_group = group_size.div_ceil(BLOCK_SIZE);
    blocks_per_group * num_groups
}

struct ReductionStrides {
    group_size: usize,
    num_groups: usize,
}

fn compute_reduction_stride(subgraph: &OpSubgraph) -> ReductionStrides {
    for node in subgraph.nodes() {
        if let Node::Intermediate(Intermediate {
            op: Op::Reduce(reduce),
            ..
        }) = subgraph.graph().get(node)
        {
            let input_shape = &subgraph.graph().get(reduce.input).descriptor().shape;

            let group_size = input_shape.dims()[(input_shape.num_dims() - reduce.depth as usize)..]
                .iter()
                .copied()
                .product();
            let num_groups = input_shape.num_elements() / group_size;
            return ReductionStrides {
                group_size,
                num_groups,
            };
        }
    }
    // No reduction; group size is same as output size
    ReductionStrides {
        group_size: compute_output_shape(subgraph).num_elements(),
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

    if let Some(input) = subgraph.inputs().find(|x| *x == node) {
        // Load from memory.
        let index = index_mapping.derive_index("out_index", kernel);
        let array = &cx.input_vars[&node];

        let input = subgraph.graph().get(input);
        let class = input.descriptor().data_type.class();
        match class {
            DataClass::Float => {
                kernel.statement(format!(
                    "float {ident} = static_cast<float>({array}[{index}]);"
                ));
            }
            DataClass::Int => {
                kernel.statement(format!(
                    "uint32_t {ident} = static_cast<uint32_t>({array}[{index}]);"
                ));
            }
            DataClass::Bool => {
                kernel.statement(format!(
                    "bool {ident} = (({array}[{index} / 32] >> ({index} % 32)) & 1) != 0;"
                ));
            }
        }

        return ident;
    }

    let Node::Intermediate(Intermediate { op, descriptor }) = subgraph.graph().get(node) else {
        unreachable!("internal node must be an intermediate")
    };

    match op {
        Op::SwapDims(op::SwapDims {
            input,
            axis_a,
            axis_b,
        }) => {
            let new_mapping = IndexMapping::Compose {
                first: Box::new(index_mapping.clone()),
                second: Box::new(IndexMapping::SwapDims {
                    in_shape: subgraph.graph().get(*input).descriptor().shape.clone(),
                    axis_a: *axis_a,
                    axis_b: *axis_b,
                }),
            };
            let res = compute_node_output(
                subgraph,
                &Position {
                    node: *input,
                    index_mapping: new_mapping,
                },
                kernel,
                cx,
            );
            cx.results_at_position.insert(position.clone(), res.clone());
            return res;
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
            let res = compute_node_output(
                subgraph,
                &Position {
                    node: broadcast.input,
                    index_mapping: new_mapping,
                },
                kernel,
                cx,
            );
            cx.results_at_position.insert(position.clone(), res.clone());
            return res;
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
                "{} {ident} = {};",
                cpp_data_class(descriptor.data_type.class()),
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
                "{} {ident} = {};",
                cpp_data_class(descriptor.data_type.class()),
                binary_pointwise_op(&lhs, &rhs, *op, descriptor.data_type.class())
            ));
        }
        Op::Compare(op::Compare { lhs, rhs, op }) => {
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
            kernel.statement(format!("bool {ident} = {};", compare_op(&lhs, &rhs, *op)));
        }
        Op::Select(op::Select { lhs, rhs, selector }) => {
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
            let selector = compute_node_output(
                subgraph,
                &Position {
                    node: *selector,
                    index_mapping: index_mapping.clone(),
                },
                kernel,
                cx,
            );
            kernel.statement(format!(
                "{} {ident} = {selector} ? {rhs} : {lhs};",
                cpp_data_class(descriptor.data_type.class())
            ));
        }
        Op::Reduce(op) => {
            let mut input = compute_node_output(
                subgraph,
                &Position {
                    node: op.input,
                    index_mapping: index_mapping.clone(),
                },
                kernel,
                cx,
            );
            let reduced_val = kernel.new_ident();
            if op.op == ReduceOp::Mean {
                let coefficient = (compute_reduction_stride(subgraph).group_size as f64).recip();
                let new_input = kernel.new_ident();
                kernel.statement(format!("float {new_input} = {input} * {coefficient};"));
                input = new_input;
            }

            let reduce_block_level = reduce_op(&reduced_val, "reduction_mem[partner_idx]", op.op);
            let reduce_warp_level = reduce_op(&reduced_val, "partner", op.op);

            let output = kernel.param(KernelParam::Output(node), DataType::F32);
            let reduce_grid_level =
                atomic_reduce(&reduced_val, &format!("{output} + group"), op.op);

            kernel.statement(formatdoc! {"
            // Block-level reduction
            float {reduced_val} = {input};
            for (uint32_t offset = {BLOCK_SIZE} / 2; offset >= 32; offset /= 2) {{
                reduction_mem[threadIdx.x] = {reduced_val};
                __syncthreads();
                if (threadIdx.x < offset) {{
                    if (index_in_group + offset < group_size) {{
                        uint32_t partner_idx = threadIdx.x + offset;
                        {reduced_val} = {reduce_block_level};
                    }}
                }}
            }}

            // Warp-level reduction (last 32 elements in block)
            __syncthreads();
            if (threadIdx.x < 32) {{
                for (uint32_t offset = 16; offset > 0; offset /= 2) {{
                    float partner = __shfl_down_sync(active_mask, {reduced_val}, offset);
                    if (index_in_group + offset < group_size) {{
                        {reduced_val} = {reduce_warp_level};
                    }}
                }}

                // Grid-level reduction across blocks
                if (threadIdx.x == 0) {{
                    {reduce_grid_level}
                }}
            }}
            "});
        }
        Op::ChangeDataType(op::ChangeDataType { input, target_type })
            if target_type.class()
                != subgraph.graph().get(*input).descriptor().data_type.class() =>
        {
            // Change in data class.
            let in_class = subgraph.graph().get(*input).descriptor().data_type.class();
            let out_class = target_type.class();

            let input = compute_node_output(
                subgraph,
                &Position {
                    node: *input,
                    index_mapping: index_mapping.clone(),
                },
                kernel,
                cx,
            );

            let expr = match (in_class, out_class) {
                (DataClass::Float, DataClass::Bool) => format!("{input} != 0.0f"),
                (DataClass::Bool, DataClass::Float) => format!("{input} ? 1.0f : 0.0f"),
                (DataClass::Float, DataClass::Int) => format!("static_cast<uint32_t>({input})"),
                (DataClass::Int, DataClass::Float) => format!("static_cast<float>({input})"),
                (DataClass::Bool, DataClass::Int) => format!("{input} ? 1 : 0"),
                (DataClass::Int, DataClass::Bool) => format!("{input} != 0"),
                _ => unreachable!("in_class != out_class"),
            };
            kernel.statement(format!("{} {ident} = {expr};", cpp_data_class(out_class)));
        }
        // No-ops in the context of pointwise kernel
        // (different data type within same data class only affects
        // load/stores and not intermediate computations)
        Op::Reshape(op::Reshape { input, .. })
        | Op::ChangeDataType(op::ChangeDataType { input, .. }) => {
            let res = compute_node_output(
                subgraph,
                &Position {
                    node: *input,
                    index_mapping: index_mapping.clone(),
                },
                kernel,
                cx,
            );
            cx.results_at_position.insert(position.clone(), res.clone());
            return res;
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
        UnaryPointwiseOp::Sigmoid => format!("1.0f / (1.0f + expf(-{input}))"),
        UnaryPointwiseOp::Tanh => format!("tanhf({input})"),
        UnaryPointwiseOp::Relu => format!("fmaxf({input}, 0.0)"),
        UnaryPointwiseOp::Log => format!("logf({input})"),
        UnaryPointwiseOp::Sqrt => format!("sqrtf({input})"),
    }
}

fn binary_pointwise_op(lhs: &str, rhs: &str, op: BinaryPointwiseOp, class: DataClass) -> String {
    match op {
        BinaryPointwiseOp::Add => format!("{lhs} + {rhs}"),
        BinaryPointwiseOp::Mul => format!("{lhs} * {rhs}"),
        BinaryPointwiseOp::Pow => format!("powf({lhs}, {rhs})"),
        BinaryPointwiseOp::Min if class == DataClass::Float => format!("fminf({lhs}, {rhs})"),
        BinaryPointwiseOp::Min => format!("min({lhs}, {rhs})"),
        BinaryPointwiseOp::Max if class == DataClass::Float => format!("fmaxf({lhs}, {rhs})"),
        BinaryPointwiseOp::Max => format!("max({lhs}, {rhs})"),
    }
}

fn compare_op(lhs: &str, rhs: &str, op: CompareOp) -> String {
    match op {
        CompareOp::Equal => format!("{lhs} == {rhs}"),
        CompareOp::NotEqual => format!("{lhs} != {rhs}"),
        CompareOp::LessThan => format!("{lhs} < {rhs}"),
        CompareOp::LessThanOrEqual => format!("{lhs} <= {rhs}"),
        CompareOp::GreaterThan => format!("{lhs} > {rhs}"),
        CompareOp::GreaterThanOrEqual => format!("{lhs} >= {rhs}"),
    }
}

fn reduce_op(a: &str, b: &str, op: ReduceOp) -> String {
    match op {
        ReduceOp::Sum | ReduceOp::Mean => format!("{a} + {b}"),
        ReduceOp::Max => format!("fmaxf({a}, {b})"),
        ReduceOp::Min => format!("fminf({a}, {b})"),
    }
}

fn atomic_reduce(val: &str, addr: &str, op: ReduceOp) -> String {
    match op {
        ReduceOp::Sum | ReduceOp::Mean => format!("atomicAdd({addr}, {val});"),
        ReduceOp::Max => format!("atomicMaxFloat({addr}, {val});"),
        ReduceOp::Min => format!("atomicMinFloat({addr}, {val});"),
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
    SwapDims {
        in_shape: Shape,
        axis_a: usize,
        axis_b: usize,
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
            IndexMapping::Identity => current.to_string(),
            IndexMapping::SwapDims {
                in_shape,
                axis_a,
                axis_b,
            } => {
                let temp_var = kernel.new_ident();

                let mut out_shape = in_shape.dims().to_vec();
                out_shape.swap(*axis_a, *axis_b);
                let out_shape = Shape::new(out_shape);

                let mut out_stride = 1;
                let mut compute_dims = String::new();
                for (i, dim_out) in out_shape.dims().iter().copied().enumerate().rev() {
                    writeln!(
                        compute_dims,
                        "uint32_t coord_dim{i} = ({current} / {out_stride}) % {dim_out};"
                    )
                    .unwrap();
                    out_stride *= dim_out;
                }

                let mut in_stride = 1;
                for j in (0..in_shape.num_dims()).rev() {
                    let k = if j == *axis_a {
                        *axis_b
                    } else if j == *axis_b {
                        *axis_a
                    } else {
                        j
                    };
                    writeln!(compute_dims, "in_index += coord_dim{k} * {in_stride};").unwrap();
                    in_stride *= in_shape.dims()[j];
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

/// Returns the C++ type used for intermediate computations
/// in the given data class.
fn cpp_data_class(class: DataClass) -> &'static str {
    match class {
        DataClass::Float => "float",
        DataClass::Int => "uint32_t",
        DataClass::Bool => "bool",
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
