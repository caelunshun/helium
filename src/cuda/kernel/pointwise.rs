//! Generation of pointwise kernels. These take `n` input tensors
//! of the same size, apply one or more pointwise operations to each
//! tuple of elements in those tensors, and places them into a single
//! output tensor.
//!
//! All input tensors and the output tensor have the same length.

use crate::{
    cuda::kernel::{cpp_type_name, Context, Kernel, KernelParam},
    opgraph::{
        op::{BinaryPointwiseOp, Op, UnaryPointwiseOp},
        subgraph::OpSubgraph,
        Node,
    },
};
use indoc::formatdoc;
use slotmap::SecondaryMap;

fn generate_for_unary(op: UnaryPointwiseOp, input: &str, cx: &Context) -> String {
    match op {
        UnaryPointwiseOp::AddScalar(x) => {
            let x = cx.var(x);
            format!("{input} + {x}")
        }
        UnaryPointwiseOp::MulScalar(x) => {
            let x = cx.var(x);
            format!("{input} * {x}")
        }
        UnaryPointwiseOp::PowScalar(x) => {
            let x = cx.var(x);
            format!("powf({input}, {x})")
        }
        UnaryPointwiseOp::Neg => format!("-{input}"),
        UnaryPointwiseOp::Exp => format!("expf({input})"),
        UnaryPointwiseOp::Sin => format!("sinf({input})"),
        UnaryPointwiseOp::Cos => format!("cosf({input})"),
        UnaryPointwiseOp::Tan => format!("tanf({input})"),
        UnaryPointwiseOp::Sigmoid => format!("1.0 / (1.0 + expf({input})"),
        UnaryPointwiseOp::Tanh => format!("tanhf{input})"),
        UnaryPointwiseOp::Relu => format!("fmaxf(0, {input})"),
    }
}

fn generate_for_binary(op: BinaryPointwiseOp, lhs: &str, rhs: &str) -> String {
    match op {
        BinaryPointwiseOp::Add => format!("{lhs} + {rhs}"),
        BinaryPointwiseOp::Mul => format!("{lhs} * {rhs}"),
        BinaryPointwiseOp::Pow => format!("powf({lhs}, {rhs})"),
    }
}

/// Generates a CUDA C++ kernel that applies a sequence of pointwise
/// operations to its inputs as defined by the given `OpSubgraph`.
pub fn generate_kernel(subgraph: &OpSubgraph) -> Kernel {
    let mut cx = Context::new();

    let mut params = Vec::new();
    let mut params_code = String::new();
    let mut statements = String::new();

    for input in subgraph.inputs() {
        let input_ident = cx.insert_input(input);
        let typ = subgraph.graph().get(input).descriptor().data_type;
        let typ = cpp_type_name(typ);
        params.push(KernelParam::Node(input));
        params_code.push_str(&format!("{typ} *{input_ident}, "));

        // Load from memory
        let intermediate_ident = cx.insert_intermediate(input);
        statements.push_str(&format!(
            "float {intermediate_ident} = {input_ident}[index];\n"
        ));
    }

    for var in subgraph.referenced_vars() {
        let var_ident = cx.insert_var(var);
        params.push(KernelParam::Var(var));
        params_code.push_str(&format!("float {var_ident}, "));
    }

    // Output tensor
    let output_type = subgraph.graph().get(subgraph.leaf()).descriptor().data_type;
    let output_type = cpp_type_name(output_type);
    params.push(KernelParam::Output);
    params_code.push_str(&format!("{output_type} *out"));

    params.push(KernelParam::Size);

    statements.push_str(&generate_pointwise_statements(subgraph, &mut cx));

    let out_ident = cx.intermediate(subgraph.leaf());

    let code = formatdoc! {"
        #include <math.h>
        #include <cuda_fp16.h>
        #include <cuda_bf16.h>

        __global__ void generatedPointwiseKernel({params_code}, size_t size) {{
            size_t index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index >= size) return;
            {statements}
            out[index] = static_cast<{output_type}>({out_ident});
        }}
    "};
    Kernel { code, params }
}

/// Generates a list of CUDA C++ statements to evaluate the given subgraph of
/// pointwise operations. All inputs must be already loaded from memory and
/// provided in the context intermediates map. The output will be stored
/// in a variable whose identifier is placed in the context intermediates map.
pub fn generate_pointwise_statements(subgraph: &OpSubgraph, cx: &mut Context) -> String {
    let mut code = String::new();

    let mut indegrees = SecondaryMap::new();
    let mut stack = Vec::new();
    for node in subgraph.nodes() {
        let indegree = subgraph.internal_indegree(node);
        indegrees.insert(node, indegree);
        if indegree == 0 {
            stack.push(node);
        }
    }

    let mut processed_count = 0;
    while let Some(node_id) = stack.pop() {
        let node = subgraph.graph().get(node_id);
        if let Node::Intermediate(node) = node {
            let output = cx.insert_intermediate(node_id);
            match &node.op {
                Op::UnaryPointwise(unary) => {
                    let input = cx.intermediate(unary.input);
                    code.push_str(&format!(
                        "float {output} = {};\n",
                        generate_for_unary(unary.op, input, cx)
                    ));
                }
                Op::BinaryPointwise(binary) => {
                    let lhs = cx.intermediate(binary.lhs);
                    let rhs = cx.intermediate(binary.rhs);
                    code.push_str(&format!(
                        "float {output} = {};\n",
                        generate_for_binary(binary.op, lhs, rhs)
                    ));
                }
                Op::ChangeDataType(change) => {
                    // Nothing to do: data type change only affects writes
                    // to memory, which happen at the end of the kernel.
                    let input = cx.intermediate(change.input);
                    code.push_str(&format!("float {output} = {input};\n"));
                }
                _ => panic!("illegal op for pointwise: {:?}", node.op),
            }
        }

        for outbound in subgraph.graph().outbound_edges(node_id) {
            if subgraph.contains_node(*outbound) {
                let indegree = indegrees.get_mut(*outbound).unwrap();
                *indegree = indegree.checked_sub(1).unwrap();
                if *indegree == 0 {
                    stack.push(*outbound);
                }
            }
        }

        processed_count += 1;
    }

    assert_eq!(
        processed_count,
        subgraph.num_nodes(),
        "op graph cannot contain cycles"
    );

    code
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        data_type::DataType,
        opgraph::{
            op::{BinaryPointwise, ChangeDataType, UnaryPointwise},
            Descriptor, OpGraph,
        },
    };

    #[test]
    fn test_generate_kernel() {
        let mut graph = OpGraph::new();
        let a = graph.new_input(Descriptor {
            dimension: 2,
            data_type: DataType::Bf16,
        });
        let b = graph.new_input(Descriptor {
            dimension: 2,
            data_type: DataType::F32,
        });

        let c = graph.new_op(Op::BinaryPointwise(BinaryPointwise {
            lhs: a,
            rhs: b,
            op: BinaryPointwiseOp::Pow,
        }));

        let d = graph.new_op(Op::UnaryPointwise(UnaryPointwise {
            input: a,
            op: UnaryPointwiseOp::Relu,
        }));

        let var = graph.new_var();
        let e = graph.new_op(Op::UnaryPointwise(UnaryPointwise {
            input: d,
            op: UnaryPointwiseOp::PowScalar(var),
        }));

        let out = graph.new_op(Op::BinaryPointwise(BinaryPointwise {
            lhs: c,
            rhs: e,
            op: BinaryPointwiseOp::Mul,
        }));

        let out_casted = graph.new_op(Op::ChangeDataType(ChangeDataType {
            input: out,
            target_type: DataType::F16,
        }));

        graph.new_output(out_casted);

        let subgraph = OpSubgraph::from_nodes(&graph, vec![c, d, e, out, out_casted]);

        let kernel = generate_kernel(&subgraph);
        insta::assert_debug_snapshot!(kernel);
    }
}
