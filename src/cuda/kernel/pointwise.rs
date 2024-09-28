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
        Intermediate, Node,
    },
};
use indoc::formatdoc;
use slotmap::SecondaryMap;

pub const KERNEL_NAME: &str = "generatedPointwiseKernel";

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
        UnaryPointwiseOp::Sigmoid => format!("1.0 / (1.0 + expf({input}))"),
        UnaryPointwiseOp::Tanh => format!("tanhf{input})"),
        UnaryPointwiseOp::Relu => format!("fmaxf(0, {input})"),
        UnaryPointwiseOp::Recip => format!("__frcp_rn({input})"),
    }
}

fn generate_for_binary(op: BinaryPointwiseOp, lhs: &str, rhs: &str) -> String {
    match op {
        BinaryPointwiseOp::Add => format!("{lhs} + {rhs}"),
        BinaryPointwiseOp::Mul => format!("{lhs} * {rhs}"),
        BinaryPointwiseOp::Pow => format!("powf({lhs}, {rhs})"),
    }
}

pub fn load_inputs(
    params: &mut Vec<KernelParam>,
    params_code: &mut String,
    subgraph: &OpSubgraph,
    statements: &mut String,
    cx: &mut Context,
) {
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
}

pub fn store_outputs(
    params: &mut Vec<KernelParam>,
    params_code: &mut String,
    subgraph: &OpSubgraph,
    statements: &mut String,
    cx: &mut Context,
) {
    for leaf_id in subgraph.leafs() {
        if matches!(
            subgraph.graph().get(leaf_id),
            Node::Intermediate(Intermediate {
                op: Op::Reduce(_),
                ..
            })
        ) {
            // Handled in reduction kernel
            continue;
        }

        let descriptor = subgraph.graph().get(leaf_id).descriptor();
        let data_type = descriptor.data_type;
        params.push(KernelParam::Output(leaf_id));

        let output_ident = cx.generate_identifier();
        params_code.push_str(&format!("{} *{output_ident}, ", cpp_type_name(data_type)));

        let value_ident = cx.intermediate(leaf_id);
        statements.push_str(&format!(
            "{output_ident}[index] = static_cast<{}>({value_ident});\n",
            cpp_type_name(data_type)
        ));
    }
}

/// Generates a CUDA C++ kernel that applies a sequence of pointwise
/// operations to its inputs as defined by the given `OpSubgraph`.
pub fn generate_kernel(subgraph: &OpSubgraph) -> Kernel {
    let mut cx = Context::new();

    let mut params = Vec::new();
    let mut params_code = String::new();
    let mut statements = String::new();

    load_inputs(
        &mut params,
        &mut params_code,
        subgraph,
        &mut statements,
        &mut cx,
    );

    statements.push_str(&generate_pointwise_statements(subgraph, &mut cx));

    store_outputs(
        &mut params,
        &mut params_code,
        subgraph,
        &mut statements,
        &mut cx,
    );
    params.push(KernelParam::Size);

    let code = formatdoc! {"
        #include <cuda_fp16.h>
        #include <cuda_bf16.h>

        typedef unsigned int uint32_t;

        extern \"C\" __global__ void {KERNEL_NAME}({params_code} uint32_t size) {{
            uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index >= size) return;
            {statements}
        }}
    "};
    Kernel {
        code,
        params,
        entrypoint_name: KERNEL_NAME,
    }
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
        cuda::kernel::CompiledKernel,
        data_type::DataType,
        opgraph::{
            op::{BinaryPointwise, ChangeDataType, UnaryPointwise},
            Descriptor, OpGraph,
        },
    };
    use cudarc::driver::CudaDevice;
    use std::sync::Arc;

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
            op: UnaryPointwiseOp::Recip,
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

        let subgraph = OpSubgraph::from_nodes(&Arc::new(graph), vec![c, d, e, out, out_casted]);

        let kernel = generate_kernel(&subgraph);
        insta::assert_snapshot!(kernel.code);
        insta::assert_debug_snapshot!(kernel.params);

        CompiledKernel::new(&kernel, &CudaDevice::new(0).unwrap()).unwrap();
    }
}
