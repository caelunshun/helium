//! Generates reduction kernels. Can fuse with
//! pointwise operations immediately preceding
//! the reduction.
//!
//! Note that the output of a reduction is always in float32
//! format for numerical stability reasons. If a different output
//! type is needed, then an additional kernel will need to be generated
//! and executed.

use crate::{
    cuda::kernel::{
        pointwise::{generate_pointwise_statements, load_inputs, store_outputs},
        Context, Kernel, KernelParam,
    },
    opgraph::{
        op::{Op, ReduceOp},
        subgraph::OpSubgraph,
        Intermediate, Node,
    },
};
use indoc::formatdoc;

pub const KERNEL_NAME: &str = "generatedReductionKernel";

fn reduce_operation(reduction: ReduceOp, a: &str, b: &str) -> String {
    match reduction {
        ReduceOp::Sum | ReduceOp::Mean => format!("{a} + {b}"),
        ReduceOp::Max => format!("max({a}, {b})"),
        ReduceOp::Min => format!("min({a}, {b})"),
    }
}

fn atomic_reduce_operation(reduction: ReduceOp, addr: &str, val: &str) -> String {
    match reduction {
        ReduceOp::Sum | ReduceOp::Mean => format!("atomicAdd({addr}, {val});"),
        ReduceOp::Max => format!("atomicMaxFloat({addr}, {val});"),
        ReduceOp::Min => format!("atomicMinFloat({addr}, {val});"),
    }
}

fn init_val(reduction: ReduceOp) -> &'static str {
    match reduction {
        ReduceOp::Sum | ReduceOp::Mean => "0.0",
        ReduceOp::Max => "-CUDART_INF_F",
        ReduceOp::Min => "CUDART_INF_F",
    }
}

/// Generates a reduction kernel for the given subgraph.
///
/// The leaf node in the graph must be a reduction. All other
/// nodes must be pointwise operations.
pub fn generate_kernel(subgraph: &OpSubgraph) -> Kernel {
    let mut cx = Context::new();
    let mut params = Vec::new();
    let mut params_code = String::new();

    let mut pointwise_code = String::new();

    load_inputs(
        &mut params,
        &mut params_code,
        subgraph,
        &mut pointwise_code,
        &mut cx,
    );

    let reduction_leaf = subgraph
        .leafs()
        .find(|id| {
            matches!(
                subgraph.graph().get(*id),
                Node::Intermediate(Intermediate {
                    op: Op::Reduce(_),
                    ..
                })
            )
        })
        .expect("no reduction in subgraph");

    let pointwise_subgraph = OpSubgraph::from_nodes(
        subgraph.graph(),
        subgraph.nodes().filter(|n| *n != reduction_leaf).collect(),
    );
    pointwise_code.push_str(&generate_pointwise_statements(&pointwise_subgraph, &mut cx));

    store_outputs(
        &mut params,
        &mut params_code,
        subgraph,
        &mut pointwise_code,
        &mut cx,
    );

    params.push(KernelParam::Output(reduction_leaf));
    params_code.push_str("float *reductionOut, ");
    params.push(KernelParam::ReductionStride);
    params.push(KernelParam::Size);

    let Node::Intermediate(leaf) = subgraph.graph().get(reduction_leaf) else {
        panic!("leaf must be a reduction")
    };
    let Op::Reduce(reduce) = &leaf.op else {
        panic!("leaf must be a reduction")
    };

    let atomic_reduce = atomic_reduce_operation(reduce.op, "reductionOut + group", "val");
    let init_val = init_val(reduce.op);

    let mut input_ident = cx.intermediate(reduce.input);
    if reduce.op == ReduceOp::Mean {
        pointwise_code.push_str(&format!("float scaledVal = {input_ident} / stride;\n"));
        input_ident = "scaledVal";
    }

    let reduce_block_level = reduce_operation(reduce.op, "val", "localReduction[target]");
    let reduce_warp_level = reduce_operation(reduce.op, "val", "otherWarpVal");

    let code = formatdoc! {"
        #include <cuda_fp16.h>
        #include <cuda_bf16.h>
        #include <cuda_runtime.h>
        #include <math_constants.h>

        typedef unsigned int uint32_t;

        __device__ __forceinline__ float atomicMinFloat (float *addr, float value) {{
            float old;
            old = (value >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(value))) :
                __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));
    
            return old;
        }}

        __device__ __forceinline__ float atomicMaxFloat(float *addr, float value) {{
            float old;
            old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
                __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));
    
            return old;
        }}

        extern \"C\" __global__ void {KERNEL_NAME}({params_code} uint32_t stride, uint32_t totalSize) {{
            extern __shared__ float localReduction[];

            uint32_t totalIndex = threadIdx.x + blockDim.x * blockIdx.x;
            uint32_t strideRoundedUp = (stride + blockDim.x - 1) / blockDim.x * blockDim.x;

            uint32_t group = totalIndex / strideRoundedUp;
            uint32_t indexInGroup = totalIndex % strideRoundedUp;
            
            float val = {init_val};
            if (indexInGroup < stride) {{
                uint32_t index = group * stride + indexInGroup;
                {pointwise_code}
                val = {input_ident};
            }}

            // Block-level reduction
            for (int offset = blockDim.x / 2; offset >= 32; offset /= 2) {{
                localReduction[threadIdx.x] = val;
                __syncthreads();
                if (threadIdx.x < offset) {{
                    int target = threadIdx.x + offset;
                    val = {reduce_block_level};
                }}
            }}

            // Warp-level reduction
            __syncthreads();
            if (threadIdx.x < 32) {{
                uint32_t mask = 0xFFFFFFFF;
                for (int offset = 16; offset > 0; offset /= 2) {{
                    float otherWarpVal = __shfl_down_sync(mask, val, offset);
                    val = {reduce_warp_level};
                }}

                // Grid-level reduction
                if (threadIdx.x == 0) {{
                    {atomic_reduce}
                }}
            }}
        }}
    "};

    Kernel {
        params,
        code,
        entrypoint_name: KERNEL_NAME,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cuda::kernel::CompiledKernel,
        data_type::DataType,
        opgraph::{
            op::{
                BinaryPointwise, BinaryPointwiseOp, Op, Reduce, ReduceOp, UnaryPointwise,
                UnaryPointwiseOp,
            },
            subgraph::OpSubgraph,
            Descriptor, OpGraph, VarId,
        },
        shape::Shape,
    };
    use cudarc::driver::CudaDevice;
    use std::sync::Arc;

    #[test]
    fn test_generate_kernel() {
        let mut graph = OpGraph::new();

        let input1 = graph.new_input(Descriptor {
            data_type: DataType::F16,
            shape: Shape::new([1, 1]),
        });
        let input2 = graph.new_input(Descriptor {
            data_type: DataType::Bf16,
            shape: Shape::new([1, 1]),
        });

        let var = VarId::new();

        let a = graph.new_op(Op::UnaryPointwise(UnaryPointwise {
            input: input1,
            op: UnaryPointwiseOp::MulScalar(var),
        }));
        let b = graph.new_op(Op::BinaryPointwise(BinaryPointwise {
            lhs: a,
            rhs: input2,
            op: BinaryPointwiseOp::Add,
        }));

        let out = graph.new_op(Op::Reduce(Reduce {
            op: ReduceOp::Mean,
            input: b,
            depth: 2,
        }));
        graph.new_output(out);

        let _out2 = graph.new_output(b);

        let subgraph = OpSubgraph::from_nodes(&Arc::new(graph), vec![a, b, out]);

        let kernel = generate_kernel(&subgraph);
        insta::assert_snapshot!(kernel.code);

        CompiledKernel::new(&kernel, &CudaDevice::new(0).unwrap()).unwrap();
    }
}
