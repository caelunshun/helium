use crate::{
    cuda::kernel::{cpp_type_name, Kernel, KernelParam},
    opgraph::{
        op,
        op::{BroadcastAxis, RestrctureOp},
        Descriptor, NodeId,
    },
};
use indoc::formatdoc;

pub fn generate_kernel(
    op: &op::Restructure,
    node: NodeId,
    input_descriptor: &Descriptor,
) -> Kernel {
    let dtype = cpp_type_name(input_descriptor.data_type);

    let index_mapping = match &op.op {
        RestrctureOp::BroadcastAxis { axis, new_size } => match *axis {
            BroadcastAxis::Existing(axis) => {
                let stride = input_descriptor.shape.dims()[axis..]
                    .iter()
                    .product::<usize>();
                let out_stride = stride * *new_size;
                format!("uint32_t srcIdx = globalIdx % {stride} + (globalIdx / {out_stride} * {stride});")
            }
            BroadcastAxis::Expand => {
                let stride = input_descriptor.shape.num_elements();
                format!("uint32_t srcIdx = globalIdx % {stride};")
            }
        },
    };

    let code = formatdoc! {"
        #include <cuda_fp16.h>
        #include <cuda_bf16.h>

        typedef unsigned int uint32_t;

        extern \"C\" __global__ void generatedRestructureKernel({dtype} *in, uint32_t size, {dtype} *out) {{
            uint32_t globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
            if (globalIdx >= size) return;
            {index_mapping}
            out[globalIdx] = in[srcIdx];
        }}
    "};

    Kernel {
        code,
        params: vec![
            KernelParam::Node(op.input),
            KernelParam::Size,
            KernelParam::Output(node),
        ],
        entrypoint_name: "generatedRestructureKernel",
    }
}
