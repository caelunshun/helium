use crate::{
    cuda::{
        cudnn::compute_packed_strides,
        kernel_jit::{KernelBuilder, KernelParam},
    },
    opgraph::NodeId,
    shape::Shape,
    DataType,
};
use indoc::formatdoc;
use std::fmt::{Display, Write};

pub const BLOCK_SIZE: [usize; 2] = [32, 32];

pub fn generate_kernel(
    shape: &Shape,
    out_strides: &[usize],
    in_id: NodeId,
    out_id: NodeId,
    data_type: DataType,
) -> KernelBuilder {
    assert_eq!(out_strides.len(), shape.num_dims());
    assert!(shape.num_elements() < u32::MAX as usize);

    let mut kernel = KernelBuilder::new();

    let input = kernel.param(KernelParam::Input(in_id), data_type);
    let output = kernel.param(KernelParam::Output(out_id), data_type);

    let num_dims = shape.num_dims();

    let dims_array = format_cpp_array(shape.dims());
    let in_strides_array = format_cpp_array(&compute_packed_strides(shape));
    let out_strides_array = format_cpp_array(out_strides);
    let dtype = KernelBuilder::cpp_data_type(data_type);

    let total_count = shape.num_elements();

    kernel.statement(formatdoc! {"
        // 33 instead of 32 is intentional to reduce bank conflicts
        __shared__ float values[33 * 32];
        __shared__ uint32_t outIndexes[33 * 32];

        uint32_t globalX = blockDim.x * blockIdx.x + threadIdx.x;
        uint32_t globalY = blockDim.y * blockIdx.y + threadIdx.y;

        uint32_t dims[{num_dims}] = {dims_array};
        uint32_t in_strides[{num_dims}] = {in_strides_array};
        uint32_t out_strides[{num_dims}] = {out_strides_array};

        uint32_t coords[{num_dims}];
        coords[{num_dims} - 1] = globalX;
        bool inBounds = globalX < dims[{num_dims} - 1];

        for (int i = 0; i < {num_dims} - 1; i++) {{
            uint32_t stride = out_strides[i];
            if (stride > out_strides[{num_dims} - 1]) {{
                stride /= dims[{num_dims} - 1];
            }}
            coords[i] = (globalY / stride) % dims[i];
            inBounds &= coords[i] < dims[i];
        }}

        uint32_t outIndex = 0;
        uint32_t inIndex = 0;
        for (int i = 0; i < {num_dims}; i++) {{
            outIndex += out_strides[i] * coords[i];
            inIndex += in_strides[i] * coords[i];
        }}

        if (inBounds) {{
            outIndexes[threadIdx.x + 33 * threadIdx.y] = outIndex;
        }} else {{
            outIndexes[threadIdx.x + 33 * threadIdx.y] = 0xFFFFFFFF;
        }}

        float val = 0;
        if (inBounds) {{
            val = static_cast<float>({input}[inIndex]);
        }}
        values[threadIdx.x + 33 * threadIdx.y] = val;
    
        __syncthreads();
    
        uint32_t peerOutIndex = outIndexes[threadIdx.y + 33 * threadIdx.x];
        float peerOutVal = values[threadIdx.y + 33 * threadIdx.x];
        if (peerOutIndex < {total_count}) {{
            {output}[peerOutIndex] = static_cast<{dtype}>(peerOutVal);
        }}
    "});

    kernel
}

fn format_cpp_array<T: Display>(slice: &[T]) -> String {
    let mut s = "{".to_owned();
    for (i, val) in slice.iter().enumerate() {
        write!(s, "{val}").unwrap();
        if i != slice.len() - 1 {
            s.push_str(", ");
        }
    }
    s.push('}');
    s
}

pub fn compute_grid_size(shape: &Shape, _out_strides: &[usize]) -> [usize; 2] {
    let x = shape.dim_at(-1).div_ceil(BLOCK_SIZE[0]);
    let y = shape.dims()[..shape.num_dims() - 1]
        .iter()
        .copied()
        .product::<usize>()
        .div_ceil(BLOCK_SIZE[1]);
    [x, y]
}
