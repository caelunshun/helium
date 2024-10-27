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

pub const BLOCK_SIZE: [usize; 2] = [32, 8];

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

    let total_y_grid = shape.dims()[..shape.num_dims() - 1]
        .iter()
        .copied()
        .product::<usize>();

    kernel.statement(formatdoc! {"
        // Note: each thread handles 4 elements (strided by 8=blockDim.y along Y axis)
        // to increase occupancy and get better ILP.
        // We also use grid-stride loop pattern to support larger
        // tensors over the Y dimension (max CUDA grid size is 65,535 in Y dimension).

        // 33 instead of 32 is intentional to reduce bank conflicts
        __shared__ {dtype} values[33 * 32];
        __shared__ uint32_t outIndexes[33 * 32];

        uint32_t dims[{num_dims}] = {dims_array};
        uint32_t in_strides[{num_dims}] = {in_strides_array};
        uint32_t out_strides[{num_dims}] = {out_strides_array};

        for (uint32_t offsetY = 32 * blockIdx.y; offsetY < {total_y_grid}; offsetY += 32 * gridDim.y) {{
            for (int dy = 0; dy < 4; dy++) {{
                uint32_t localX = threadIdx.x;
                uint32_t localY = threadIdx.y + 8 * dy;

                uint32_t globalX = blockDim.x * blockIdx.x + localX;
                uint32_t globalY = offsetY + localY;

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
                    outIndexes[localX + 33 * localY] = outIndex;
                }} else {{
                    outIndexes[localX + 33 * localY] = 0xFFFFFFFF;
                }}
        
                {dtype} val = static_cast<{dtype}>(0);
                if (inBounds) {{
                    val = {input}[inIndex];
                }}
                values[localX + 33 * localY] = val;
            }}
        
            __syncthreads();
    
            for (int dy = 0; dy < 4; dy++) {{
                uint32_t localX = threadIdx.x;
                uint32_t localY = threadIdx.y + 8 * dy;
    
                uint32_t peerOutIndex = outIndexes[localY + 33 * localX];
                {dtype} peerOutVal = values[localY + 33 * localX];
                if (peerOutIndex < {total_count}) {{
                    {output}[peerOutIndex] = peerOutVal;
                }}
            }}

            __syncthreads();
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
    /// CUDA limit is 65,535, but smaller value may get
    /// better performance due to thread reuse.
    const MAX_Y_GRID_SIZE: usize = 16384;

    let x = shape.dim_at(-1).div_ceil(BLOCK_SIZE[0]);
    let y = shape.dims()[..shape.num_dims() - 1]
        .iter()
        .copied()
        .product::<usize>()
        .div_ceil(BLOCK_SIZE[1])
        .min(MAX_Y_GRID_SIZE);
    [x, y]
}
