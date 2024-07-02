use crate::data_type::DataType;
use cudarc::cublaslt::sys::{
    cudaDataType,
    cudaDataType::{CUDA_R_16BF, CUDA_R_16F, CUDA_R_32F},
};

mod allocator;
pub mod context;
pub mod error;
mod execution;
mod kernel;
mod plan;
pub mod tensor;

fn cuda_data_type(data_type: DataType) -> cudaDataType {
    match data_type {
        DataType::F32 => CUDA_R_32F,
        DataType::Bf16 => CUDA_R_16BF,
        DataType::F16 => CUDA_R_16F,
    }
}
