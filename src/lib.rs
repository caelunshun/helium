#![feature(stdarch_x86_avx512, portable_simd)]

//pub mod autodiff;
#[cfg(feature = "cpu")]
pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod data_type;
pub mod device;
//pub mod dyn_tensor;
pub mod error;
//pub mod opgraph;
mod shape;
//pub mod tensor;

/*
#[doc(inline)]
pub use self::{
    autodiff::{AdTensor, Gradients, Param},
    data_type::DataType,
    device::Device,
    tensor::Tensor,
};*/
