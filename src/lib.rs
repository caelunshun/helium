#![feature(stdarch_x86_avx512, portable_simd)]

//pub mod autodiff;
mod backend;
#[cfg(feature = "cpu")]
pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod data_type;
pub mod device;
//pub mod dyn_tensor;
mod cache;
pub mod conv;
pub mod error;
pub mod module;
pub mod opgraph;
mod raw_tensor;
mod shape;
mod tensor;
mod thread_pool;

#[doc(inline)]
pub use self::{
    //autodiff::{AdTensor, Gradients, Param},
    data_type::DataType,
    device::Device,
    module::Module,
    tensor::{
        gradients::Gradients,
        param::{Param, ParamId},
        Tensor,
    },
};
/// Re-export of half-precision floating point types
/// from the `half` crate.
pub use half::{bf16, f16};
#[doc(inline)]
pub use helium_macros::Module;
