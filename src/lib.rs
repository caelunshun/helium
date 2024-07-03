#![feature(stdarch_x86_avx512, portable_simd)]

#[cfg(feature = "cpu")]
pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod data_type;
pub mod device;
pub mod error;
pub mod opgraph;
pub mod tensor;
