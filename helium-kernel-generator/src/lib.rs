#![feature(once_cell_try)]

pub mod architecture;
mod builder;
mod cute;
pub mod error;
pub mod generators;
mod pointwise;

pub use error::Error;
