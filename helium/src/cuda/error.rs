use cudarc::{
    cudnn::CudnnError,
    driver::DriverError,
    nvrtc::{CompileError, result::NvrtcError},
};

#[derive(Debug, thiserror::Error)]
pub enum CudaError {
    #[error(transparent)]
    Driver(#[from] DriverError),
    #[error(transparent)]
    Nvrtc(#[from] NvrtcError),
    #[error(transparent)]
    CompileKernel(#[from] CompileError),
    #[error(transparent)]
    Cudnn(#[from] CudnnError),
    #[error("{0}")]
    Other(String),
}
