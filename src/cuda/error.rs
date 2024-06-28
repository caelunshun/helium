use cudarc::{
    cublaslt::result::CublasError,
    cudnn::CudnnError,
    driver::DriverError,
    nvrtc::{result::NvrtcError, CompileError},
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
    Cublas(#[from] CublasError),
    #[error(transparent)]
    Cudnn(#[from] CudnnError),
}
