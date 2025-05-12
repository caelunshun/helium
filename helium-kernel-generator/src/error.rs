use std::io;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error(transparent)]
    Nvrtc(#[from] cudarc::nvrtc::CompileError),
    #[error(transparent)]
    Cuda(#[from] cudarc::driver::DriverError),
}
