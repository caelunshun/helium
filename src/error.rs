pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[cfg(feature = "cuda")]
    #[error(transparent)]
    Cuda(#[from] crate::cuda::error::CudaError),
}
