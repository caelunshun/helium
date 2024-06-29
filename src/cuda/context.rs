use cudarc::driver::CudaDevice;
use std::sync::Arc;

/// Shared state for caching CUDA values on a particular device.
///
/// Can be cloned like an `Arc`.
#[derive(Clone)]
pub struct Cuda {
    device: Arc<CudaDevice>,
}
