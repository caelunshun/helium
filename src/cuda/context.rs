use crate::cuda::{allocator::CudaAllocator, cudnn::CudnnContext, error::CudaError};
use cudarc::{
    driver,
    driver::{sys::CUstream, CudaDevice},
};
use parking_lot::{Mutex, MutexGuard, RwLock};
use std::{
    iter,
    sync::{Arc, OnceLock},
};
use thread_local::ThreadLocal;

/// Shared state for caching CUDA values on a particular device.
pub struct CudaContext {
    device: Arc<CudaDevice>,
    allocator: Mutex<CudaAllocator>,
    stream_pool: ThreadLocal<Vec<CudaStream>>,
    cudnn_pool: ThreadLocal<CudnnContext>,
}

impl CudaContext {
    pub fn global(device_index: u32) -> Result<&'static Self, CudaError> {
        static CONTEXTS: OnceLock<RwLock<Vec<Option<&'static CudaContext>>>> = OnceLock::new();
        let lock = CONTEXTS.get_or_init(Default::default);

        // Optimistic check with read-only lock
        let guard = lock.read();
        if let Some(Some(cx)) = guard.get(device_index as usize) {
            return Ok(*cx);
        }

        drop(guard);
        let mut guard = lock.write();
        if let Some(Some(cx)) = guard.get(device_index as usize) {
            return Ok(*cx);
        }

        let needed_space = device_index as usize + 1 - guard.len();
        guard.extend(iter::repeat(None).take(needed_space));
        Ok(guard[device_index as usize].insert(Box::leak(Box::new(Self::new(device_index)?))))
    }

    pub fn new(device_index: u32) -> Result<Self, CudaError> {
        let device = CudaDevice::new_with_stream(device_index as usize)?;

        let allocator = unsafe { CudaAllocator::new(*device.cu_primary_ctx()) };

        Ok(Self {
            device,
            allocator: Mutex::new(allocator),
            stream_pool: ThreadLocal::new(),
            cudnn_pool: ThreadLocal::new(),
        })
    }

    pub fn stream_pool(&self) -> Result<&[CudaStream], CudaError> {
        const STREAMS_PER_THREAD: usize = 4;
        self.stream_pool
            .get_or_try(|| {
                let mut streams = Vec::new();
                for _ in 0..STREAMS_PER_THREAD {
                    streams.push(CudaStream::new(&self.device)?);
                }
                Ok(streams)
            })
            .map(Vec::as_slice)
    }

    pub fn cudnn_handle(&self) -> &CudnnContext {
        self.cudnn_pool.get_or(|| {
            self.device.bind_to_thread().unwrap();
            CudnnContext::new().expect("failed to init cuDNN")
        })
    }

    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    pub fn allocator(&self) -> MutexGuard<CudaAllocator> {
        self.allocator.lock()
    }
}

/// Wrapper for a CUDA stream.
#[derive(Debug)]
pub struct CudaStream {
    stream: cudarc::driver::CudaStream,
}

impl CudaStream {
    pub fn new(device: &Arc<CudaDevice>) -> Result<Self, CudaError> {
        let stream = device.fork_default_stream()?;
        Ok(Self { stream })
    }

    pub fn raw(&self) -> CUstream {
        self.stream.stream
    }

    pub fn cudarc_stream(&self) -> &driver::CudaStream {
        &self.stream
    }
}

unsafe impl Send for CudaStream {}
